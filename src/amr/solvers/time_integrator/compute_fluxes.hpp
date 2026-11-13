#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"

#include "initializer/data_provider.hpp"

#include <cassert>
#include "core/numerics/differential_operators.hpp"
#include "core/numerics/constrained_transport/poynting_correction.hpp"
#include "core/numerics/primite_conservative_converter/face_energy.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <vector>
#include <optional>
#include <string>
#include <array>
#include <stdexcept>

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel>
class ComputeFluxes
{
    using level_t = MHDModel::level_t;
    // using Layout        = MHDModel::gridlayout_type;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using Ampere_t                   = Dispatchers_t::Ampere_t;
    using DissipativeElectricField_t = Dispatchers_t::DissipativeElectricField_t;

    using FVMethod_t     = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;
    using FVMethodInfo_t = FVMethod_t::info_type;

    constexpr static auto Hall = FVMethod_t::Hall;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t     = Dispatchers_t::template ConstrainedTransport_t<Rec, Hall>;
    using ConstrainedTransportInfo_t = ConstrainedTransport_t::info_type;

    using PoyntingCorrection_t = Dispatchers_t::template PoyntingCorrection_t<Rec>;

    using ToPrimitiveConverter_t    = Dispatchers_t::ToPrimitiveConverter_t;
    using ToConservativeConverter_t = Dispatchers_t::ToConservativeConverter_t;

    using VecField = MHDModel::vecfield_type;


public:
    ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , to_primitive_gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
        , to_conservative_gamma_{dict["to_conservative"]["heat_capacity_ratio"]}
        , needsCurrent_{Hall || fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()}
    {
        // eta/nu are duplicated into both info structs
        assert(fVMethodInfo_.eta == constrainedTransportInfo_.eta
               && fVMethodInfo_.nu == constrainedTransportInfo_.nu);

        // hyper-resistivity is only meaningful alongside the Hall term
        if constexpr (!Hall)
            if (fVMethodInfo_.isHyperResistive())
                throw std::runtime_error(
                    "Error - hyper-resistivity (nu > 0) requires the Hall term to be enabled");
    }

    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const newTime)
    {
        ToPrimitiveConverter_t{level, model}(state, to_primitive_gamma_);
        if (pminOn_ && newTime >= budgetT0_)
            pminProbe_(model, state, level, newTime);
        TimeSetter{level, model, newTime}(state.rho, state.rhoV, state.Etot, state.V, state.P);

        if (needsCurrent_)
        {
            Ampere_t{level, model}(state.B, state.J);
            TimeSetter{level, model, newTime}(state.B, state.J);
        }

        if (isDissipative_)
            DissipativeElectricField_t{level, model, fVMethodInfo_}(dissipative_electric_state_,
                                                                    state);

        FVMethod_t{level, model, fVMethodInfo_}(ct_, state, fluxes);
        TimeSetter{level, model, newTime}(state.rho, state.V, state.P, state.J);

        // unecessary if we decide to store both primitive and conservative variables
        ToConservativeConverter_t{level, model}(state, to_conservative_gamma_);
        TimeSetter{level, model, newTime}(state.rho, state.V, state.P, state.rhoV, state.Etot);

        if (closureEvery_ > 0 && (closureCalls_++ % closureEvery_) == 0 && newTime >= closureT0_)
            closureProbe_(model, state, fluxes, level, newTime);

        if (budgetOn_ && newTime >= budgetT0_)
            budgetProbe_(model, state, fluxes, level, newTime);

        if (Hall && (useHallFlux_ || useHallFluxFace_))
        {
            // experiment PHARE_HALL_FLUX=1 (zwF): Hall out of the Poynting flux as in the
            // Hall-neutral path, but its energy goes in flux form F_H (Q = -div F_H exactly,
            // FH/derivation.md), added to Etot_fx/fy so the RK weights and reflux carry it.
            if (fhCheckEvery_ > 0 && (fhCheckCalls_++ % fhCheckEvery_) == 0)
            {
                ConstrainedTransport_t{level, model, constrainedTransportInfo_,
                                       4}(ct_, dissipative_electric_state_, state);
                if (useHallFluxFace_)
                    fhCheckFace_(model, state, level, newTime,
                                 true); // FF-2 (+ CT Poynting control)
                else
                    fhCheck_(model, state, level, newTime);
                fhCheckNonHall_ = useHallFluxFace_;
            }
            ConstrainedTransport_t{level, model, constrainedTransportInfo_,
                                   1 | 2 | 8}(ct_, dissipative_electric_state_, state);
            bc.fillElectricGhosts(state.E, level, newTime);
            if (fhCheckNonHall_) // FF-1 non-Hall part: Q_face + div P(E) + w_sym(E) = 0
            {
                fhCheckFace_(model, state, level, newTime, false);
                fhCheckNonHall_ = false;
            }
            if (ctpCentralMask_ >= 0)
                poyntingByPart_(model, state, fluxes, bc, level, newTime);
            else
                PoyntingCorrection_t{level, model}(ct_, state, fluxes);
            hallFlux_(model, state, fluxes, level);
            ConstrainedTransport_t{level, model, constrainedTransportInfo_}(
                ct_, dissipative_electric_state_, state);
            bc.fillElectricGhosts(state.E, level, newTime);
            return;
        }

        if (Hall && hallNeutral_)
        {
            // experiment: Hall does no work, so keep its magnetic-energy change out of p.
            // Etot gets Q = -B_cc.(curl E_H)_cc instead of -div(E_H x B); Q rides in B_fx.z
            // (unused by the FV update) so the RK stage weights apply to it unchanged.
            ConstrainedTransport_t{level, model, constrainedTransportInfo_,
                                   4}(ct_, dissipative_electric_state_, state);
            hallSource_(model, state, fluxes, level);
            ConstrainedTransport_t{level, model, constrainedTransportInfo_,
                                   1 | 2 | 8}(ct_, dissipative_electric_state_, state);
            bc.fillElectricGhosts(state.E, level, newTime);
            PoyntingCorrection_t{level, model}(ct_, state, fluxes);
            ConstrainedTransport_t{level, model, constrainedTransportInfo_}(
                ct_, dissipative_electric_state_, state);
            bc.fillElectricGhosts(state.E, level, newTime);
            return;
        }

        if (ctpCentralMask_ >= 0)
            poyntingByPart_(model, state, fluxes, bc, level, newTime);

        ConstrainedTransport_t{level, model,
                               constrainedTransportInfo_}(ct_, dissipative_electric_state_, state);

        bc.fillElectricGhosts(state.E, level, newTime);

        if (ctpCentralMask_ < 0)
            PoyntingCorrection_t{level, model}(ct_, state, fluxes);
    }

    // experiment PHARE_CTP_CENTRAL_MASK=<bits> (ctpsurprise): P is linear in E, so apply it part
    // by part (1 advective, 2 fan, 8 eta/nu), with the central edge B for the parts in the mask
    // and the upwind edge B for the others. Leaves state.E holding the last part only.
    void poyntingByPart_(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                         double const newTime)
    {
        using PC_t       = core::PoyntingCorrection<typename MHDModel::gridlayout_type, Rec>;
        bool const saved = PC_t::central_;
        for (int const part : {1, 2, 8})
        {
            ConstrainedTransport_t{level, model, constrainedTransportInfo_,
                                   part}(ct_, dissipative_electric_state_, state);
            bc.fillElectricGhosts(state.E, level, newTime);
            PC_t::central_ = (ctpCentralMask_ & part) != 0;
            PoyntingCorrection_t{level, model}(ct_, state, fluxes);
        }
        PC_t::central_ = saved;
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(ct_);
        model.resourcesManager->registerResources(dissipative_electric_state_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(ct_, patch, allocateTime);
        model.resourcesManager->allocate(dissipative_electric_state_, patch, allocateTime);
    }

private:
    // experiment: energy-closure probe, PHARE_CLOSURE_DIAG=<every N flux evaluations>.
    // For each part P of the edge E (1 advective, 2 fan diffusion, 4|8 Hall + explicit),
    // R_P = -div(E_P x B_edge) + B_cc . (curl E_P)_cc is the rate at which that part changes
    // p/(gamma-1) = Etot - KE - B_cc^2/2. Linear in E, so the parts sum to the scheme's total.
    void closureProbe_(MHDModel& model, auto& state, auto& fluxes, level_t& level, double const time)
    {
        if constexpr (true) // 1D, 2D, 3D
        {
            using GridLayout   = MHDModel::gridlayout_type;
            using PC           = core::PoyntingCorrection<GridLayout, Rec>;
            constexpr auto dim = MHDModel::dimension;
            using core::Component;
            using core::Direction;
            using Idx = core::MeshIndex<dim>;

            int const masks[4] = {1, 2, 4, 8}; // advective, fan, Hall, explicit eta/nu
            std::vector<std::vector<double>> R[4], eth, xs, ys, zs, gs, gEs, Ralt;
            // experiment PHARE_PROBE_ALTP=1 (ctpsurprise part 3): also the advective rate with the other
            // P form (upwind <-> consistent) on the same state, printed as Aalt
            static bool const altP = std::getenv("PHARE_PROBE_ALTP") != nullptr;
            double absW[4] = {0, 0, 0, 0}; // sum_c |B_cc . C(curl E_P)|: scale of each part's work
            auto& rm       = *model.resourcesManager;

            for (int k = 0; k < 4; ++k)
            {
                ConstrainedTransport_t{level, model, constrainedTransportInfo_,
                                       masks[k]}(ct_, dissipative_electric_state_, state);
                std::size_t ip = 0;
                for (auto& patch : rm.enumerate(level, ct_, state, fluxes))
                {
                    auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                    PC pc{layout};
                    auto const& B   = state.B;
                    auto const& E   = state.E;
                    auto const curl = core::curl(layout, E);
                    struct C
                    {
                        using type = double;
                        std::function<double(Idx)> f;
                        double operator()(Idx i) const { return f(i); }
                    };
                    C cx{[&](Idx i) { return curl.x(i); }}, cy{[&](Idx i) { return curl.y(i); }},
                        cz{[&](Idx i) { return curl.z(i); }};
                    if (k == 0)
                    {
                        eth.emplace_back();
                        xs.emplace_back();
                        ys.emplace_back();
                        zs.emplace_back();
                        gs.emplace_back();
                        gEs.emplace_back();
                    }
                    R[k].emplace_back();
                    if (k == 0)
                        Ralt.emplace_back();
                    // zwF mode: the Hall part's energy flux is the zero-work F_H (built once per patch)
                    using ZW_t = decltype(core::make_zero_work_hall(B, state.rho, layout));
                    std::optional<ZW_t> zwp;
                    if (k == 2 && useHallFlux_ && !useHallFluxFace_)
                        zwp.emplace(core::make_zero_work_hall(B, state.rho, layout));
                    layout.evalOnBox(state.rho, [&](auto... ii) {
                        Idx const c{static_cast<std::size_t>(ii)...};
                        double divS = 0, divSalt = 0;
                        core::for_N<dim>([&](auto dd) {
                            constexpr auto dir = static_cast<Direction>(dd());
                            Idx cr             = c;
                            cr[dd()] += 1;
                            double sl = 0, sr = 0;
                            pc.template poynting_energy_flux_<dir>(ct_, E, B, c, sl);
                            pc.template poynting_energy_flux_<dir>(ct_, E, B, cr, sr);
                            divS += (sr - sl) * layout.inverseMeshSize(dir);
                            if (k == 0 && altP)
                            {
                                PC::consistent_ = !PC::consistent_;
                                double al = 0, ar = 0;
                                pc.template poynting_energy_flux_<dir>(ct_, E, B, c, al);
                                pc.template poynting_energy_flux_<dir>(ct_, E, B, cr, ar);
                                divSalt += (ar - al) * layout.inverseMeshSize(dir);
                                PC::consistent_ = !PC::consistent_;
                            }
                        });
                        if (zwp)
                        {
                            divS = 0;
                            core::for_N<dim>([&](auto dd) {
                                constexpr std::size_t d = dd();
                                Idx cr                  = c;
                                cr[d] += 1;
                                auto at = [&](Idx const& q) {
                                    if constexpr (dim == 1)
                                        return core::hall_energy_flux_at<d>(*zwp, q[0]);
                                    else if constexpr (dim == 2)
                                        return core::hall_energy_flux_at<d>(*zwp, q[0], q[1]);
                                    else
                                        return core::hall_energy_flux_at<d>(*zwp, q[0], q[1], q[2]);
                                };
                                divS += (at(cr) - at(c))
                                        * layout.inverseMeshSize(static_cast<Direction>(d));
                            });
                        }
                        // face-flux mode: the Hall part's energy flux is F_H^sym, not P(E_H)
                        if constexpr (dim >= 2)
                            if (k == 2 && useHallFluxFace_)
                                divS = faceFluxDiv_(core::make_face_pair_flux(B, state.rho, layout),
                                                    layout, c);
                        using implT  = GridLayout::implT;
                        double const bx = GridLayout::template project<implT::faceXToCellCenter>(
                            B(Component::X), c);
                        double const by = GridLayout::template project<implT::faceYToCellCenter>(
                            B(Component::Y), c);
                        double const bz = GridLayout::template project<implT::faceZToCellCenter>(
                            B(Component::Z), c);
                        double const cbx
                            = GridLayout::template project<implT::faceXToCellCenter>(cx, c);
                        double const cby
                            = GridLayout::template project<implT::faceYToCellCenter>(cy, c);
                        double const cbz
                            = GridLayout::template project<implT::faceZToCellCenter>(cz, c);
                        // with PHARE_P_FACE_ENERGY the magnetic energy is the average of squares,
                        // so its cell rate is the face->cell average of B_f . (curl E)_f, not B_cc
                        // . C(curl E)
                        double work = bx * cbx + by * cby + bz * cbz;
                        if (core::useFaceEnergy())
                        {
                            C px{[&](Idx i) { return B(Component::X)(i) * curl.x(i); }},
                                py{[&](Idx i) { return B(Component::Y)(i) * curl.y(i); }},
                                pz{[&](Idx i) { return B(Component::Z)(i) * curl.z(i); }};
                            work = GridLayout::template project<implT::faceXToCellCenter>(px, c)
                                   + GridLayout::template project<implT::faceYToCellCenter>(py, c)
                                   + GridLayout::template project<implT::faceZToCellCenter>(pz, c);
                        }
                        R[k].back().push_back(-divS + work);
                        if (k == 0)
                            Ralt.back().push_back(-divSalt + work);
                        absW[k] += std::abs(work);
                        if (k == 0)
                        {
                            eth.back().push_back(state.P(ii...) / (to_primitive_gamma_ - 1.));
                            // G: p rate of the Godunov fluxes (here before Poynting / F_H are added):
                            // -div F_E + v.div F_rhoV - v^2/2 div F_rho, KE = (rhoV)^2 / (2 rho)
                            double g = 0, gE = 0;
                            double const v[3] = {state.V(Component::X)(c), state.V(Component::Y)(c),
                                                 state.V(Component::Z)(c)};
                            core::for_N<dim>([&](auto dd) {
                                constexpr auto dir = static_cast<Direction>(dd());
                                Idx cr             = c;
                                cr[dd()] += 1;
                                auto const fl = fluxes.template get_dir<dir>(c);
                                auto const fr = fluxes.template get_dir<dir>(cr);
                                double const v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                                gE += -(fr.Etot() - fl.Etot()) * layout.inverseMeshSize(dir);
                                g += (-(fr.Etot() - fl.Etot()) + v[0] * (fr.V.x - fl.V.x)
                                      + v[1] * (fr.V.y - fl.V.y) + v[2] * (fr.V.z - fl.V.z)
                                      - 0.5 * v2 * (fr.rho - fl.rho))
                                     * layout.inverseMeshSize(dir);
                            });
                            gs.back().push_back(g);
                            gEs.back().push_back(gE);
                            // cellCenteredCoordinates takes an AMR index (origin 0), not a local one
                            auto const xy = layout.cellCenteredCoordinates(layout.localToAMR(
                                core::Point<int, dim>{static_cast<int>(ii)...}));
                            xs.back().push_back(xy[0]);
                            ys.back().push_back(dim >= 2 ? xy[dim >= 2 ? 1 : 0] : 0.);
                            zs.back().push_back(dim == 3 ? xy[dim == 3 ? 2 : 0] : 0.);
                        }
                    });
                    ++ip;
                }
            }

            // cold cells only (p < 0.5): min of each rate relative to e_th, the components at
            // the worst total, and level sums (sum R_P dV = minus the discrete work of E_P)
            double mn[4] = {1e300, 1e300, 1e300, 1e300}, mt = 1e300, at[4] = {0, 0, 0, 0}, wx = 0,
                   wy = 0, wz = 0, wp = 0;
            double sum[4] = {0, 0, 0, 0};
            double atN[4] = {0, 0, 0, 0}, nx = 0, ny = 0; // all parts at the worst-nu cell
            // all parts at the min-p cell (any p): the most violating cell of this rank
            double pmn = 1e300, atP[4] = {0, 0, 0, 0}, px = 0, py = 0, pz = 0, atG = 0, atGE = 0, atAlt = 0;
            for (std::size_t ip = 0; ip < eth.size(); ++ip)
                for (std::size_t c = 0; c < eth[ip].size(); ++c)
                {
                    for (int k = 0; k < 4; ++k)
                        sum[k] += R[k][ip][c];
                    if (eth[ip][c] < pmn)
                    {
                        pmn = eth[ip][c];
                        for (int k = 0; k < 4; ++k)
                            atP[k] = R[k][ip][c] / eth[ip][c];
                        atG  = gs[ip][c] / eth[ip][c];
                        atGE = gEs[ip][c] / eth[ip][c];
                        atAlt = Ralt[ip][c] / eth[ip][c];
                        px = xs[ip][c];
                        py = ys[ip][c];
                        pz = zs[ip][c];
                    }
                    if (eth[ip][c] * (to_primitive_gamma_ - 1.) > 0.5)
                        continue;
                    double tot = 0;
                    if (R[3][ip][c] / eth[ip][c] < mn[3])
                    {
                        for (int k = 0; k < 4; ++k)
                            atN[k] = R[k][ip][c] / eth[ip][c];
                        nx = xs[ip][c];
                        ny = ys[ip][c];
                    }
                    for (int k = 0; k < 4; ++k)
                    {
                        double const r = R[k][ip][c] / eth[ip][c];
                        mn[k]          = std::min(mn[k], r);
                        tot += r;
                    }
                    if (tot < mt)
                    {
                        mt = tot;
                        for (int k = 0; k < 4; ++k)
                            at[k] = R[k][ip][c] / eth[ip][c];
                        wx = xs[ip][c];
                        wy = ys[ip][c];
                        wz = zs[ip][c];
                        wp = eth[ip][c] * (to_primitive_gamma_ - 1.);
                    }
                }
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            std::fprintf(stderr,
                         "CLOSURE r=%s t=%.6e minA=%.3e minD=%.3e minH=%.3e | worst tot=%.3e "
                         "A=%.3e D=%.3e H=%.3e @(%.3f,%.3f) | sumA=%.3e sumD=%.3e sumH=%.3e "
                         "z=%.3f p=%.4e\n",
                         rk ? rk : "0", time, mn[0], mn[1], mn[2], mt, at[0], at[1], at[2], wx, wy,
                         sum[0], sum[1], sum[2], wz, wp);
            std::fprintf(stderr,
                         "CLOSUREN r=%s t=%.6e minN=%.3e @(%.3f,%.3f) A=%.3e D=%.3e H=%.3e | "
                         "sumN=%.3e absWH=%.3e\n",
                         rk ? rk : "0", time, mn[3], nx, ny, atN[0], atN[1], atN[2], sum[3],
                         absW[2]);
            std::fprintf(stderr,
                         "CLOSUREP r=%s t=%.6e pmin=%.6e @(%.3f,%.3f,%.3f) A=%.3e D=%.3e H=%.3e "
                         "N=%.3e G=%.3e GE=%.3e GK=%.3e Aalt=%.3e\n",
                         rk ? rk : "0", time, pmn * (to_primitive_gamma_ - 1.), px, py, pz, atP[0],
                         atP[1], atP[2], atP[3], atG, atGE, atG - atGE, altP ? atAlt : 0.);
            // experiment PHARE_CLOSURE_CELLS="x,y,z;x,y,z": all columns at fixed cells (centre coordinates)
            static std::vector<std::array<double, 3>> const fixedCells = [] {
                std::vector<std::array<double, 3>> v;
                auto const* e = std::getenv("PHARE_CLOSURE_CELLS");
                if (!e)
                    return v;
                std::string str{e};
                std::size_t pos = 0;
                while (pos < str.size())
                {
                    auto const end = str.find(';', pos);
                    std::array<double, 3> c{0, 0, 0};
                    std::sscanf(str.substr(pos, end - pos).c_str(), "%lf,%lf,%lf", &c[0], &c[1], &c[2]);
                    v.push_back(c);
                    if (end == std::string::npos)
                        break;
                    pos = end + 1;
                }
                return v;
            }();
            for (auto const& fc : fixedCells)
                for (std::size_t ip = 0; ip < eth.size(); ++ip)
                    for (std::size_t c = 0; c < eth[ip].size(); ++c)
                        if (std::abs(xs[ip][c] - fc[0]) < 1e-3 && std::abs(ys[ip][c] - fc[1]) < 1e-3
                            && std::abs(zs[ip][c] - fc[2]) < 1e-3)
                        {
                            double const e = eth[ip][c];
                            std::fprintf(stderr,
                                         "CLOSUREC r=%s t=%.6e p=%.6e @(%.3f,%.3f,%.3f) A=%.3e D=%.3e "
                                         "H=%.3e N=%.3e G=%.3e GE=%.3e GK=%.3e Aalt=%.3e\n",
                                         rk ? rk : "0", time, e * (to_primitive_gamma_ - 1.), fc[0],
                                         fc[1], fc[2], R[0][ip][c] / e, R[1][ip][c] / e,
                                         R[2][ip][c] / e, R[3][ip][c] / e, gs[ip][c] / e,
                                         gEs[ip][c] / e, (gs[ip][c] - gEs[ip][c]) / e,
                                         altP ? Ralt[ip][c] / e : 0.);
                        }
        }
    }

    // experiment (ctpsurprise) PHARE_CTP_PMIN=1: per flux evaluation (RK stage), min p of the
    // stage input on this rank, where, and the count of non-finite p
    void pminProbe_(MHDModel& model, auto& state, level_t& level, double const time)
    {
        using GridLayout = MHDModel::gridlayout_type;
        double pmin = 1e300, px = 0, py = 0;
        std::size_t nbad = 0;
        for (auto& patch : model.resourcesManager->enumerate(level, state))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            layout.evalOnBox(state.P, [&](auto... ii) {
                double const p = state.P(ii...);
                if (!std::isfinite(p))
                    ++nbad;
                if (p < pmin)
                {
                    pmin          = p;
                    auto const xy = layout.cellCenteredCoordinates(layout.localToAMR(core::Point<int, MHDModel::dimension>{static_cast<int>(ii)...}));
                    px            = xy[0];
                    if constexpr (MHDModel::dimension >= 2)
                        py = xy[1];
                }
            });
        }
        auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
        std::fprintf(stderr, "PMIN r=%s t=%.9e call=%zu pmin=%.6e @(%.3f,%.3f) nonfinite=%zu\n",
                     rk ? rk : "0", time, pminCalls_++, pmin, px, py, nbad);
    }
    std::size_t pminCalls_ = 0;

    // experiment (ctpsurprise) PHARE_CTP_BUDGET="x0,y0,W" (+ PHARE_CTP_BUDGET_T0): per flux
    // evaluation, for each cell within W cells of (x0, y0), the semi-discrete rate of
    // e_int = Etot - KE - e_mag split as
    //   Hyd  = -div F_E(hydro) - (KEdot - L)       (all non-magnetic terms, incl. Rusanov diss.)
    //   L    = v . (-div M), M = the magnetic stress in the Godunov momentum flux
    //   R_k  = -div P(E_k) + work(E_k), for k = 1 advective, 2 fan, 4 Hall (F_H if flux mode),
    //          with P built on the central (c) and the upwind (u) edge B, on the same state
    // so that e_int rate = Hyd - L + sum_k R_k, under either P form. Probe only: leaves state.E
    // overwritten (ComputeFluxes recomputes it after).
    void budgetProbe_(MHDModel& model, auto& state, auto& fluxes, level_t& level, double const time)
    {
        if constexpr (MHDModel::dimension == 2)
        {
            using GridLayout = MHDModel::gridlayout_type;
            using PC         = core::PoyntingCorrection<GridLayout, Rec>;
            using Rtor       = core::Reconstructor<Rec<GridLayout>>;
            using core::Component;
            using core::Direction;
            using Idx   = core::MeshIndex<2>;
            using I     = std::size_t;
            using implT = GridLayout::implT;
            auto& rm    = *model.resourcesManager;

            struct Row
            {
                std::size_t ip;
                I i, j;
                double x, y, eth, hyd, L, Rc[3], Ru[3], Rk[3];
            };
            std::vector<Row> rows;

            std::size_t ip = 0;
            for (auto& patch : rm.enumerate(level, state, fluxes))
            {
                auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                double const ix   = layout.inverseMeshSize(Direction::X);
                double const iy   = layout.inverseMeshSize(Direction::Y);
                double const dx = 1. / ix, dy = 1. / iy;
                auto const& B   = state.B;
                auto stress     = [&](auto const& b, int dir) { // magnetic part of the momentum flux
                    double const b2 = b.x * b.x + b.y * b.y + b.z * b.z;
                    double const bn = dir == 0 ? b.x : b.y;
                    return std::array<double, 3>{-bn * b.x + (dir == 0 ? 0.5 * b2 : 0.),
                                                 -bn * b.y + (dir == 1 ? 0.5 * b2 : 0.),
                                                 -bn * b.z};
                };
                auto M = [&](int dir, I i, I j) {
                    std::array<double, 3> m{0, 0, 0};
                    if (dir == 0)
                    {
                        auto const [bl, br]
                            = Rtor::template transverse_reconstruct<Direction::X>(B, Idx{i, j});
                        auto const ml = stress(bl, 0), mr = stress(br, 0);
                        for (int a = 0; a < 3; ++a)
                            m[a] = 0.5 * (ml[a] + mr[a]);
                    }
                    else
                    {
                        auto const [bl, br]
                            = Rtor::template transverse_reconstruct<Direction::Y>(B, Idx{i, j});
                        auto const ml = stress(bl, 1), mr = stress(br, 1);
                        for (int a = 0; a < 3; ++a)
                            m[a] = 0.5 * (ml[a] + mr[a]);
                    }
                    return m;
                };
                layout.evalOnBox(state.rho, [&](auto i_, auto j_) {
                    I const i = i_, j = j_;
                    auto const xy = layout.cellCenteredCoordinates(layout.localToAMR(core::Point<int, 2>{static_cast<int>(i_), static_cast<int>(j_)}));
                    if (std::abs(xy[0] - budgetX_) > (budgetW_ + 0.5) * dx
                        || std::abs(xy[1] - budgetY_) > (budgetW_ + 0.5) * dy)
                        return;
                    auto dv = [&](auto const& fx, auto const& fy) {
                        return (fx(i + 1, j) - fx(i, j)) * ix + (fy(i, j + 1) - fy(i, j)) * iy;
                    };
                    double const divRho = dv(fluxes.rho_fx, fluxes.rho_fy);
                    double const divE   = dv(fluxes.Etot_fx, fluxes.Etot_fy);
                    double const v[3]   = {state.V(Component::X)(i, j), state.V(Component::Y)(i, j),
                                           state.V(Component::Z)(i, j)};
                    double const v2     = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    Component const cs[3] = {Component::X, Component::Y, Component::Z};
                    auto const mxl = M(0, i, j), mxr = M(0, i + 1, j), myl = M(1, i, j),
                               myr = M(1, i, j + 1);
                    double KEdot = 0.5 * v2 * divRho, L = 0;
                    for (int a = 0; a < 3; ++a)
                    {
                        double const divm = dv(fluxes.rhoV_fx(cs[a]), fluxes.rhoV_fy(cs[a]));
                        double const divM = (mxr[a] - mxl[a]) * ix + (myr[a] - myl[a]) * iy;
                        KEdot -= v[a] * divm;
                        L -= v[a] * divM;
                    }
                    Row r{ip, i, j, xy[0], xy[1], state.P(i, j) / (to_primitive_gamma_ - 1.),
                          -divE - (KEdot - L), L, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                    rows.push_back(r);
                });
                ++ip;
            }

            int const masks[3] = {1, 2, 4};
            bool const saved   = PC::central_;
            bool const savedK  = PC::consistent_;
            for (int k = 0; k < 3; ++k)
            {
                ConstrainedTransport_t{level, model, constrainedTransportInfo_, masks[k]}(
                    ct_, dissipative_electric_state_, state);
                ip = 0;
                for (auto& patch : rm.enumerate(level, ct_, state))
                {
                    auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                    PC pc{layout};
                    auto const& B   = state.B;
                    auto const& E   = state.E;
                    auto const curl = core::curl(layout, E);
                    double const ix = layout.inverseMeshSize(Direction::X);
                    double const iy = layout.inverseMeshSize(Direction::Y);
                    for (auto& r : rows)
                    {
                        if (r.ip != ip)
                            continue;
                        I const i = r.i, j = r.j;
                        auto X    = [&](int c, I a, I b) {
                            Idx const q{a, b};
                            return c == 0 ? curl.x(q) : c == 1 ? curl.y(q) : curl.z(q);
                        };
                        double work;
                        if (core::useFaceEnergy())
                            work = 0.5 * (B(Component::X)(i, j) * X(0, i, j)
                                          + B(Component::X)(i + 1, j) * X(0, i + 1, j))
                                   + 0.5 * (B(Component::Y)(i, j) * X(1, i, j)
                                            + B(Component::Y)(i, j + 1) * X(1, i, j + 1))
                                   + B(Component::Z)(i, j) * X(2, i, j);
                        else
                            work = 0.5 * (B(Component::X)(i, j) + B(Component::X)(i + 1, j)) * 0.5
                                       * (X(0, i, j) + X(0, i + 1, j))
                                   + 0.5 * (B(Component::Y)(i, j) + B(Component::Y)(i, j + 1)) * 0.5
                                         * (X(1, i, j) + X(1, i, j + 1))
                                   + B(Component::Z)(i, j) * X(2, i, j);
                        auto divP = [&](bool central, bool consistent = false) {
                            PC::central_    = central;
                            PC::consistent_ = consistent;
                            double sxl = 0, sxr = 0, syl = 0, syr = 0;
                            pc.template poynting_energy_flux_<Direction::X>(ct_, E, B, Idx{i, j}, sxl);
                            pc.template poynting_energy_flux_<Direction::X>(ct_, E, B, Idx{i + 1, j}, sxr);
                            pc.template poynting_energy_flux_<Direction::Y>(ct_, E, B, Idx{i, j}, syl);
                            pc.template poynting_energy_flux_<Direction::Y>(ct_, E, B, Idx{i, j + 1}, syr);
                            return (sxr - sxl) * ix + (syr - syl) * iy;
                        };
                        double dc = divP(true), du = divP(false), dk = divP(false, true);
                        if (masks[k] == 4 && Hall && useHallFluxFace_)
                        {
                            using Field_t = std::decay_t<decltype(B(Component::X))>;
                            using Rho_t   = std::decay_t<decltype(state.rho)>;
                            core::FacePairFluxSym<Field_t, Rho_t> const fs{
                                B(Component::X), B(Component::Y), B(Component::Z), state.rho, ix, iy};
                            dc = du = dk = (fs.Fx(i + 1, j) - fs.Fx(i, j)) * ix
                                      + (fs.Fy(i, j + 1) - fs.Fy(i, j)) * iy;
                        }
                        r.Rc[k] = -dc + work;
                        r.Ru[k] = -du + work;
                        r.Rk[k] = -dk + work;
                    }
                    ++ip;
                }
            }
            PC::central_    = saved;
            PC::consistent_ = savedK;

            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            for (auto const& r : rows)
                std::fprintf(stderr,
                             "BUDGET r=%s t=%.9e c=%zu x=%.3f y=%.3f eth=%.6e Hyd=%.6e L=%.6e "
                             "RAc=%.6e RAu=%.6e RDc=%.6e RDu=%.6e RHc=%.6e RHu=%.6e RAk=%.6e RDk=%.6e RHk=%.6e\n",
                             rk ? rk : "0", time, budgetCalls_, r.x, r.y, r.eth, r.hyd, r.L,
                             r.Rc[0], r.Ru[0], r.Rc[1], r.Ru[1], r.Rc[2], r.Ru[2], r.Rk[0], r.Rk[1],
                             r.Rk[2]);
            ++budgetCalls_;
        }
    }
    std::size_t budgetCalls_ = 0;
    bool const pminOn_       = std::getenv("PHARE_CTP_PMIN") != nullptr;
    bool const budgetOn_     = std::getenv("PHARE_CTP_BUDGET") != nullptr;
    double budgetX_ = 0, budgetY_ = 0, budgetW_ = 0;
    double const budgetT0_ = [] {
        auto const* s = std::getenv("PHARE_CTP_BUDGET_T0");
        return s ? std::atof(s) : 0.;
    }();
    bool const budgetParsed_ = [this] {
        if (auto const* s = std::getenv("PHARE_CTP_BUDGET"))
            std::sscanf(s, "%lf,%lf,%lf", &budgetX_, &budgetY_, &budgetW_);
        return true;
    }();

    void hallSource_(MHDModel& model, auto& state, auto& fluxes, level_t& level)
    {
        if constexpr (MHDModel::dimension == 2)
        {
            using GridLayout = MHDModel::gridlayout_type;
            using core::Component;
            using Idx   = core::MeshIndex<2>;
            using implT = GridLayout::implT;
            auto& rm    = *model.resourcesManager;
            for (auto& patch : rm.enumerate(level, state, fluxes))
            {
                auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                auto const& B     = state.B;
                auto const curl   = core::curl(layout, state.E);
                struct C
                {
                    using type = double;
                    std::function<double(Idx)> f;
                    double operator()(Idx i) const { return f(i); }
                };
                C cx{[&](Idx i) { return curl.x(i); }}, cy{[&](Idx i) { return curl.y(i); }},
                    cz{[&](Idx i) { return curl.z(i); }};
                auto& Q = fluxes.B_fx(Component::Z);
                layout.evalOnBox(state.rho, [&](auto i, auto j) {
                    Idx const c{static_cast<std::size_t>(i), static_cast<std::size_t>(j)};
                    Q(c[0], c[1]) = -(
                        GridLayout::template project<implT::faceXToCellCenter>(B(Component::X), c)
                            * GridLayout::template project<implT::faceXToCellCenter>(cx, c)
                        + GridLayout::template project<implT::faceYToCellCenter>(B(Component::Y), c)
                              * GridLayout::template project<implT::faceYToCellCenter>(cy, c)
                        + GridLayout::template project<implT::faceZToCellCenter>(B(Component::Z), c)
                              * GridLayout::template project<implT::faceZToCellCenter>(cz, c));
                });
            }
        }
    }

    void hallFlux_(MHDModel& model, auto& state, auto& fluxes, level_t& level)
    {
        using GridLayout = MHDModel::gridlayout_type;
        for (auto& patch : model.resourcesManager->enumerate(level, state, fluxes))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            if constexpr (MHDModel::dimension >= 2)
                if (useHallFluxFace_)
                {
                    core::add_hall_energy_flux_face(fluxes, state.B, state.rho, layout);
                    continue;
                }
            core::add_hall_energy_flux(fluxes, state.B, state.rho, layout);
        }
    }

    // F_H^sym evaluated at a face index (dir 0/1/2)
    template<int dir>
    static double faceFluxAt_(auto const& fs, auto const& idx)
    {
        constexpr auto dim = MHDModel::dimension;
        auto call          = [&](auto... ii) {
            if constexpr (dir == 0)
                return fs.Fx(ii...);
            else if constexpr (dir == 1)
                return fs.Fy(ii...);
            else
                return fs.Fz(ii...);
        };
        if constexpr (dim == 2)
            return call(idx[0], idx[1]);
        else
            return call(idx[0], idx[1], idx[2]);
    }
    // div_h F_H^sym of cell c
    static double faceFluxDiv_(auto const& fs, auto const& layout, auto const& c)
    {
        double div = 0;
        core::for_N<MHDModel::dimension>([&](auto dd) {
            constexpr int d = dd();
            auto cr         = c;
            cr[d] += 1;
            div += (faceFluxAt_<d>(fs, cr) - faceFluxAt_<d>(fs, c))
                   * layout.inverseMeshSize(static_cast<core::Direction>(d));
        });
        return div;
    }

    // FF gates (2D, 3D, PHARE_HALL_FLUX_FACE): with the face energy, per cell
    //  hall=true  (E = face-pair E_H): FF-2 |Q_face + div F_H^sym|, control |Q_face + div P(E_H)|
    //  hall=false (E = non-Hall E):    FF-1 |Q_face + div P(E) + w_sym(E)| (S2b identity; w_sym = each
    //  edge's E.J split evenly over the cells sharing it: 1/2 or 1/4 in 2D, 1/4 in 3D)
    void fhCheckFace_(MHDModel& model, auto& state, level_t& level, double const time, bool const hall)
    {
        constexpr auto dim = MHDModel::dimension;
        if constexpr (dim >= 2)
        {
            using GridLayout = MHDModel::gridlayout_type;
            using PC         = core::PoyntingCorrection<GridLayout, Rec>;
            using core::Component;
            using core::Direction;
            using Idx   = core::MeshIndex<dim>;
            double maxQ = 0, maxR = 0, maxC = 0;
            Component const comps[3] = {Component::X, Component::Y, Component::Z};
            for (auto& patch : model.resourcesManager->enumerate(level, ct_, state))
            {
                auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                PC pc{layout};
                auto const& B   = state.B;
                auto const& E   = state.E;
                auto const& J   = state.J;
                auto const curl = core::curl(layout, E);
                auto const fs   = core::make_face_pair_flux(B, state.rho, layout);
                auto X          = [&](int c, Idx const& k) {
                    return c == 0 ? curl.x(k) : c == 1 ? curl.y(k) : curl.z(k);
                };
                auto ej = [&](int c, Idx const& k) { return E(comps[c])(k) * J(comps[c])(k); };
                layout.evalOnBox(state.rho, [&](auto... ii) {
                    Idx const c{static_cast<std::size_t>(ii)...};
                    auto up = [&](Idx k, int d) {
                        k[d] += 1;
                        return k;
                    };
                    double Q = 0;
                    for (int d = 0; d < 3; ++d)
                        Q -= d < static_cast<int>(dim)
                                 ? 0.5 * (B(comps[d])(c) * X(d, c) + B(comps[d])(up(c, d)) * X(d, up(c, d)))
                                 : B(comps[d])(c) * X(d, c);
                    double divP = 0;
                    core::for_N<dim>([&](auto dd) {
                        constexpr auto dir = static_cast<Direction>(dd());
                        double fl = 0, fr = 0;
                        pc.template poynting_energy_flux_<dir>(ct_, E, B, c, fl);
                        pc.template poynting_energy_flux_<dir>(ct_, E, B, up(c, dd()), fr);
                        divP += (fr - fl) * layout.inverseMeshSize(dir);
                    });
                    maxQ = std::max(maxQ, std::abs(Q));
                    if (hall)
                    {
                        maxR = std::max(maxR, std::abs(Q + faceFluxDiv_(fs, layout, c)));
                        maxC = std::max(maxC, std::abs(Q + divP));
                    }
                    else
                    {
                        double wsym = 0;
                        for (int cc = 0; cc < 3; ++cc)
                        {
                            // average over the shifts along the in-grid axes other than cc
                            int ax[2], na = 0;
                            for (int a = 0; a < static_cast<int>(dim); ++a)
                                if (a != cc)
                                    ax[na++] = a;
                            double s = 0;
                            for (int m = 0; m < (1 << na); ++m)
                            {
                                Idx k = c;
                                for (int q = 0; q < na; ++q)
                                    if (m & (1 << q))
                                        k[ax[q]] += 1;
                                s += ej(cc, k);
                            }
                            wsym += s / (1 << na);
                        }
                        maxR = std::max(maxR, std::abs(Q + divP + wsym));
                    }
                });
            }
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            if (hall)
                std::fprintf(stderr,
                             "FFCHECK r=%s t=%.6e FF2 maxQ=%.6e maxRes=%.6e rel=%.3e | control CTP "
                             "rel=%.3e\n",
                             rk ? rk : "0", time, maxQ, maxR, maxQ > 0 ? maxR / maxQ : 0.,
                             maxQ > 0 ? maxC / maxQ : 0.);
            else
                std::fprintf(stderr,
                             "FFCHECK r=%s t=%.6e FF1nonHall maxQ=%.6e maxRes=%.6e rel=%.3e\n",
                             rk ? rk : "0", time, maxQ, maxR, maxQ > 0 ? maxR / maxQ : 0.);
        }
    }
    bool fhCheckNonHall_ = false;

    // gate FH1: with state.E = E_H (CT mask 4), max_c |Q_c + div F_H| vs max|Q| on this rank
    void fhCheck_(MHDModel& model, auto& state, level_t& level, double const time)
    {
        using GridLayout   = MHDModel::gridlayout_type;
        constexpr auto dim = MHDModel::dimension;
        using core::Component;
        using Idx   = core::MeshIndex<dim>;
        using implT = GridLayout::implT;
        double maxQ = 0, maxR = 0;
        for (auto& patch : model.resourcesManager->enumerate(level, state))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            auto const& B     = state.B;
            auto const curl   = core::curl(layout, state.E);
            struct C
            {
                using type = double;
                std::function<double(Idx)> f;
                double operator()(Idx i) const { return f(i); }
            };
            C cx{[&](Idx i) { return curl.x(i); }}, cy{[&](Idx i) { return curl.y(i); }},
                cz{[&](Idx i) { return curl.z(i); }};
            auto const zw   = core::make_zero_work_hall(B, state.rho, layout);
            double const ix = layout.inverseMeshSize(core::Direction::X);
            layout.evalOnBox(state.rho, [&](auto... ii) {
                Idx const c{static_cast<std::size_t>(ii)...};
                double const Q
                    = -(GridLayout::template project<implT::faceXToCellCenter>(B(Component::X), c)
                            * GridLayout::template project<implT::faceXToCellCenter>(cx, c)
                        + GridLayout::template project<implT::faceYToCellCenter>(B(Component::Y), c)
                              * GridLayout::template project<implT::faceYToCellCenter>(cy, c)
                        + GridLayout::template project<implT::faceZToCellCenter>(B(Component::Z), c)
                              * GridLayout::template project<implT::faceZToCellCenter>(cz, c));
                double divF = 0;
                if constexpr (dim == 1)
                {
                    std::size_t const I = c[0];
                    divF                = (zw.Fx(I + 1) - zw.Fx(I)) * ix;
                }
                else if constexpr (dim == 2)
                {
                    std::size_t const I = c[0], J = c[1];
                    divF = (zw.Fx(I + 1, J) - zw.Fx(I, J)) * ix
                           + (zw.Fy(I, J + 1) - zw.Fy(I, J))
                                 * layout.inverseMeshSize(core::Direction::Y);
                }
                else
                {
                    std::size_t const I = c[0], J = c[1], K = c[2];
                    divF = (zw.Fx(I + 1, J, K) - zw.Fx(I, J, K)) * ix
                           + (zw.Fy(I, J + 1, K) - zw.Fy(I, J, K))
                                 * layout.inverseMeshSize(core::Direction::Y)
                           + (zw.Fz(I, J, K + 1) - zw.Fz(I, J, K))
                                 * layout.inverseMeshSize(core::Direction::Z);
                }
                maxQ = std::max(maxQ, std::abs(Q));
                maxR = std::max(maxR, std::abs(Q + divF));
            });
        }
        auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
        std::fprintf(stderr, "FHCHECK r=%s t=%.6e maxQ=%.6e maxRes=%.6e rel=%.3e\n", rk ? rk : "0",
                     time, maxQ, maxR, maxQ > 0 ? maxR / maxQ : 0.);
    }

    bool const useHallFlux_     = std::getenv("PHARE_HALL_FLUX") != nullptr;
    bool const useHallFluxFace_ = std::getenv("PHARE_HALL_FLUX_FACE") != nullptr;
    int const fhCheckEvery_     = [] {
        auto const* s = std::getenv("PHARE_FH_CHECK");
        return s ? std::atoi(s) : 0;
    }();
    std::size_t fhCheckCalls_ = 0;

    bool const hallNeutral_ = std::getenv("PHARE_HALL_NEUTRAL") != nullptr;

    int const ctpCentralMask_ = [] {
        auto const* s = std::getenv("PHARE_CTP_CENTRAL_MASK");
        return s ? std::atoi(s) : -1;
    }();

    int const closureEvery_ = [] {
        auto const* s = std::getenv("PHARE_CLOSURE_DIAG");
        return s ? std::atoi(s) : 0;
    }();
    std::size_t closureCalls_ = 0;
    // experiment PHARE_CLOSURE_DIAG_T0: no probe before this time
    double const closureT0_ = [] {
        auto const* s = std::getenv("PHARE_CLOSURE_DIAG_T0");
        return s ? std::atof(s) : 0.;
    }();

    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;
    bool const isDissipative_{fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()};

    core::UpwindConstrainedTransportState<VecField> ct_{Hall};
    core::DissipativeElectricFieldState<VecField> dissipative_electric_state_{isDissipative_};
    double to_primitive_gamma_;
    double to_conservative_gamma_;
    bool needsCurrent_;
};
} // namespace PHARE::solver

#endif
