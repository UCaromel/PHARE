#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"

#include "core/numerics/constrained_transport/o4_zero_work_hall.hpp"

#include "initializer/data_provider.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel, typename PointValueApproximation>
class ComputeFluxes
{
    using level_t       = MHDModel::level_t;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using DissipativeElectricField_t = Dispatchers_t::DissipativeElectricField_t;

    using FVMethod_t     = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;
    using FVMethodInfo_t = FVMethod_t::info_type;

    constexpr static auto Hall = FVMethod_t::Hall;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t     = Dispatchers_t::template ConstrainedTransport_t<Rec, Hall>;
    using ConstrainedTransportInfo_t = ConstrainedTransport_t::info_type;

    // from point values, CT and the flux averaging read the dissipative EMF beyond the cell box
    constexpr static bool pointValues = Rec<typename MHDModel::gridlayout_type>::pointValues;

    using VecField = MHDModel::vecfield_type;


public:
    explicit ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
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

    // The flux pipeline runs on the approximation's point values: the state itself at second
    // order, a separate point-value state at fourth order, whose fluxes and EMF are then turned
    // back into face and edge averages.
    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const newTime)
    {
        auto& numericalState = approximation_.averageStateToPointValues(
            model, state, level, newTime, gamma_, needsCurrent_);

        if constexpr (pointValues && MHDModel::dimension == 2)
            if (pminDiag_ > 0)
                pminProbe_(model, numericalState, level, newTime);

        if (isDissipative_)
            DissipativeElectricField_t{level, model, fVMethodInfo_,
                                       pointValues}(dissipative_electric_state_, numericalState);

        FVMethod_t{level, model, fVMethodInfo_}(ct_, dissipative_electric_state_, numericalState,
                                                fluxes);

        ConstrainedTransport_t{level, model, constrainedTransportInfo_}(
            ct_, dissipative_electric_state_, numericalState, state.E);

        // EXPERIMENT PHARE_O4_POYNTING=1 (with PHARE_O4_HALL_ZW): closure Poynting flux of the total
        // EMF, as a point-value face flux, before the unchanged flux averaging (O4/pipeline)
        if constexpr (Hall && pointValues && MHDModel::dimension == 2)
            if (o4zw_ && o4poy_)
                addPoyntingO4_(model, state, fluxes, level, newTime);

        approximation_.pointValueFluxesToAverages(model, fluxes, state.E, level, newTime);

        // EXPERIMENT PHARE_O4_HALL_ZW=1 (2D, O4, Hall): the zero-work Hall EMF from the stage
        // averages replaces the upwinded Hall blocks of UCT (O4/derivation.md S6)
        if constexpr (Hall && pointValues && MHDModel::dimension == 2)
            if (o4zw_)
                addZeroWorkHallO4_(model, state, level, newTime);

        bc.fillElectricGhosts(state.E, level, newTime);
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(approximation_);
        model.resourcesManager->registerResources(ct_);
        model.resourcesManager->registerResources(dissipative_electric_state_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(approximation_, patch, allocateTime);
        model.resourcesManager->allocate(ct_, patch, allocateTime);
        model.resourcesManager->allocate(dissipative_electric_state_, patch, allocateTime);
    }

private:
    void addZeroWorkHallO4_(MHDModel& model, auto& state, level_t& level, double const time)
    {
        using GridLayout = MHDModel::gridlayout_type;
        using core::Component;
        using core::Direction;
        using core::o4zw::Grid2;

        ++o4zwCalls_;
        bool const diag = o4zwDiag_ > 0 && (o4zwCalls_ % o4zwDiag_) == 1 % o4zwDiag_;
        double sumW = 0, absW = 0;
        std::size_t nonFinite = 0;

        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, state))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            int const NX = layout.ghostEndIndex(core::QtyCentering::primal, Direction::X) + 1;
            int const NY = layout.ghostEndIndex(core::QtyCentering::primal, Direction::Y) + 1;

            auto load = [&](auto const& f) {
                Grid2 g{NX, NY};
                int const ex = layout.ghostEndIndex(f, Direction::X);
                int const ey = layout.ghostEndIndex(f, Direction::Y);
                for (int i = 0; i <= ex; ++i)
                    for (int j = 0; j <= ey; ++j)
                        g(i, j) = f(i, j);
                return g;
            };

            auto const& B = state.B;
            auto const dl = layout.meshSize();
            auto r = core::o4zw::hallE(load(B(Component::X)), load(B(Component::Y)),
                                       load(B(Component::Z)), load(state.rho), dl[0], dl[1]);
            if (double const sc = core::o4zw::hallEmfScale(); sc != 1.0)
                for (auto* g : {&r.Ex, &r.Ey, &r.Ez})
                    for (auto& v : g->a)
                        v *= sc;

            auto add = [&](auto& Ec, Grid2 const& EH, Grid2 const& Jt) {
                int const sx = layout.physicalStartIndex(Ec, Direction::X);
                int const ex = layout.physicalEndIndex(Ec, Direction::X);
                int const sy = layout.physicalStartIndex(Ec, Direction::Y);
                int const ey = layout.physicalEndIndex(Ec, Direction::Y);
                // owned edges for the global work sum: drop the shared last primal index
                bool const px = layout.centering(Ec.physicalQuantity())[0]
                                == core::QtyCentering::primal;
                bool const py = layout.centering(Ec.physicalQuantity())[1]
                                == core::QtyCentering::primal;
                for (int i = sx; i <= ex; ++i)
                    for (int j = sy; j <= ey; ++j)
                    {
                        double const e = EH(i, j);
                        if (!std::isfinite(e))
                            ++nonFinite;
                        Ec(i, j) += e;
                        if (diag && !(px && i == ex) && !(py && j == ey))
                        {
                            sumW += e * Jt(i, j);
                            absW += std::abs(e * Jt(i, j));
                        }
                    }
            };
            add(state.E(Component::X), r.Ex, r.Jtx);
            add(state.E(Component::Y), r.Ey, r.Jty);
            add(state.E(Component::Z), r.Ez, r.Jtz);
        }
        // no throw (it would desynchronise the ranks): the NaN reaches the solver's NaN check
        if (nonFinite > 0)
            std::printf("O4ZW nonfinite EMF on %zu edges t=%.6e\n", nonFinite, time);
        if (diag)
        {
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            std::printf("O4ZW r=%s t=%.6e sumW=%.6e absW=%.6e\n", rk ? rk : "0", time, sumW,
                        absW);
            std::fflush(stdout);
        }
    }

    template<typename State, typename Fluxes>
    void addPoyntingO4_(MHDModel& model, State& state, Fluxes& fluxes, level_t& level,
                        double const time)
    {
        using GridLayout = MHDModel::gridlayout_type;
        using core::Component;
        using core::Direction;
        using core::QtyCentering;
        using core::o4zw::Grid2;

        ++o4poyCalls_;
        bool const diag = o4poyDiag_ > 0 && (o4poyCalls_ % o4poyDiag_) == 1 % o4poyDiag_;
        std::size_t nonFinite = 0;
        // diagnostics: Hall leak into p_pv per cell (cold cells / e_th), total closure residual
        double minH = 0, maxAbsSH = 0, maxAbsQH = 0, maxAbsStot = 0, maxAbsQw = 0, sumSH = 0;
        double minHx = 0, minHy = 0, minHp = 0;
        std::size_t nDiag = 0, nCells = 0;

        auto& pv = approximation_.pointValues();
        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, state, fluxes, pv))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            auto const dl     = layout.meshSize();
            if (std::abs(dl[0] - dl[1]) > 1e-12 * dl[0])
                throw std::runtime_error("O4POY: needs dx == dy");
            double const h = dl[0];
            int const NX   = layout.ghostEndIndex(QtyCentering::primal, Direction::X) + 1;
            int const NY   = layout.ghostEndIndex(QtyCentering::primal, Direction::Y) + 1;

            auto load = [&](auto const& f) {
                Grid2 g{NX, NY};
                int const ex = layout.ghostEndIndex(f, Direction::X);
                int const ey = layout.ghostEndIndex(f, Direction::Y);
                for (int i = 0; i <= ex; ++i)
                    for (int j = 0; j <= ey; ++j)
                        g(i, j) = f(i, j);
                return g;
            };
            // point-value E from UCT, only where UCT evaluated it (own axis +3, transverse +2)
            auto loadBand = [&](auto const& f, int own) {
                Grid2 g{NX, NY};
                int lo[2], hi[2];
                for (int a = 0; a < 2; ++a)
                {
                    auto const dir = a == 0 ? Direction::X : Direction::Y;
                    int const grow = (a == own) ? 3 : 2;
                    lo[a]          = static_cast<int>(layout.physicalStartIndex(f, dir)) - grow;
                    hi[a]          = static_cast<int>(layout.physicalEndIndex(f, dir)) + grow;
                }
                for (int i = lo[0]; i <= hi[0]; ++i)
                    for (int j = lo[1]; j <= hi[1]; ++j)
                        g(i, j) = f(i, j);
                return g;
            };

            Grid2 Bb[3] = {load(state.B(Component::X)), load(state.B(Component::Y)),
                           load(state.B(Component::Z))};
            auto H = core::o4zw::hallE(Bb[0], Bb[1], Bb[2], load(state.rho), h, h);
            if (double const sc = core::o4zw::hallEmfScale(); sc != 1.0)
                for (auto* g : {&H.Ex, &H.Ey, &H.Ez})
                    for (auto& v : g->a)
                        v *= sc;

            auto Epx = loadBand(state.E(Component::X), 0);
            auto Epy = loadBand(state.E(Component::Y), 1);
            auto Epz = loadBand(state.E(Component::Z), 2);
            using core::o4zw::D2;
            using core::o4zw::make;
            Grid2 Et[3];
            Et[0] = make(Epx, [&](int i, int j) {
                return Epx(i, j) + D2(Epx, 0, i, j) / 24 + H.Ex(i, j);
            });
            Et[1] = make(Epy, [&](int i, int j) {
                return Epy(i, j) + D2(Epy, 1, i, j) / 24 + H.Ey(i, j);
            });
            Et[2] = make(Epz, [&](int i, int j) { return Epz(i, j) + H.Ez(i, j); });

            auto const F = core::o4zw::pipelineFpv(Bb, Et, h);

            auto addFlux = [&](auto& Ff, Grid2 const& Fp, int dirNormal) {
                int lo[2], hi[2];
                for (int a = 0; a < 2; ++a)
                {
                    auto const dir = a == 0 ? Direction::X : Direction::Y;
                    int const grow = (a == dirNormal) ? 0 : 1; // flux averaging reads +-1 across
                    lo[a]          = static_cast<int>(layout.physicalStartIndex(Ff, dir)) - grow;
                    hi[a]          = static_cast<int>(layout.physicalEndIndex(Ff, dir)) + grow;
                }
                for (int i = lo[0]; i <= hi[0]; ++i)
                    for (int j = lo[1]; j <= hi[1]; ++j)
                    {
                        if (!std::isfinite(Fp(i, j)))
                            ++nonFinite;
                        Ff(i, j) += Fp(i, j);
                    }
            };
            addFlux(fluxes.Etot_fx, F[0], 0);
            addFlux(fluxes.Etot_fy, F[1], 1);

            if (diag)
            {
                // s = (1 - Lap/24)(-div Fbar) - target, Fbar = (1 + T/24) F_pv; cells where finite
                auto cellRate = [&](std::array<Grid2, 2> const& Fp) {
                    auto Fbx = make(Fp[0], [&](int i, int j) {
                        return Fp[0](i, j) + D2(Fp[0], 1, i, j) / 24;
                    });
                    auto Fby = make(Fp[1], [&](int i, int j) {
                        return Fp[1](i, j) + D2(Fp[1], 0, i, j) / 24;
                    });
                    auto dv = make(Fbx, [&](int i, int j) {
                        return -((Fbx.s(i, j, 1, 0) - Fbx(i, j)) + (Fby.s(i, j, 0, 1) - Fby(i, j)))
                               / h;
                    });
                    return make(dv, [&](int i, int j) {
                        return dv(i, j) - (D2(dv, 0, i, j) + D2(dv, 1, i, j)) / 24;
                    });
                };
                Grid2 EH[3] = {H.Ex, H.Ey, H.Ez};
                auto const cH  = core::o4zw::chain(Bb, EH, h, true);
                auto const rH  = cellRate(core::o4zw::pipelineFpv(Bb, EH, h));
                auto const cT  = core::o4zw::chain(Bb, Et, h, true);
                auto const rT  = cellRate(F);
                auto const& Pp = pv.P;
                int const sx   = layout.physicalStartIndex(QtyCentering::dual, Direction::X);
                int const ex   = layout.physicalEndIndex(QtyCentering::dual, Direction::X);
                int const sy   = layout.physicalStartIndex(QtyCentering::dual, Direction::Y);
                int const ey   = layout.physicalEndIndex(QtyCentering::dual, Direction::Y);
                double const eth0 = 1.0 / (gamma_ - 1.0);
                for (int i = sx; i <= ex; ++i)
                    for (int j = sy; j <= ey; ++j)
                    {
                        ++nCells;
                        double const sH = rH(i, j) - cH.Q(i, j); // Hall leak into p_pv
                        double const sT = rT(i, j) - (cT.Q(i, j) + cT.w4(i, j));
                        if (!std::isfinite(sH) || !std::isfinite(sT))
                            continue;
                        ++nDiag;
                        sumSH += sH;
                        maxAbsSH   = std::max(maxAbsSH, std::abs(sH));
                        maxAbsQH   = std::max(maxAbsQH, std::abs(cH.Q(i, j)));
                        maxAbsStot = std::max(maxAbsStot, std::abs(sT));
                        maxAbsQw   = std::max(maxAbsQw, std::abs(cT.Q(i, j) + cT.w4(i, j)));
                        double const p = Pp(i, j);
                        if (p < 1.0) // cold (ambient p = 0.1)
                        {
                            double const r = sH / (p * eth0);
                            if (r < minH)
                            {
                                minH  = r;
                                auto const a = layout.localToAMR(core::Point<int, 2>{i, j});
                                minHx = (a[0] + 0.5) * h;
                                minHy = (a[1] + 0.5) * h;
                                minHp = p;
                            }
                        }
                    }
            }
        }
        if (nonFinite > 0)
            std::printf("O4POY nonfinite flux on %zu faces t=%.6e\n", nonFinite, time);
        if (diag)
        {
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            std::printf("O4POY r=%s t=%.6e minH/eth=%.3e @(%.3f,%.3f) p=%.3e | max|sH|=%.3e "
                        "max|QH|=%.3e sumSH=%.3e | max|s_tot|=%.3e max|Q+w4|=%.3e | cells %zu/%zu\n",
                        rk ? rk : "0", time, minH, minHx, minHy, minHp, maxAbsSH, maxAbsQH, sumSH,
                        maxAbsStot, maxAbsQw, nDiag, nCells);
            std::fflush(stdout);
        }
    }

    // EXPERIMENT PHARE_O4_PMIN_DIAG=N: min point-value pressure on the physical cells every N flux
    // calls, and on every call while it is negative
    void pminProbe_(MHDModel& model, auto& pv, level_t& level, double const time)
    {
        using GridLayout = MHDModel::gridlayout_type;
        ++pminCalls_;
        double pmin = 1e300, xmin = 0, ymin = 0, xnan = 0, ynan = 0;
        std::size_t nneg = 0, nnan = 0;
        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, pv))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            layout.evalOnBox(pv.P, [&](auto i, auto j) {
                double const p = pv.P(i, j);
                auto const a   = layout.localToAMR(core::Point<int, 2>{int(i), int(j)});
                if (!std::isfinite(p))
                {
                    if (nnan++ == 0)
                    {
                        xnan = (a[0] + 0.5) * layout.meshSize()[0];
                        ynan = (a[1] + 0.5) * layout.meshSize()[1];
                    }
                    return;
                }
                if (p <= 0)
                    ++nneg;
                if (p < pmin)
                {
                    pmin = p;
                    xmin = (a[0] + 0.5) * layout.meshSize()[0];
                    ymin = (a[1] + 0.5) * layout.meshSize()[1];
                }
            });
        }
        if (nneg > 0 || nnan > 0 || (pminCalls_ % pminDiag_) == 1 % pminDiag_)
        {
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            std::printf("PMIN r=%s t=%.6e call=%zu pmin_pv=%.4e @(%.3f,%.3f) nonpos=%zu nan=%zu "
                        "firstnan@(%.3f,%.3f)\n",
                        rk ? rk : "0", time, pminCalls_, pmin, xmin, ymin, nneg, nnan, xnan, ynan);
            std::fflush(stdout);
        }
    }
    int const pminDiag_     = core::o4zw::envInt("PHARE_O4_PMIN_DIAG", 0);
    std::size_t pminCalls_  = 0;

    bool const o4poy_     = core::o4zw::envFlag("PHARE_O4_POYNTING");
    int const o4poyDiag_  = core::o4zw::envInt("PHARE_O4POY_DIAG", 0);
    std::size_t o4poyCalls_ = 0;

    bool const o4zw_     = core::o4zw::envFlag("PHARE_O4_HALL_ZW");
    int const o4zwDiag_  = core::o4zw::envInt("PHARE_O4ZW_DIAG", 0);
    std::size_t o4zwCalls_ = 0;

    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;
    bool const isDissipative_{fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()};

    PointValueApproximation approximation_{};
    core::UpwindConstrainedTransportState<VecField> ct_{Hall};
    core::DissipativeElectricFieldState<VecField> dissipative_electric_state_{isDissipative_};
    double gamma_;
    bool needsCurrent_;
};
} // namespace PHARE::solver

#endif
