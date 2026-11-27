#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"

#include "core/numerics/point_values_handler/point_value_converter.hpp"

#include "initializer/data_provider.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

        // EXPERIMENT PHARE_O4_PV_LIMIT_DIAG=N: flagged (troubled) fraction of physical cells
        if constexpr (pointValues)
            if (pvlimDiag_ > 0 && ++pvlimCalls_ % pvlimDiag_ == 1 % pvlimDiag_)
            {
                auto& st       = core::pvLimitStats;
                auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
                std::printf("PVLIM r=%s t=%.6e flagged=%ld total=%ld\n", rk ? rk : "0", newTime,
                            st.flagged, st.total);
                std::fflush(stdout);
            }
        core::pvLimitStats = {};

        if constexpr (pointValues && MHDModel::dimension == 2)
            if (pminDiag_ > 0)
                pminProbe_(model, numericalState, level, newTime);

        if (facePairHall_)
            DissipativeElectricField_t{level, model, fVMethodInfo_, pointValues}(
                dissipative_electric_state_, numericalState, state);
        else if (isDissipative_)
            DissipativeElectricField_t{level, model, fVMethodInfo_,
                                       pointValues}(dissipative_electric_state_, numericalState);

        FVMethod_t{level, model, fVMethodInfo_}(ct_, dissipative_electric_state_, numericalState,
                                                fluxes);

        if (core::o4FaceProbe.n > 0)
        {
            if (facePrints_++ < 40)
            {
                auto const& f  = core::o4FaceProbe;
                auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
                std::printf("FACEP r=%s t=%.6e nbad=%ld first: dir=%d face@(%.3f,%.3f) P=%.4e "
                            "rho=%.4e\n",
                            rk ? rk : "0", newTime, f.n, f.dir, f.x, f.y, f.P, f.rho);
                std::fflush(stdout);
            }
            core::o4FaceProbe.n = 0;
        }

        ConstrainedTransport_t{level, model, constrainedTransportInfo_}(
            ct_, dissipative_electric_state_, numericalState, state.E);

        approximation_.pointValueFluxesToAverages(model, fluxes, state.E, level, newTime);

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
    int const pvlimDiag_ = []() {
        auto const* s = std::getenv("PHARE_O4_PV_LIMIT_DIAG");
        return s ? std::atoi(s) : 0;
    }();
    std::size_t pvlimCalls_ = 0;

    // EXPERIMENT PHARE_O4_PMIN_DIAG=N: min point-value pressure and density on the physical cells
    // every N flux calls, and on (at most 40) calls where one is non-positive or non-finite
    void pminProbe_(MHDModel& model, auto& pv, level_t& level, double const time)
    {
        using GridLayout = MHDModel::gridlayout_type;
        ++pminCalls_;
        double pmin = 1e300, rmin = 1e300, xmin = 0, ymin = 0;
        std::size_t nneg = 0, nnan = 0;
        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, pv))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            layout.evalOnBox(pv.P, [&](auto i, auto j) {
                double const p = pv.P(i, j), r = pv.rho(i, j);
                if (!std::isfinite(p) || !std::isfinite(r))
                {
                    ++nnan;
                    return;
                }
                if (p <= 0 || r <= 0)
                    ++nneg;
                rmin = std::min(rmin, r);
                if (p < pmin)
                {
                    pmin         = p;
                    auto const a = layout.localToAMR(core::Point<int, 2>{int(i), int(j)});
                    xmin         = (a[0] + 0.5) * layout.meshSize()[0];
                    ymin         = (a[1] + 0.5) * layout.meshSize()[1];
                }
            });
        }
        bool const bad = nneg > 0 || nnan > 0;
        if ((bad && pminBadPrints_++ < 40) || (pminCalls_ % pminDiag_) == 1 % pminDiag_)
        {
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            std::printf("PMIN r=%s t=%.6e pmin=%.6e @(%.3f,%.3f) rhomin=%.6e nonpos=%zu nan=%zu\n",
                        rk ? rk : "0", time, pmin, xmin, ymin, rmin, nneg, nnan);
            std::fflush(stdout);
        }
    }
    int const pminDiag_ = []() {
        auto const* s = std::getenv("PHARE_O4_PMIN_DIAG");
        return s ? std::atoi(s) : 0;
    }();
    std::size_t pminCalls_ = 0, pminBadPrints_ = 0, facePrints_ = 0;

    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;
    bool const isDissipative_{fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()};
    // Hall field from the F12 face pair instead of the Riemann solver and UCT (runtime switch)
    bool const facePairHall_{Hall && core::hallFacePairF12()};

    PointValueApproximation approximation_{};
    core::UpwindConstrainedTransportState<VecField> ct_{Hall && !facePairHall_};
    core::DissipativeElectricFieldState<VecField> dissipative_electric_state_{isDissipative_,
                                                                              facePairHall_};
    double gamma_;
    bool needsCurrent_;
};
} // namespace PHARE::solver

#endif
