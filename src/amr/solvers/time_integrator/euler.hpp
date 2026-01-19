#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_HPP

#include "amr/debugod.hpp"

#include "initializer/data_provider.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "amr/solvers/time_integrator/euler_using_computed_flux.hpp"

namespace PHARE::solver
{
template<template<typename, typename> typename FVMethodStrategy, typename MHDModel>
class Euler
{
    using level_t = typename MHDModel::level_t;

    using ComputeFluxes_t          = ComputeFluxes<FVMethodStrategy, MHDModel>;
    using EulerUsingComputedFlux_t = EulerUsingComputedFlux<MHDModel>;

public:
    Euler(PHARE::initializer::PHAREDict const& dict)
        : compute_fluxes_{dict}
    {
    }

    void operator()(MHDModel& model, auto& state, auto& statenew, auto& fluxes, auto& bc,
                    level_t& level, double const currentTime, double const newTime,
                    double dt = std::nan(""))
    {
        if (std::isnan(dt))
            dt = newTime - currentTime;

        auto& god = amr::DEBUGOD<SimOpts{2, 1}>::INSTANCE();

        if (god.isActive())
        {
            auto& E  = state.E;
            using TF = std::decay_t<decltype(E)>;

            std::cout << "DEBUGOD: SolverMHD::euler before compute_fluxes_ "
                         "filling at level "
                      << level.getLevelNumber() << "\n";


            {
                auto jesus = god.template inspect<TF>({0.3926, 0.321}, {0.4026, 0.321},
                                                      std::string("mhd_state_E"),
                                                      std::string("mhd_state_E_z"));
                god.print(jesus);
            }
        }

        compute_fluxes_(model, state, fluxes, bc, level, newTime);

        if (god.isActive())
        {
            auto& E  = state.E;
            using TF = std::decay_t<decltype(E)>;

            std::cout << "DEBUGOD: SolverMHD::euler after compute_fluxes_ "
                         "filling at level "
                      << level.getLevelNumber() << "\n";


            {
                auto jesus = god.template inspect<TF>({0.3926, 0.321}, {0.4026, 0.321},
                                                      std::string("mhd_state_E"),
                                                      std::string("mhd_state_E_z"));
                god.print(jesus);
            }
            {
                auto jesus = god.template inspect<TF>({0.395, 0.321}, {0.4, 0.321},
                                                      std::string("mhd_state_B"),
                                                      std::string("mhd_state_B_x"));
                god.print(jesus);
            }
        }

        euler_using_computed_flux_(model, state, statenew, state.E, fluxes, bc, level, newTime, dt);

        if (god.isActive())
        {
            auto& E  = state.E;
            using TF = std::decay_t<decltype(E)>;

            std::cout << "DEBUGOD: SolverMHD::euler after euler_using_computed_flux_ "
                         "filling at level "
                      << level.getLevelNumber() << "\n";


            {
                auto jesus = god.template inspect<TF>({0.3926, 0.321}, {0.4026, 0.321},
                                                      std::string("mhd_state_E"),
                                                      std::string("mhd_state_E_z"));
                god.print(jesus);
            }
            {
                auto jesus = god.template inspect<TF>({0.395, 0.321}, {0.4, 0.321},
                                                      std::string("mhd_state_B"),
                                                      std::string("mhd_state_B_x"));
                god.print(jesus);
            }
        }
    }

    void registerResources(MHDModel& model) { compute_fluxes_.registerResources(model); }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        compute_fluxes_.allocate(model, patch, allocateTime);
    }

private:
    ComputeFluxes_t compute_fluxes_;
    EulerUsingComputedFlux_t euler_using_computed_flux_;
};
} // namespace PHARE::solver

#endif
