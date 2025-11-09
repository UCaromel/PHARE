#ifndef PHARE_MHD_RESOLVER_HPP
#define PHARE_MHD_RESOLVER_HPP

#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"

namespace PHARE
{
template<template<template<typename, typename> typename, typename> typename TimeIntegrator,
         template<typename, typename> typename Reconstruction, typename SlopeLimiter,
         template<bool> typename RiemannSolver, template<bool, bool, bool> typename Equations,
         bool Hall, bool Resistivity, bool HyperResistivity>
struct MHDResolver
{
    using Equations_t = Equations<Hall, Resistivity, HyperResistivity>;

    using RiemannSolver_t = RiemannSolver<Hall>;

    template<typename Layout>
    using Reconstruction_t = Reconstruction<Layout, SlopeLimiter>;

    template<typename Layout, typename MHDModel>
    using FVMethodStrategy
        = core::Godunov<Layout, MHDModel, Reconstruction_t, RiemannSolver_t, Equations_t>;

    template<typename MHDModel>
    using TimeIntegrator_t = TimeIntegrator<FVMethodStrategy, MHDModel>;
};
} // namespace PHARE

#endif // PHARE_MHD_RESOLVER_HPP
