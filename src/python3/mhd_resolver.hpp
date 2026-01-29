#ifndef PHARE_MHD_RESOLVER_HPP
#define PHARE_MHD_RESOLVER_HPP

#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "phare_simulator_options.hpp"

#include "amr/solvers/time_integrator/euler_integrator.hpp"
#include "amr/solvers/time_integrator/tvdrk2_integrator.hpp"
#include "amr/solvers/time_integrator/tvdrk3_integrator.hpp"
#include "amr/solvers/time_integrator/ssprk4_5_integrator.hpp"

#include "core/numerics/reconstructions/constant.hpp"
#include "core/numerics/reconstructions/linear.hpp"
#include "core/numerics/reconstructions/weno3.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/numerics/reconstructions/mp5.hpp"

#include "core/numerics/slope_limiters/min_mod.hpp"
#include "core/numerics/slope_limiters/van_leer.hpp"

#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/riemann_solvers/hll.hpp"
#include "core/numerics/riemann_solvers/hlld.hpp"

#include "core/numerics/MHD_equations/MHD_equations.hpp"
#include "python3/mhd_defaults/default_mhd_time_stepper.hpp"
#include <amr/physical_models/mhd_model.hpp>

namespace PHARE
{

template<auto opts, typename MHDModel>
struct MHDResolver
{
    using TimeIntegratorType = MHDOpts::TimeIntegratorType;
    using ReconstructionType = MHDOpts::ReconstructionType;
    using SlopeLimiterType   = MHDOpts::SlopeLimiterType;
    using RiemannSolverType  = MHDOpts::RiemannSolverType;

    // Selectors

    template<TimeIntegratorType T>
    struct TimeIntegratorSelector;

    template<ReconstructionType T>
    struct ReconstructionSelector;

    template<ReconstructionType R, SlopeLimiterType S>
    struct SlopeLimiterSelector;

    template<RiemannSolverType T>
    struct RiemannSolverSelector;

    template<>
    struct TimeIntegratorSelector<TimeIntegratorType::Default>
    {
        template<template<typename> typename FVmethod>
        using type = DefaultTimeIntegrator<FVmethod, MHDModel>;
    };

    template<>
    struct TimeIntegratorSelector<TimeIntegratorType::Euler>
    {
        template<template<typename> typename FVmethod>
        using type = solver::EulerIntegrator<FVmethod, MHDModel>;
    };

    template<>
    struct TimeIntegratorSelector<TimeIntegratorType::TVDRK2>
    {
        template<template<typename> typename FVmethod>
        using type = solver::TVDRK2Integrator<FVmethod, MHDModel>;
    };

    template<>
    struct TimeIntegratorSelector<TimeIntegratorType::TVDRK3>
    {
        template<template<typename> typename FVmethod>
        using type = solver::TVDRK3Integrator<FVmethod, MHDModel>;
    };

    template<>
    struct TimeIntegratorSelector<TimeIntegratorType::SSPRK4_5>
    {
        template<template<typename> typename FVmethod>
        using type = solver::SSPRK4_5Integrator<FVmethod, MHDModel>;
    };

    template<>
    struct ReconstructionSelector<ReconstructionType::Default>
    {
        template<typename GridLayout, typename SlopeLimiter>
        using type = DefaultReconstruction<GridLayout, SlopeLimiter>;
    };

    template<>
    struct ReconstructionSelector<ReconstructionType::Constant>
    {
        template<typename GridLayout, typename SlopeLimiter>
        using type = core::ConstantReconstruction<GridLayout, SlopeLimiter>;
    };

    template<>
    struct ReconstructionSelector<ReconstructionType::Linear>
    {
        template<typename GridLayout, typename SlopeLimiter>
        using type = core::LinearReconstruction<GridLayout, SlopeLimiter>;
    };

    template<>
    struct ReconstructionSelector<ReconstructionType::WENO3>
    {
        template<typename GridLayout, typename SlopeLimiter>
        using type = core::WENO3Reconstruction<GridLayout, SlopeLimiter>;
    };

    template<>
    struct ReconstructionSelector<ReconstructionType::WENOZ>
    {
        template<typename GridLayout, typename SlopeLimiter>
        using type = core::WENOZReconstruction<GridLayout, SlopeLimiter>;
    };

    template<>
    struct ReconstructionSelector<ReconstructionType::MP5>
    {
        template<typename GridLayout, typename SlopeLimiter>
        using type = core::MP5Reconstruction<GridLayout, SlopeLimiter>;
    };

    template<ReconstructionType R, SlopeLimiterType S>
    struct SlopeLimiterSelector
    {
        using type = void;
    };

    template<>
    struct SlopeLimiterSelector<ReconstructionType::Linear, SlopeLimiterType::VanLeer>
    {
        using type = core::VanLeerLimiter;
    };

    template<>
    struct SlopeLimiterSelector<ReconstructionType::Linear, SlopeLimiterType::MinMod>
    {
        using type = core::MinModLimiter;
    };

    template<>
    struct RiemannSolverSelector<RiemannSolverType::Default>
    {
        template<bool Hall>
        using type = DefaultRiemannSolver<Hall>;
    };

    template<>
    struct RiemannSolverSelector<RiemannSolverType::Rusanov>
    {
        template<bool Hall>
        using type = core::Rusanov<Hall>;
    };

    template<>
    struct RiemannSolverSelector<RiemannSolverType::HLL>
    {
        template<bool Hall>
        using type = core::HLL<Hall>;
    };

    template<>
    struct RiemannSolverSelector<RiemannSolverType::HLLD>
    {
        template<bool Hall>
        using type = core::HLLD<Hall>;
    };

    // Get the types from opts

    static constexpr bool Hall             = opts.mhd_opts.Hall;
    static constexpr bool Resistivity      = opts.mhd_opts.Resistivity;
    static constexpr bool HyperResistivity = opts.mhd_opts.HyperResistivity;

    using SlopeLimiter = SlopeLimiterSelector<opts.mhd_opts.reconstruction_type,
                                              opts.mhd_opts.slope_limiter_type>::type;

    template<bool HallFlag>
    using RiemannSolver
        = RiemannSolverSelector<opts.mhd_opts.riemann_solver_type>::template type<HallFlag>;

    template<typename Layout, typename Limiter>
    using Reconstruction
        = ReconstructionSelector<opts.mhd_opts.reconstruction_type>::template type<Layout, Limiter>;

    template<template<typename> typename FVMethod>
    using MHDTimeStepper
        = TimeIntegratorSelector<opts.mhd_opts.time_integrator_type>::template type<FVMethod>;

    // Resolution

    using Equations_t = core::MHDEquations<Hall, Resistivity, HyperResistivity>;

    using RiemannSolver_t = RiemannSolver<Hall>;

    template<typename Layout>
    using Reconstruction_t = Reconstruction<Layout, SlopeLimiter>;

    template<typename Layout>
    using FVMethodStrategy
        = core::Godunov<Layout, MHDModel, Reconstruction_t, RiemannSolver_t, Equations_t>;

    using MHDTimeStepper_t = MHDTimeStepper<FVMethodStrategy>;
};
} // namespace PHARE

#endif // PHARE_MHD_RESOLVER_HPP
