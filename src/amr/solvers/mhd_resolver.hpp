#ifndef PHARE_AMR_SOLVERS_MHD_RESOLVER_HPP
#define PHARE_AMR_SOLVERS_MHD_RESOLVER_HPP

#include "phare_simulator_options.hpp"

#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"

#include "amr/solvers/time_integrator/euler_integrator.hpp"
#include "amr/solvers/time_integrator/tvdrk2_integrator.hpp"
#include "amr/solvers/time_integrator/tvdrk3_integrator.hpp"
#include "amr/solvers/time_integrator/ssprk4_5_integrator.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"

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

namespace PHARE::solver
{

template<MHDOpts::MHDOrder Order, typename MHDModel>
struct MHDProfileTraits;

template<typename MHDModel>
struct MHDProfileTraits<MHDOpts::MHDOrder::O2, MHDModel>
{
    using PointValueApproximation = SecondOrderPointValueApproximation<MHDModel>;
    static constexpr std::size_t amrSpatialOrder = 2;
};

template<typename MHDModel>
struct MHDProfileTraits<MHDOpts::MHDOrder::O4, MHDModel>
{
    using PointValueApproximation = FourthOrderPointValueApproximation<MHDModel>;
    static constexpr std::size_t amrSpatialOrder = 4;
};

// Selectors

template<MHDOpts::TimeIntegratorType T, typename MHDModel, typename PointValueApproximation>
struct TimeIntegratorSelector;

template<MHDOpts::ReconstructionType T, MHDOpts::MHDOrder Order>
struct ReconstructionSelector;

template<MHDOpts::ReconstructionType R, MHDOpts::SlopeLimiterType S>
struct SlopeLimiterSelector;

template<MHDOpts::RiemannSolverType T>
struct RiemannSolverSelector;

template<typename MHDModel, typename PointValueApproximation>
struct TimeIntegratorSelector<MHDOpts::TimeIntegratorType::Euler, MHDModel,
                              PointValueApproximation>
{
    template<typename FVmethod>
    using type = EulerIntegrator<FVmethod, MHDModel, PointValueApproximation>;
};

template<typename MHDModel, typename PointValueApproximation>
struct TimeIntegratorSelector<MHDOpts::TimeIntegratorType::TVDRK2, MHDModel,
                              PointValueApproximation>
{
    template<typename FVmethod>
    using type = TVDRK2Integrator<FVmethod, MHDModel, PointValueApproximation>;
};

template<typename MHDModel, typename PointValueApproximation>
struct TimeIntegratorSelector<MHDOpts::TimeIntegratorType::TVDRK3, MHDModel,
                              PointValueApproximation>
{
    template<typename FVmethod>
    using type = TVDRK3Integrator<FVmethod, MHDModel, PointValueApproximation>;
};

template<typename MHDModel, typename PointValueApproximation>
struct TimeIntegratorSelector<MHDOpts::TimeIntegratorType::SSPRK4_5, MHDModel,
                              PointValueApproximation>
{
    template<typename FVmethod>
    using type = SSPRK4_5Integrator<FVmethod, MHDModel, PointValueApproximation>;
};

template<MHDOpts::MHDOrder Order>
struct ReconstructionSelector<MHDOpts::ReconstructionType::Constant, Order>
{
    template<typename GridLayout, typename SlopeLimiter>
    using type = core::ConstantReconstruction<GridLayout, SlopeLimiter>;
};

template<MHDOpts::MHDOrder Order>
struct ReconstructionSelector<MHDOpts::ReconstructionType::Linear, Order>
{
    template<typename GridLayout, typename SlopeLimiter>
    using type = core::LinearReconstruction<GridLayout, SlopeLimiter>;
};

template<MHDOpts::MHDOrder Order>
struct ReconstructionSelector<MHDOpts::ReconstructionType::WENO3, Order>
{
    template<typename GridLayout, typename SlopeLimiter>
    using type = core::WENO3Reconstruction<GridLayout, SlopeLimiter>;
};

template<MHDOpts::MHDOrder Order>
struct ReconstructionSelector<MHDOpts::ReconstructionType::WENOZ, Order>
{
    template<typename GridLayout, typename SlopeLimiter>
    using type = core::WENOZReconstruction<GridLayout, SlopeLimiter,
                                           Order == MHDOpts::MHDOrder::O4>;
};

template<MHDOpts::MHDOrder Order>
struct ReconstructionSelector<MHDOpts::ReconstructionType::MP5, Order>
{
    template<typename GridLayout, typename SlopeLimiter>
    using type
        = core::MP5Reconstruction<GridLayout, SlopeLimiter, Order == MHDOpts::MHDOrder::O4>;
};

// SlopeLimiterSelector is only declared above, never defined: every (reconstruction, limiter) pair
// we support must be listed explicitly below, and any pair that is not listed fails to compile
// rather than silently resolving to something. That is how a half-configured MHD build is caught --
// e.g. reconstruction set but limiter left at MHDOff has no specialization, so it does not build.
// Only Linear actually consults a limiter; the others still need an entry for None, resolving to
// void, to say "this combination is valid, the limiter is simply unused".
template<>
struct SlopeLimiterSelector<MHDOpts::ReconstructionType::Constant, MHDOpts::SlopeLimiterType::None>
{
    using type = void;
};

template<>
struct SlopeLimiterSelector<MHDOpts::ReconstructionType::WENO3, MHDOpts::SlopeLimiterType::None>
{
    using type = void;
};

template<>
struct SlopeLimiterSelector<MHDOpts::ReconstructionType::WENOZ, MHDOpts::SlopeLimiterType::None>
{
    using type = void;
};

template<>
struct SlopeLimiterSelector<MHDOpts::ReconstructionType::MP5, MHDOpts::SlopeLimiterType::None>
{
    using type = void;
};

template<>
struct SlopeLimiterSelector<MHDOpts::ReconstructionType::Linear, MHDOpts::SlopeLimiterType::VanLeer>
{
    using type = core::VanLeerLimiter;
};

template<>
struct SlopeLimiterSelector<MHDOpts::ReconstructionType::Linear, MHDOpts::SlopeLimiterType::MinMod>
{
    using type = core::MinModLimiter;
};

template<>
struct RiemannSolverSelector<MHDOpts::RiemannSolverType::Rusanov>
{
    template<bool UpwindWhistler>
    using type = core::Rusanov<UpwindWhistler>;
};

template<>
struct RiemannSolverSelector<MHDOpts::RiemannSolverType::HLL>
{
    template<bool UpwindWhistler>
    using type = core::HLL<UpwindWhistler>;
};

template<>
struct RiemannSolverSelector<MHDOpts::RiemannSolverType::HLLD>
{
    template<bool UpwindWhistler>
    using type = core::HLLD<UpwindWhistler>;
};

template<auto opts, typename MHDModel>
struct MHDResolver
{
    // Get the types from opts

    static constexpr bool Hall             = opts.Hall;
    static constexpr bool Resistivity      = opts.Resistivity;
    static constexpr bool HyperResistivity = opts.HyperResistivity;
    using ProfileTraits = MHDProfileTraits<opts.mhd_order, MHDModel>;
    using PointValueApproximation = typename ProfileTraits::PointValueApproximation;
    static constexpr std::size_t amrSpatialOrder = ProfileTraits::amrSpatialOrder;


    using SlopeLimiter
        = SlopeLimiterSelector<opts.reconstruction_type, opts.slope_limiter_type>::type;

    // The dispersive Hall branch is damped either by the upwind whistler speed in the wave fan or
    // by hyper-resistivity, never by both: they act on the same branch. The fan is the default
    // because its dissipation shrinks with the reconstruction's interface jump, so it follows the
    // scheme's order instead of capping it.
    static constexpr bool UpwindWhistler = Hall && !HyperResistivity;

    template<bool WhistlerFlag>
    using RiemannSolver
        = RiemannSolverSelector<opts.riemann_solver_type>::template type<WhistlerFlag>;

    template<typename Layout, typename Limiter>
    using Reconstruction = ReconstructionSelector<
        opts.reconstruction_type, opts.mhd_order>::template type<Layout, Limiter>;

    template<typename FVMethod>
    using MHDTimeStepper
        = typename TimeIntegratorSelector<opts.time_integrator_type, MHDModel,
                                          PointValueApproximation>::template type<FVMethod>;

    // Resolution

    using GridLayout = MHDModel::gridlayout_type;

    using Equations_t = core::MHDEquations<Hall, Resistivity, HyperResistivity>;

    using RiemannSolver_t = RiemannSolver<UpwindWhistler>;

    template<typename Layout>
    using Reconstruction_t = Reconstruction<Layout, SlopeLimiter>;

    using FVMethodStrategy
        = core::Godunov<GridLayout, Reconstruction_t, RiemannSolver_t, Equations_t>;

    using MHDTimeStepper_t = MHDTimeStepper<FVMethodStrategy>;
};

} // namespace PHARE::solver

#endif // PHARE_AMR_SOLVERS_MHD_RESOLVER_HPP
