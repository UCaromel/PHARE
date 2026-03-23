#ifndef PHARE_SIMULATOR_OPTIONS_HPP
#define PHARE_SIMULATOR_OPTIONS_HPP

#include "core/utilities/meta/meta_utilities.hpp"

#include <cstddef>

namespace PHARE
{

// if mhd is off, use default empty objects
namespace MHDOpts
{
    enum class TimeIntegratorType : uint8_t { Default, Euler, TVDRK2, TVDRK3, SSPRK4_5, count };
    enum class ReconstructionType : uint8_t { Default, Constant, Linear, WENO3, WENOZ, MP5, count };
    enum class SlopeLimiterType : uint8_t { None, VanLeer, MinMod, count };
    enum class RiemannSolverType : uint8_t { Default, Rusanov, HLL, HLLD, count };

}; // namespace MHDOpts

// Unified SimOpts structure that can represent Hybrid, MHD, or both.
// The actual model used is determined at runtime from Python configuration.
// For MHD-only permutations, interp_order and nbRefinedPart use sentinel values (0, 0)
// so Hybrid-only template paths can be compile-time disabled.
struct SimOpts
{
    std::size_t dimension    = 1;
    std::size_t interp_order = 1;

    std::size_t nbRefinedPart = core::defaultNbrRefinedParts(dimension, interp_order);

    MHDOpts::TimeIntegratorType time_integrator_type = MHDOpts::TimeIntegratorType::Default;
    MHDOpts::ReconstructionType reconstruction_type  = MHDOpts::ReconstructionType::Default;
    MHDOpts::SlopeLimiterType slope_limiter_type     = MHDOpts::SlopeLimiterType::None;
    MHDOpts::RiemannSolverType riemann_solver_type   = MHDOpts::RiemannSolverType::Default;
    bool Hall                                        = false;
    bool Resistivity                                 = false;
    bool HyperResistivity                            = false;

    // Runtime model detection
    // These methods provide a cleaner API than checking template traits
    constexpr bool has_hybrid_model() const
    {
        return interp_order > 0 && nbRefinedPart > 0;
    }

    constexpr bool has_mhd_model() const
    {
        return reconstruction_type != MHDOpts::ReconstructionType::Default;
    }
};


// Model detection type traits
// Hybrid-only: reconstruction_type == Default (no MHD reconstruction selected)
// MHD or Both: reconstruction_type != Default (MHD reconstruction active)
template<SimOpts opts>
struct is_hybrid_model
    : std::bool_constant<opts.reconstruction_type == MHDOpts::ReconstructionType::Default>
{
};

template<SimOpts opts>
inline constexpr bool is_hybrid_v = is_hybrid_model<opts>::value;

template<SimOpts opts>
inline constexpr bool is_mhd_v = !is_hybrid_v<opts>;


// Sentinel types for unused model components
// Used to avoid template instantiation when model doesn't require certain types
struct NoSplitter
{
    static constexpr std::string_view name = "NoSplitter";
    
    // Dummy member types to satisfy template requirements
    // These are never actually used at runtime for MHD-only simulations
    struct InteriorParticleRefineOp {};
    struct CoarseToFineRefineOpOld {};
    struct CoarseToFineRefineOpNew {};
};

struct NoRefinementParams
{
    static constexpr std::string_view name = "NoRefinementParams";
    
    // Dummy member types to satisfy HybridHybridMessengerStrategy template
    using InteriorParticleRefineOp = NoSplitter::InteriorParticleRefineOp;
    using CoarseToFineRefineOpOld = NoSplitter::CoarseToFineRefineOpOld;
    using CoarseToFineRefineOpNew = NoSplitter::CoarseToFineRefineOpNew;
};

struct NoReconstruction
{
    static constexpr std::string_view name = "NoReconstruction";
};

struct NoRiemannSolver
{
    static constexpr std::string_view name = "NoRiemannSolver";
};


} // namespace PHARE

#endif // PHARE_SIMULATOR_OPTIONS_HPP
