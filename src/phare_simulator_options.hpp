#ifndef PHARE_SIMULATOR_OPTIONS_HPP
#define PHARE_SIMULATOR_OPTIONS_HPP

#include "core/utilities/meta/meta_utilities.hpp"

#include <cstddef>

namespace PHARE
{

// if mhd is off, use default empty objects
struct MHDOpts
{
    enum class TimeIntegratorType : uint8_t { Euler, TVDRK2, TVDRK3, SSPRK4_5, count };
    enum class ReconstructionType : uint8_t { Constant, Linear, WENO3, WENOZ, MP5, count };
    enum class SlopeLimiterType : uint8_t { VanLeer, MinMod, count };
    enum class RiemannSolverType : uint8_t { Rusanov, HLL, HLLD, count };

    TimeIntegratorType time_integrator_type = TimeIntegratorType::TVDRK2;
    ReconstructionType reconstruction_type  = ReconstructionType::Linear;
    SlopeLimiterType slope_limiter_type     = SlopeLimiterType::VanLeer;
    RiemannSolverType riemann_solver_type   = RiemannSolverType::Rusanov;
    bool Hall                               = false;
    bool Resistivity                        = false;
    bool HyperResistivity                   = false;
};

struct SimOpts
{
    std::size_t dimension    = 1;
    std::size_t interp_order = 1;

    std::size_t nbRefinedPart = core::defaultNbrRefinedParts(dimension, interp_order);

    MHDOpts mhd_opts;
};



} // namespace PHARE

#endif // PHARE_SIMULATOR_OPTIONS_HPP
