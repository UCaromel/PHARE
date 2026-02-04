#ifndef PHARE_SIMULATOR_OPTIONS_HPP
#define PHARE_SIMULATOR_OPTIONS_HPP

#include "core/utilities/meta/meta_utilities.hpp"

#include <cstddef>

namespace PHARE
{

// if mhd is off, use default empty objects
struct MHDOpts
{
    enum class TimeIntegratorType : uint8_t { Default, Euler, TVDRK2, TVDRK3, SSPRK4_5, count };
    enum class ReconstructionType : uint8_t { Default, Constant, Linear, WENO3, WENOZ, MP5, count };
    enum class SlopeLimiterType : uint8_t { None, VanLeer, MinMod, count };
    enum class RiemannSolverType : uint8_t { Default, Rusanov, HLL, HLLD, count };

    TimeIntegratorType time_integrator_type = TimeIntegratorType::Default;
    ReconstructionType reconstruction_type  = ReconstructionType::Default;
    SlopeLimiterType slope_limiter_type     = SlopeLimiterType::None;
    RiemannSolverType riemann_solver_type   = RiemannSolverType::Default;
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

    constexpr SimOpts(std::size_t dim, std::size_t interp)
        : dimension{dim}
        , interp_order{interp}
        , nbRefinedPart{core::defaultNbrRefinedParts(dim, interp)}
        , mhd_opts{}
    {
    }

    constexpr SimOpts(std::size_t dim, std::size_t interp, std::size_t nbRef)
        : dimension{dim}
        , interp_order{interp}
        , nbRefinedPart{nbRef}
        , mhd_opts{}
    {
    }

    constexpr SimOpts(std::size_t dim, std::size_t interp, std::size_t nbRef,
                      MHDOpts::TimeIntegratorType ti, MHDOpts::ReconstructionType rec,
                      MHDOpts::SlopeLimiterType sl, MHDOpts::RiemannSolverType rs, bool hall,
                      bool resistivity, bool hyper_resistivity)
        : dimension{dim}
        , interp_order{interp}
        , nbRefinedPart{nbRef}
        , mhd_opts{ti, rec, sl, rs, hall, resistivity, hyper_resistivity}
    {
    }
};



} // namespace PHARE

#endif // PHARE_SIMULATOR_OPTIONS_HPP
