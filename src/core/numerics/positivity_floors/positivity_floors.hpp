#ifndef PHARE_CORE_NUMERICS_POSITIVITY_FLOORS_HPP
#define PHARE_CORE_NUMERICS_POSITIVITY_FLOORS_HPP

#include "initializer/data_provider.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <ostream>

namespace PHARE::core
{
// Sites where a floor may fire. S3 (Hall CT denominators) is deferred pending measurement of
// whether it still fires after ToPrimitive+Reconstruction floors are in place.
enum class FloorSite : std::size_t { ToPrimitive, Reconstruction, PostReflux, Count };

// Runtime-configurable stopgap positivity floors. Defaults to fully off (bit-identical to no
// floors at all) so existing simulations/tests are unaffected unless explicitly enabled.
struct FloorParams
{
    bool enabled          = false;
    double density_floor  = 0.0; // eps_stab for rho, absolute, sized per-run from measurement
    double pressure_floor = 0.0; // eps_stab for P, absolute, sized per-run from measurement

    static FloorParams FROM(initializer::PHAREDict const& dict)
    {
        return {cppdict::get_value(dict, "positivity_floors/enabled", false),
                cppdict::get_value(dict, "positivity_floors/density_floor", 0.0),
                cppdict::get_value(dict, "positivity_floors/pressure_floor", 0.0)};
    }
};

// eps_adm: admissibility floor, derived from the local energy scale, never user-tuned. Keeps
// arithmetic defined against roundoff once floors are enabled (C=4 matches the corroborated
// 4*u*Etot margin measured on feature/pcp).
inline double admissibilityFloor(double localScale)
{
    return 4.0 * std::numeric_limits<double>::epsilon() * std::abs(localScale);
}

// Per-rank accumulator of floor firings, reset and reported once per SolverMHD::advanceLevel.
// A floor that fires silently is a lie in the mass/energy budget, so every firing is counted.
class FloorDiagnostics
{
public:
    static FloorDiagnostics& instance()
    {
        static FloorDiagnostics diag;
        return diag;
    }

    void record(FloorSite site, bool isDensity, double rawValue, double flooredValue)
    {
        auto& s = stats_[index_(site, isDensity)];
        ++s.count;
        s.sumDelta += (flooredValue - rawValue);
        s.minRaw = std::min(s.minRaw, rawValue);
    }

    void reset() { stats_.fill(Stats{}); }

    void report(std::ostream& os, int levelNumber, double time) const
    {
        constexpr char const* siteNames[]  = {"ToPrimitive", "Reconstruction", "PostReflux"};
        constexpr char const* quantities[] = {"rho", "P"};

        for (std::size_t site = 0; site < static_cast<std::size_t>(FloorSite::Count); ++site)
        {
            for (std::size_t q = 0; q < 2; ++q)
            {
                auto const& s = stats_[site * 2 + q];
                if (s.count == 0)
                    continue;

                os << "[floor] level=" << levelNumber << " t=" << time
                   << " site=" << siteNames[site] << " qty=" << quantities[q]
                   << " count=" << s.count << " sumDelta=" << s.sumDelta << " minRaw=" << s.minRaw
                   << '\n';
            }
        }
    }

private:
    struct Stats
    {
        std::size_t count = 0;
        double sumDelta   = 0.0;
        double minRaw     = std::numeric_limits<double>::max();
    };

    static std::size_t index_(FloorSite site, bool isDensity)
    {
        return static_cast<std::size_t>(site) * 2 + (isDensity ? 0 : 1);
    }

    std::array<Stats, 2 * static_cast<std::size_t>(FloorSite::Count)> stats_{};
};

// Floors x from below at max(eps_adm(scale), params floor) when params.enabled; records the
// firing to the diagnostics accumulator. Returns whether it fired.
inline bool floorScalarInPlace(double& x, double scale, double configFloor, FloorSite site,
                               bool isDensity, FloorParams const& params)
{
    if (!params.enabled)
        return false;

    double const eps = std::max(admissibilityFloor(scale), configFloor);
    if (x >= eps)
        return false;

    FloorDiagnostics::instance().record(site, isDensity, x, eps);
    x = eps;
    return true;
}
} // namespace PHARE::core

#endif
