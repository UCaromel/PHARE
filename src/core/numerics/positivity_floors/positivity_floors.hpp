#ifndef PHARE_CORE_NUMERICS_POSITIVITY_FLOORS_HPP
#define PHARE_CORE_NUMERICS_POSITIVITY_FLOORS_HPP

#include "core/def.hpp"
#include "initializer/data_provider.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace PHARE::core
{
// Sites where a floor may fire. S3 (Hall CT denominators): measurement showed S1+S2 do not
// suppress it (the CT path reconstructs rhot a second time, via the raw scheme, downstream of
// S2's floor), so it is floored independently.
enum class FloorSite : std::size_t { ToPrimitive, Reconstruction, PostReflux, HallCT, Count };

// Single source of truth for site names, positionally indexed against FloorSite. Shared by
// FloorDiagnostics::report and floorScalarInPlace's non-finite throw message so an enum
// addition without a name here fails to compile (array size mismatch) instead of silently
// mislabelling.
inline constexpr std::array<char const*, static_cast<std::size_t>(FloorSite::Count)> floorSiteNames{
    "ToPrimitive", "Reconstruction", "PostReflux", "HallCT"};

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

    // A non-finite input is a corruption event, not a positivity event: it must never share
    // count/sumDelta with record() above, since (NaN - rawValue) would permanently poison
    // sumDelta for the rest of the level's report. See floorScalarInPlace's non-finite guard.
    void recordNonFinite(FloorSite site, bool isDensity)
    {
        ++nonfiniteCount_[index_(site, isDensity)];
    }

    void reset()
    {
        stats_.fill(Stats{});
        nonfiniteCount_.fill(0);
    }

    void report(std::ostream& os, int levelNumber, double time) const
    {
        constexpr char const* quantities[] = {"rho", "P"};

        for (std::size_t site = 0; site < static_cast<std::size_t>(FloorSite::Count); ++site)
        {
            for (std::size_t q = 0; q < 2; ++q)
            {
                auto const& s = stats_[site * 2 + q];
                if (s.count > 0)
                    os << "[floor] level=" << levelNumber << " t=" << time
                       << " site=" << floorSiteNames[site] << " qty=" << quantities[q]
                       << " count=" << s.count << " sumDelta=" << s.sumDelta
                       << " minRaw=" << s.minRaw << '\n';

                auto const nf = nonfiniteCount_[site * 2 + q];
                if (nf > 0)
                    os << "[floor-nonfinite] level=" << levelNumber << " t=" << time
                       << " site=" << floorSiteNames[site] << " qty=" << quantities[q]
                       << " count=" << nf << '\n';
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
    std::array<std::size_t, 2 * static_cast<std::size_t>(FloorSite::Count)> nonfiniteCount_{};
};

// Floors x from below at max(eps_adm(scale), params floor) when params.enabled; records the
// firing to the diagnostics accumulator. Returns whether it fired.
//
// `index` is whatever mesh index is available at the call site (used only to enrich the
// PHARE_DEBUG_DO throw message below; never dereferenced otherwise).
inline bool floorScalarInPlace(double& x, double scale, double configFloor, FloorSite site,
                               bool isDensity, FloorParams const& params, auto const& index)
{
    if (!params.enabled)
        return false;

    // A non-finite x is a corruption event, not a small-positive excursion: floors clamp from
    // below, and NaN >= eps is false, so without this guard a NaN silently falls through
    // unclamped (zero protection) while still being counted as a firing (poisons sumDelta via
    // NaN - NaN, inflates count). Must return before admissibilityFloor(scale) is computed: at
    // the reconstructor call sites scale IS x, so admissibilityFloor(NaN) = NaN and
    // std::max(NaN, configFloor) is NaN regardless of configFloor (the same order-dependent
    // std::max trap fixed in rusanov.hpp, one layer up).
    if (!std::isfinite(x))
    {
        FloorDiagnostics::instance().recordNonFinite(site, isDensity);
        PHARE_DEBUG_DO({
            std::ostringstream ss;
            ss << "Non-finite value at positivity floor site="
               << floorSiteNames[static_cast<std::size_t>(site)]
               << " qty=" << (isDensity ? "rho" : "P") << " x=" << x << " index=" << index;
            throw std::runtime_error(ss.str());
        })
        // Non-debug: count and continue, leave x non-finite. Overwriting with eps would launder
        // corruption into plausible-looking data and let the run complete with a physically
        // meaningless sheet; leaving it non-finite means the existing mhdNaNCheck_ (or an
        // upstream floor call reading this same value, e.g. PostReflux after Reconstruction)
        // still catches it downstream instead.
        return false;
    }

    double const eps = std::max(admissibilityFloor(scale), configFloor);
    if (x >= eps)
        return false;

    FloorDiagnostics::instance().record(site, isDensity, x, eps);
    x = eps;
    return true;
}
} // namespace PHARE::core

#endif
