#ifndef PHARE_AMR_REFINE_PROBE_HPP
#define PHARE_AMR_REFINE_PROBE_HPP

#include <atomic>
#include <cstdint>
#include <ostream>
#include <string>


namespace PHARE::amr
{

/**
 * @brief Counters of a limited refinement kernel: how many dual rows a slope rule evaluated, how
 * many of them it changed from the unlimited slope, and how many coarse cells the energy row left
 * to the per-variable Etot row (non-positive or non-finite density).
 *
 * One probe per operator; the holder reports it.
 */
struct RefineProbe
{
    std::atomic<std::uint64_t> rows{0};
    std::atomic<std::uint64_t> limitedRows{0};
    std::atomic<std::uint64_t> energyRowSkipped{0};

    void row(double const slope, double const unlimited) noexcept
    {
        rows.fetch_add(1, std::memory_order_relaxed);
        if (slope != unlimited)
            limitedRows.fetch_add(1, std::memory_order_relaxed);
    }

    void skippedEnergyRow() noexcept { energyRowSkipped.fetch_add(1, std::memory_order_relaxed); }

    void report(std::string const& label, std::ostream& os) const
    {
        auto const n       = rows.load(std::memory_order_relaxed);
        auto const limited = limitedRows.load(std::memory_order_relaxed);
        os << "refine probe " << label << ": " << limited << " of " << n << " rows limited";
        if (n > 0)
            os << " (" << 100. * static_cast<double>(limited) / static_cast<double>(n) << " %)";
        os << ", energy row skipped on " << energyRowSkipped.load(std::memory_order_relaxed)
           << " coarse cells (non-positive density)\n";
    }
};

} // namespace PHARE::amr

#endif
