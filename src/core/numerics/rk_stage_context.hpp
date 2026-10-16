#ifndef PHARE_CORE_NUMERICS_RK_STAGE_CONTEXT_HPP
#define PHARE_CORE_NUMERICS_RK_STAGE_CONTEXT_HPP

#include <cstddef>

namespace PHARE::core
{
// Where a Runge-Kutta stage sits in its step, for the coarse-fine temporal transfer.
struct RKStageContext
{
    static constexpr std::size_t finalStage = 4;

    std::size_t stageIndex;
    double stepStartTime;
    double stepDuration;

    // a fill outside the stages of a step (final state, reflux): no stage duration to record
    static constexpr RKStageContext endOfStep(double const time) { return {finalStage, time, 0.0}; }
};
} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_RK_STAGE_CONTEXT_HPP
