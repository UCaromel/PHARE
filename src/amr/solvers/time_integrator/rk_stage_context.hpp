#ifndef PHARE_RK_STAGE_CONTEXT_HPP
#define PHARE_RK_STAGE_CONTEXT_HPP

#include <cstddef>

namespace PHARE::solver
{
struct RKStageContext
{
    std::size_t stageIndex;
    double stepStartTime;
    double stepDuration;
};
} // namespace PHARE::solver

#endif // PHARE_RK_STAGE_CONTEXT_HPP
