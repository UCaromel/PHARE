#ifndef PHARE_REFINEMENT_CONFIG_HPP
#define PHARE_REFINEMENT_CONFIG_HPP

#include <string>

namespace PHARE::amr
{

/**
 * @brief Runtime selection of the field-refinement operators.
 *
 * order == 0 (default) routes every refine-op member to the unchanged legacy templated
 * FieldRefineOperator<...,Policy> (byte-identical to master). order == 2 (Linear) / 4 (Cubic)
 * routes the composable members to the runtime KernelFieldRefineOperator built from
 * makeRefineKernel / makeMagneticRefineKernel. limiter selects the limited Primitive-B slope
 * ("none" | "minmod" | "vanleer"); empty == "none". Distinct from the EXISTING particle
 * split-operator template param named RefinementParams (MessengerFactory /
 * HybridHybridMessengerStrategy) — do not conflate.
 */
struct RefinementConfig
{
    int order           = 0;  // 0 => legacy operators
    std::string limiter = ""; // "" / "none" | "minmod" | "vanleer"
};

} // namespace PHARE::amr

#endif
