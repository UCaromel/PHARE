#ifndef PHARE_REFINEMENT_CONFIG_HPP
#define PHARE_REFINEMENT_CONFIG_HPP

#include "amr/data/field/refine/field_refiner_kernel.hpp"

namespace PHARE::amr
{

/**
 * @brief The field-refinement order the messengers build their refine-ops with.
 *
 * The refine-op members of the messengers are built from the runtime kernels
 * (makeRefineKernel / makeMagneticRefineKernel), whose stencil is selected by this order.
 * The value is derived from the compile-time solver profile, not read from the input: a dictionary
 * value could only ever contradict the profile the build was compiled for. Distinct from the
 * EXISTING particle split-operator template param named RefinementParams (MessengerFactory /
 * HybridHybridMessengerStrategy) — do not conflate.
 */
struct RefinementConfig
{
    FieldRefinementOrder order = FieldRefinementOrder::Linear;
};

} // namespace PHARE::amr

#endif
