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
 * The value follows the compiled MHD order (refinementConfigFor), it is not an input. Not to be
 * confused with RefinementParams, the particle split-operator template parameter of
 * MessengerFactory and HybridHybridMessengerStrategy.
 */
struct RefinementConfig
{
    FieldRefinementOrder order = FieldRefinementOrder::Linear;
};

} // namespace PHARE::amr

#endif
