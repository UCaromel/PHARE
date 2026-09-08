#ifndef PHARE_REFINEMENT_CONFIG_HPP
#define PHARE_REFINEMENT_CONFIG_HPP

#include "initializer/data_provider.hpp"
#include "amr/data/field/refine/field_refiner_kernel.hpp"

#include <stdexcept>
#include <string>

namespace PHARE::amr
{

/**
 * @brief Runtime selection of the field-refinement order.
 *
 * The refine-op members of the messengers are built from the runtime kernels
 * (makeRefineKernel / makeMagneticRefineKernel), whose stencil is selected by this order.
 * FieldRefinementOrder::Linear and FieldRefinementOrder::Cubic are the supported typed values.
 * Distinct from the EXISTING particle split-operator template param named RefinementParams
 * (MessengerFactory / HybridHybridMessengerStrategy) — do not conflate.
 */
struct RefinementConfig
{
    FieldRefinementOrder order = FieldRefinementOrder::Linear;

    //! Read optional field-refinement selection from the dict. Absent path nodes select Linear.
    //! This is the only place a raw dictionary value is validated.
    RefinementConfig static FROM(PHARE::initializer::PHAREDict const& dict)
    {
        PHARE::amr::RefinementConfig config;
        auto const rawOrder = cppdict::get_value(dict, "simulation/AMR/refinement/order", int{2});
        if (rawOrder != 2 && rawOrder != 4)
            throw std::runtime_error("unsupported field refinement order: "
                                     + std::to_string(rawOrder) + " (supported: 2, 4)");
        config.order = static_cast<FieldRefinementOrder>(rawOrder);
        return config;
    }
};

} // namespace PHARE::amr

#endif
