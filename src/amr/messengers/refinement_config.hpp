#ifndef PHARE_REFINEMENT_CONFIG_HPP
#define PHARE_REFINEMENT_CONFIG_HPP

#include "initializer/data_provider.hpp"
#include "amr/data/field/refine/field_refiner_kernel.hpp"

#include <stdexcept>
#include <string>

namespace PHARE::amr
{

/**
 * @brief Runtime selection of the field-refinement order and limiter.
 *
 * The refine-op members of the messengers are built from the runtime kernels
 * (makeRefineKernel / makeMagneticRefineKernel), whose stencil is selected by this order and
 * limiter.
 * FieldRefinementOrder::Linear (order 2) is the only supported value for now; this struct is the
 * extension point for higher orders. Distinct from the EXISTING particle split-operator template
 * param named RefinementParams (MessengerFactory / HybridHybridMessengerStrategy) — do not
 * conflate.
 */
struct RefinementConfig
{
    FieldRefinementOrder order     = FieldRefinementOrder::Linear;
    FieldRefinementLimiter limiter = FieldRefinementLimiter::None;

    //! Read the optional field-refinement selection from the dict. Absent "AMR"/"refinement"/
    //! "order" or "limiter" nodes at any level of the path ⇒ the RefinementConfig defaults. Only
    //! order 2 (Linear) is supported; this is the one place raw dict values are validated.
    RefinementConfig static FROM(PHARE::initializer::PHAREDict const& dict)
    {
        PHARE::amr::RefinementConfig config;
        auto const rawOrder = cppdict::get_value(dict, "simulation/AMR/refinement/order", int{2});
        if (rawOrder != 2)
            throw std::runtime_error("unsupported field refinement order: "
                                     + std::to_string(rawOrder) + " (only 2 is supported)");
        config.order = static_cast<FieldRefinementOrder>(rawOrder);

        auto const rawLimiter
            = cppdict::get_value(dict, "simulation/AMR/refinement/limiter", std::string{"none"});
        if (rawLimiter == "weno")
            config.limiter = FieldRefinementLimiter::WENO;
        else if (rawLimiter != "none")
            throw std::runtime_error("unsupported field refinement limiter: " + rawLimiter
                                     + " (none or weno)");
        return config;
    }
};

} // namespace PHARE::amr

#endif
