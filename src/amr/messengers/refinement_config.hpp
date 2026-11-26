#ifndef PHARE_REFINEMENT_CONFIG_HPP
#define PHARE_REFINEMENT_CONFIG_HPP

#include "initializer/data_provider.hpp"
#include "amr/data/field/refine/field_refiner_kernel.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace PHARE::amr
{

//! Deck names of the slope rules; throws on an unknown name.
inline FieldRefinementLimiter limiterFromName(std::string const& name)
{
    if (name == "none")
        return FieldRefinementLimiter::None;
    if (name == "vanleer")
        return FieldRefinementLimiter::VanLeer;
    if (name == "mc")
        return FieldRefinementLimiter::MC;
    if (name == "mc_gate")
        return FieldRefinementLimiter::MCGate;
    if (name == "ao5z")
        return FieldRefinementLimiter::AO5Z;
    if (name == "mp")
        return FieldRefinementLimiter::MP;
    throw std::runtime_error("unknown refinement limiter: " + name
                             + " (none, vanleer, mc, mc_gate, ao5z, mp)");
}

inline RefinementTensorMode tensorModeFromName(std::string const& name)
{
    if (name == "product")
        return RefinementTensorMode::Product;
    if (name == "sweeps")
        return RefinementTensorMode::Sweeps;
    throw std::runtime_error("unknown refinement tensor mode: " + name + " (product, sweeps)");
}

inline RefinementSweepOrders sweepOrdersFromName(std::string const& name)
{
    if (name == "all")
        return RefinementSweepOrders::All;
    if (name == "cyclic")
        return RefinementSweepOrders::Cyclic;
    if (name == "single")
        return RefinementSweepOrders::Single;
    throw std::runtime_error("unknown refinement sweep orders: " + name + " (all, cyclic, single)");
}


/**
 * @brief Runtime selection of the field-refinement order and limiting.
 *
 * The refine-op members of the messengers are built from the runtime kernels
 * (makeRefineKernel / makeMagneticRefineKernel), whose stencil is selected by this order.
 * FieldRefinementOrder::Linear (order 2) is the only supported value for now; this struct is the
 * extension point for higher orders. Distinct from the EXISTING particle split-operator template
 * param named RefinementParams (MessengerFactory / HybridHybridMessengerStrategy) — do not
 * conflate.
 */
struct RefinementConfig
{
    FieldRefinementOrder order = FieldRefinementOrder::Linear;
    LimiterSettings fill{};      // level creation and regrid
    LimiterSettings ghostFill{}; // ghost fills; equals `fill` unless overridden by refinement/ghost
    bool probe = false;

    //! Read the optional field-refinement selection from the dict. Absent "AMR"/"refinement"/
    //! "order" nodes at any level of the path ⇒ the RefinementConfig default order. Only order 2
    //! (Linear) is supported; this is the one place a raw dict value is validated against it.
    //! Every limiter key is optional; absent keys give master's table. ghost/* keys override the
    //! fill settings for ghost fills only.
    RefinementConfig static FROM(PHARE::initializer::PHAREDict const& dict)
    {
        PHARE::amr::RefinementConfig config;
        auto const rawOrder = cppdict::get_value(dict, "simulation/AMR/refinement/order", int{2});
        if (rawOrder != 2)
            throw std::runtime_error("unsupported field refinement order: "
                                     + std::to_string(rawOrder) + " (only 2 is supported)");
        config.order = static_cast<FieldRefinementOrder>(rawOrder);

        std::string const root = "simulation/AMR/refinement/";
        auto& fill             = config.fill;
        if (auto const name = optional_<std::string>(dict, root + "limiter"))
            fill.limiter = limiterFromName(*name);
        fill.param = optional_<double>(dict, root + "limiter_param");
        if (auto const name = optional_<std::string>(dict, root + "tensor"))
            fill.tensor = tensorModeFromName(*name);
        if (auto const name = optional_<std::string>(dict, root + "sweep_orders"))
            fill.sweepOrders = sweepOrdersFromName(*name);
        fill.energyRow = cppdict::get_value(dict, root + "energy_row", false);

        auto& ghost = config.ghostFill;
        ghost       = fill;
        if (auto const name = optional_<std::string>(dict, root + "ghost/limiter"))
        {
            ghost.limiter = limiterFromName(*name);
            if (ghost.limiter != fill.limiter) // fill's parameter belongs to fill's rule
                ghost.param = std::nullopt;
        }
        if (auto const param = optional_<double>(dict, root + "ghost/limiter_param"))
            ghost.param = param;
        ghost.energyRow = cppdict::get_value(dict, root + "ghost/energy_row", fill.energyRow);

        config.probe = cppdict::get_value(dict, root + "probe", false);
        return config;
    }

private:
    template<typename T>
    static std::optional<T> optional_(PHARE::initializer::PHAREDict const& dict,
                                      std::string const& path)
    {
        if (auto leaf = cppdict::get(cppdict::detail::split_string(path), 0, dict))
            return leaf->template to<T>();
        return std::nullopt;
    }
};

} // namespace PHARE::amr

#endif
