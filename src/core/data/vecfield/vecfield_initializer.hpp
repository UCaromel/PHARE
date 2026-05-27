#ifndef VECFIELD_INITIALIZER_HPP
#define VECFIELD_INITIALIZER_HPP

#include <array>
#include <memory>

#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/span.hpp"
#include "initializer/data_provider.hpp"

namespace PHARE
{
namespace core
{
    namespace
    {
        template<std::size_t dim>
        initializer::InitFunction<dim> zero_init_fn()
        {
            if constexpr (dim == 1)
                return [](auto const& x) -> std::shared_ptr<Span<double>> {
                    return std::make_shared<VectorSpan<double>>(x.size(), 0.0);
                };
            else if constexpr (dim == 2)
                return [](auto const& x, auto const&) -> std::shared_ptr<Span<double>> {
                    return std::make_shared<VectorSpan<double>>(x.size(), 0.0);
                };
            else
                return [](auto const& x, auto const&, auto const&) -> std::shared_ptr<Span<double>> {
                    return std::make_shared<VectorSpan<double>>(x.size(), 0.0);
                };
        }
    } // namespace

    template<std::size_t dimension>
    class VecFieldInitializer
    {
    public:
        VecFieldInitializer()
            : x_{zero_init_fn<dimension>()}
            , y_{zero_init_fn<dimension>()}
            , z_{zero_init_fn<dimension>()}
        {
        }

        VecFieldInitializer(initializer::PHAREDict const& dict)
            : x_{dict["x_component"].template to<initializer::InitFunction<dimension>>()}
            , y_{dict["y_component"].template to<initializer::InitFunction<dimension>>()}
            , z_{dict["z_component"].template to<initializer::InitFunction<dimension>>()}
        {
        }

        template<typename VecField, typename GridLayout>
        void initialize(VecField& v, GridLayout const& layout)
        {
            static_assert(GridLayout::dimension == VecField::dimension,
                          "dimension mismatch between vecfield and gridlayout");

            FieldUserFunctionInitializer::initialize(v.getComponent(Component::X), layout, x_);
            FieldUserFunctionInitializer::initialize(v.getComponent(Component::Y), layout, y_);
            FieldUserFunctionInitializer::initialize(v.getComponent(Component::Z), layout, z_);
        }

    private:
        initializer::InitFunction<dimension> x_;
        initializer::InitFunction<dimension> y_;
        initializer::InitFunction<dimension> z_;
    };

} // namespace core

} // namespace PHARE

#endif // VECFIELD_INITIALIZER_HPP
