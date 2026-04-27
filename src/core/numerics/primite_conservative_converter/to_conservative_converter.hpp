#ifndef PHARE_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_CONVERTER_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "core/utilities/index/index.hpp"
#include "initializer/data_provider.hpp"

namespace PHARE::core
{
inline auto vToRhoV(auto const& rho, auto const& Vx, auto const& Vy, auto const& Vz)
{
    auto const rhoVx = rho * Vx;
    auto const rhoVy = rho * Vy;
    auto const rhoVz = rho * Vz;

    return std::make_tuple(rhoVx, rhoVy, rhoVz);
}

template<typename GridLayout>
class ToConservativeConverter_ref;

template<typename GridLayout>
class ToConservativeConverter : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    ToConservativeConverter(PHARE::initializer::PHAREDict const& dict)
        : gamma_{dict["heat_capacity_ratio"].template to<double>()}
    {
    }

    template<typename Field, typename VecField>
    void operator()(Field const& rho, VecField const& V, VecField const& B1, VecField const& B0,
                    Field const& P, VecField& rhoV, Field& Etot1) const
    {
        ToConservativeConverter_ref<GridLayout>{*this->layout_, gamma_}(rho, V, B1, B0, P, rhoV,
                                                                        Etot1);
    }

private:
    double const gamma_;
};

template<typename GridLayout>
class ToConservativeConverter_ref
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    ToConservativeConverter_ref(GridLayout const& layout, double const gamma)
        : layout_{layout}
        , gamma_{gamma}
    {
    }

    template<typename Field, typename VecField>
    void operator()(Field const& rho, VecField const& V, VecField const& B1, VecField const& B0,
                    Field const& P, VecField& rhoV, Field& Etot1) const
    {
        layout_.evalOnGhostBox(rho,
                               [&](auto&... args) mutable { vToRhoV_(rho, V, rhoV, {args...}); });

        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            eosPToEtot1_(gamma_, rho, V, B1, B0, P, Etot1, {args...});
        });
    }

private:
    template<typename Field, typename VecField>
    static void vToRhoV_(Field const& rho, VecField const& V, VecField& rhoV,
                         MeshIndex<Field::dimension> index)
    {
        auto const& Vx = V(Component::X);
        auto const& Vy = V(Component::Y);
        auto const& Vz = V(Component::Z);

        auto& rhoVx = rhoV(Component::X);
        auto& rhoVy = rhoV(Component::Y);
        auto& rhoVz = rhoV(Component::Z);

        auto&& [x, y, z] = vToRhoV(rho(index), Vx(index), Vy(index), Vz(index));
        rhoVx(index)     = x;
        rhoVy(index)     = y;
        rhoVz(index)     = z;
    }

    template<typename Field, typename VecField>
    static void eosPToEtot1_(double const gamma, Field const& rho, VecField const& V,
                             VecField const& B1, VecField const&, Field const& P, Field& Etot1,
                             MeshIndex<Field::dimension> index)
    {
        auto const& Vx = V(Component::X);
        auto const& Vy = V(Component::Y);
        auto const& Vz = V(Component::Z);

        auto const& B1x = B1(Component::X);
        auto const& B1y = B1(Component::Y);
        auto const& B1z = B1(Component::Z);
        auto const b1x  = GridLayout::template project<GridLayout::faceXToCellCenter>(B1x, index);
        auto const b1y  = GridLayout::template project<GridLayout::faceYToCellCenter>(B1y, index);
        auto const b1z  = GridLayout::template project<GridLayout::faceZToCellCenter>(B1z, index);
        Etot1(index)
            = eosPToEtot1(gamma, rho(index), Vx(index), Vy(index), Vz(index), b1x, b1y, b1z,
                          P(index));
    }

private:
    GridLayout layout_;

    double const gamma_;
};

} // namespace PHARE::core

#endif
