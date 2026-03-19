#ifndef PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/mhd_conversion.hpp"
#include "core/utilities/index/index.hpp"
#include "initializer/data_provider.hpp"

namespace PHARE::core
{
static auto const min_value = std::sqrt(1024 * std::numeric_limits<double>::min());

auto rhoVToV(auto& rho, auto const& rhoVx, auto const& rhoVy, auto const& rhoVz)
{
    auto const vx = rhoVx / rho;
    auto const vy = rhoVy / rho;
    auto const vz = rhoVz / rho;

    return std::make_tuple(vx, vy, vz);
}

template<typename GridLayout>
class ToPrimitiveConverter_ref;

template<typename GridLayout>
class ToPrimitiveConverter : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    ToPrimitiveConverter(PHARE::initializer::PHAREDict const& dict)
        : gamma_{dict["heat_capacity_ratio"].template to<double>()}
    {
    }

    template<typename Field, typename VecField>
    void operator()(Field& rho, VecField const& rhoV, VecField const& B1, VecField const& B0,
                    Field const& Etot1, VecField& V, Field& P) const
    {
        ToPrimitiveConverter_ref<GridLayout>{*this->layout_}(gamma_, rho, rhoV, B1, B0, Etot1, V,
                                                            P);
    }

private:
    double const gamma_;
};

template<typename GridLayout>
class ToPrimitiveConverter_ref
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    ToPrimitiveConverter_ref(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename Field, typename VecField>
    void operator()(double const gamma, Field& rho, VecField const& rhoV, VecField const& B1,
                    VecField const& B0, Field const& Etot1, VecField& V, Field& P) const
    {
        rhoVToVOnGhostBox(rho, rhoV, V);

        eosEtot1ToPOnGhostBox(gamma, rho, rhoV, B1, B0, Etot1, P);
    }

    // used for diagnostics
    template<typename Field, typename VecField>
    void rhoVToVOnGhostBox(Field& rho, VecField const& rhoV, VecField& V) const
    {
        layout_.evalOnGhostBox(rho,
                               [&](auto&... args) mutable { rhoVToV_(rho, rhoV, V, {args...}); });
    }

    template<typename Field, typename VecField>
    void eosEtot1ToPOnGhostBox(double const gamma, Field const& rho, VecField const& rhoV,
                               VecField const& B1, VecField const& B0, Field const& Etot1,
                               Field& P) const
    {
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            eosEtot1ToP_(gamma, rho, rhoV, B1, B0, Etot1, P, {args...});
        });
    }

    template<typename Field, typename VecField>
    void eosEtotToPOnGhostBox(double const gamma, Field const& rho, VecField const& rhoV,
                              VecField const& B1, VecField const& B0, Field const& Etot1,
                              Field& P) const
    {
        eosEtot1ToPOnGhostBox(gamma, rho, rhoV, B1, B0, Etot1, P);
    }

private:
    template<typename Field, typename VecField>
    static void rhoVToV_(Field& rho, VecField const& rhoV, VecField& V,
                         MeshIndex<Field::dimension> index)
    {
        auto const& rhoVx = rhoV(Component::X);
        auto const& rhoVy = rhoV(Component::Y);
        auto const& rhoVz = rhoV(Component::Z);

        auto& Vx = V(Component::X);
        auto& Vy = V(Component::Y);
        auto& Vz = V(Component::Z);

        auto&& [x, y, z] = rhoVToV(rho(index), rhoVx(index), rhoVy(index), rhoVz(index));
        Vx(index)        = x;
        Vy(index)        = y;
        Vz(index)        = z;
    }

    template<typename Field, typename VecField>
    static void eosEtot1ToP_(double const gamma, Field const& rho, VecField const& rhoV,
                             VecField const& B1, VecField const&, Field const& Etot1, Field& P,
                             MeshIndex<Field::dimension> index)
    {
        auto const& rhoVx = rhoV(Component::X);
        auto const& rhoVy = rhoV(Component::Y);
        auto const& rhoVz = rhoV(Component::Z);

        auto const& B1x = B1(Component::X);
        auto const& B1y = B1(Component::Y);
        auto const& B1z = B1(Component::Z);
        auto const vx = rhoVx(index) / rho(index);
        auto const vy = rhoVy(index) / rho(index);
        auto const vz = rhoVz(index) / rho(index);
        auto const b1x = GridLayout::project(B1x, index, GridLayout::faceXToCellCenter());
        auto const b1y = GridLayout::project(B1y, index, GridLayout::faceYToCellCenter());
        auto const b1z = GridLayout::project(B1z, index, GridLayout::faceZToCellCenter());
        P(index)       = eosEtot1ToP(gamma, rho(index), vx, vy, vz, b1x, b1y, b1z, Etot1(index));
    }


private:
    GridLayout layout_;
};

} // namespace PHARE::core

#endif
