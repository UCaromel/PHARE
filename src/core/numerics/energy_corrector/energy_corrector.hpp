#ifndef PHARE_CORE_NUMERICS_ENERGY_CORRECTOR_HPP
#define PHARE_CORE_NUMERICS_ENERGY_CORRECTOR_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/utilities/index/index.hpp"

namespace PHARE::core
{
template<typename GridLayout>
class EnergyCorrector_ref;

template<typename GridLayout>
class EnergyCorrector : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<typename Field, typename VecField>
    void operator()(Field& Etot, VecField const& B, VecField const& Bc) const
    {
        if (!this->hasLayout())
            throw std::runtime_error(
                "Error - Ampere - GridLayout not set, cannot proceed to calculate ampere()");

        EnergyCorrector_ref{*this->layout_}(Etot, B, Bc);
    }
};

template<typename GridLayout>
class EnergyCorrector_ref
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    EnergyCorrector_ref(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename Field, typename VecField>
    void operator()(Field& Etot, VecField const& B, VecField const& Bc) const
    {
        layout_.evalOnBox(Etot, [&](auto&... args) mutable { correct_(Etot, B, Bc, {args...}); });
    }


private:
    GridLayout layout_;


    void correct_(auto& Etot, auto const& B, auto const& Bc, MeshIndex<dimension> index) const
    {
        auto const& [Bx, By, Bz]    = B();
        auto const& [Bxc, Byc, Bzc] = Bc();

        auto const Bxf = GridLayout::project(Bx, index, GridLayout::faceXToCellCenter());
        auto const Byf = GridLayout::project(By, index, GridLayout::faceYToCellCenter());
        auto const Bzf = GridLayout::project(Bz, index, GridLayout::faceZToCellCenter());

        auto const B2c
            = Bxc(index) * Bxc(index) + Byc(index) * Byc(index) + Bzc(index) * Bzc(index);
        auto const B2f = Bxf * Bxf + Byf * Byf + Bzf * Bzf;
        Etot(index) += B2f - B2c;
    }
};

} // namespace PHARE::core
#endif
