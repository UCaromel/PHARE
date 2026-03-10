#ifndef PHARE_CORE_NUMERICS_PROJECTOR_HPP
#define PHARE_CORE_NUMERICS_PROJECTOR_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/utilities/index/index.hpp"

namespace PHARE::core
{
template<typename GridLayout>
class Projector_ref;

// we might be able to make this more general by writting a function that can infer the right
// projection from the centering of 2 fields.
template<typename GridLayout>
class Projector : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<typename VecField>
    void operator()(VecField const& B, VecField& Bc) const
    {
        if (!this->hasLayout())
            throw std::runtime_error(
                "Error - Ampere - GridLayout not set, cannot proceed to calculate ampere()");

        Projector_ref{*this->layout_}(B, Bc);
    }
};

template<typename GridLayout>
class Projector_ref
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    Projector_ref(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename VecField>
    void operator()(VecField const& B, VecField& Bc) const
    {
        auto const [Bx, By, Bz]    = B();
        auto const [Bxc, Byc, Bzc] = Bc();

        layout_.evalOnGhostBox(Bxc, [&](auto&... args) mutable { project_x_(Bx, Bxc, {args...}); });
        layout_.evalOnGhostBox(Byc, [&](auto&... args) mutable { project_y_(By, Byc, {args...}); });
        layout_.evalOnGhostBox(Bzc, [&](auto&... args) mutable { project_z_(Bz, Bzc, {args...}); });
    }


private:
    GridLayout layout_;


    void project_x_(auto const& Bx, auto& Bxc, MeshIndex<dimension> index) const
    {
        Bxc(index) = GridLayout::project(Bx, index, GridLayout::faceXToCellCenter());
    }

    void project_y_(auto const& By, auto& Byc, MeshIndex<dimension> index) const
    {
        Byc(index) = GridLayout::project(By, index, GridLayout::faceYToCellCenter());
    }

    void project_z_(auto const& Bz, auto& Bzc, MeshIndex<dimension> index) const
    {
        Bzc(index) = GridLayout::project(Bz, index, GridLayout::faceZToCellCenter());
    }
};

} // namespace PHARE::core
#endif
