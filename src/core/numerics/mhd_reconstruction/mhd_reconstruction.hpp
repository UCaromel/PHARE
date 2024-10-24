#ifndef PHARE_CORE_NUMERICS_MHD_RECONSTRUCTION_HPP
#define PHARE_CORE_NUMERICS_MHD_RECONSTRUCTION_HPP

#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/index/index.hpp"
#include <cstddef>

namespace PHARE::core
{
template<typename GridLayout>
class Reconstruction : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<auto direction, typename Field>
    void operator()(Field const& F, Field& uL, Field& uR)
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - Reconstruction - GridLayout not set, cannot proceed "
                                     "to reconstruction");

        if (!(F.isUsable() && uL.isUsable() && uR.isUsable()))
            throw std::runtime_error(
                "Error - Reconstruction - not all Field parameters are usable");


        layout_->evalOnBox(
            uL, [&](auto&... args) { this->template reconstruct_uL_<direction>(F, uL, args...); });

        layout_->evalOnBox(
            uR, [&](auto&... args) { this->template reconstruct_uL_<direction>(F, uR, args...); });
    }

private:
    template<auto direction, typename Field, typename... Indexes>
    void reconstruct_uL_(Field const& F, Field& uL, Indexes const&... ijk) const
    {
        uL(ijk...) = this->template constant_uL_<direction>(F, {ijk...});
    }

    template<auto direction, typename Field, typename... Indexes>
    void reconstruct_uR_(Field const& F, Field& uR, Indexes const&... ijk) const
    {
        uR(ijk...) = this->template constant_uR_<direction>(F, {ijk...});
    }

    template<auto direction, typename Field>
    auto constant_uL_(Field const& F, MeshIndex<Field::dimension> index) const
    {
        auto fieldCentering = layout_->centering(F.physicalQuantity());

        std::size_t dir;
        if (direction == Direction::X)
            dir = PHARE::core::dirX;
        else if (direction == Direction::Y)
            dir = PHARE::core::dirY;
        else
            dir = PHARE::core::dirZ;

        return (layout_->prevIndex(fieldCentering[dir], index[0]));
    }

    template<auto direction, typename Field>
    auto constant_uR_(Field const& F, MeshIndex<Field::dimension> index) const
    {
        std::size_t dir;
        if (direction == Direction::X)
            dir = PHARE::core::dirX;
        else if (direction == Direction::Y)
            dir = PHARE::core::dirY;
        else
            dir = PHARE::core::dirZ;

        return F(index[0]);
    }
};

} // namespace PHARE::core

#endif
