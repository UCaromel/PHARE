#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/constants.hpp"
#include "core/utilities/index/index.hpp"

namespace PHARE::core
{
template<typename GridLayout>
class ConstrainedTransport : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<typename VecField, typename... Fluxes>
    void operator()(VecField& E, const Fluxes&... fluxes) const
    {
        auto& Ex = E(Component::X);
        auto& Ey = E(Component::Y);
        auto& Ez = E(Component::Z);

        auto const& [_, By_x, Bz_x] = std::get<0>(fluxes...);

        if constexpr (dimension == 1)
        {
            layout_->evalOnBox(Ey,
                               [&](auto&... args) mutable { this->Ey_(Ey, {args...}, {Bz_x}); });

            layout_->evalOnBox(Ez,
                               [&](auto&... args) mutable { this->Ez_(Ez, {args...}, {By_x}); });
        }
        else if constexpr (dimension >= 2)
        {
            auto const& [Bx_y, _, Bz_y] = std::get<1>(fluxes...);

            if constexpr (dimension == 2)
            {
                layout_->evalOnBox(
                    Ex, [&](auto&... args) mutable { this->Ez_(Ex, {args...}, {Bz_y}); });

                layout_->evalOnBox(
                    Ey, [&](auto&... args) mutable { this->Ey_(Ey, {args...}, {Bz_x}); });

                layout_->evalOnBox(
                    Ez, [&](auto&... args) mutable { this->Ez_(Ez, {args...}, {By_x, Bx_y}); });
            }
            else if constexpr (dimension == 3)
            {
                auto const& [Bx_z, By_z, _] = std::get<1>(fluxes...);

                layout_->evalOnBox(
                    Ex, [&](auto&... args) mutable { this->Ez_(Ex, {args...}, {Bz_y, By_z}); });

                layout_->evalOnBox(
                    Ey, [&](auto&... args) mutable { this->Ey_(Ey, {args...}, {Bz_x, Bx_z}); });

                layout_->evalOnBox(
                    Ez, [&](auto&... args) mutable { this->Ez_(Ez, {args...}, {By_x, Bx_y}); });
            }
        }
    }

private:
    template<typename Field, typename... Fluxes>
    void Ex_(Field& Ex, MeshIndex<Field::dimension> index, const Fluxes&... fluxes) const
    {
        if constexpr (dimension == 1) {}
        else if constexpr (dimension >= 2)
        {
            auto& Bz_y = std::get<0>(fluxes...);

            if constexpr (dimension == 2)
            {
                Ex(index) = -Bz_y(index);
            }
            else if constexpr (dimension == 3)
            {
                auto& By_z = std::get<1>(fluxes...);

                auto CenteringY = layout_->centering(Bz_y.physicalQuantity());
                auto CenteringZ = layout_->centering(By_z.physicalQuantity());
                Ex(index)
                    = 0.25
                      * (-Bz_y(index)
                         - Bz_y(index[0], index[1], layout_->prevIndex(CenteringY[dirZ], index[2]))
                         + By_z(index)
                         + By_z(index[0], layout_->prevIndex(CenteringZ[dirY], index[1]),
                                index[2]));
            }
        }
    }

    template<typename Field, typename... Fluxes>
    void Ey_(Field& Ey, MeshIndex<Field::dimension> index, const Fluxes&... fluxes) const
    {
        auto& Bz_x = std::get<0>(fluxes...);

        if constexpr (dimension <= 2)
        {
            Ey(index) = Bz_x(index);
        }
        else if constexpr (dimension == 3)
        {
            auto& Bx_z = std::get<1>(fluxes...);

            auto CenteringX = layout_->centering(Bz_x.physicalQuantity());
            auto CenteringZ = layout_->centering(Bx_z.physicalQuantity());
            Ey(index)
                = 0.25
                  * (Bz_x(index)
                     + Bz_x(index[0], index[1], layout_->prevIndex(CenteringX[dirZ], index[2]))
                     - Bx_z(index)
                     - Bx_z(layout_->prevIndex(CenteringZ[dirX], index[0]), index[1], index[2]));
        }
    }

    template<typename Field, typename... Fluxes>
    void Ez_(Field& Ez, MeshIndex<Field::dimension> index, const Fluxes&... fluxes) const
    {
        auto& By_x = std::get<0>(fluxes...);

        if constexpr (dimension == 1)
        {
            Ez(index) = -By_x(index);
        }
        else if constexpr (dimension >= 2)
        {
            auto& Bx_y = std::get<1>(fluxes...);

            if constexpr (dimension == 2)
            {
                auto CenteringX = layout_->centering(By_x.physicalQuantity());
                auto CenteringY = layout_->centering(Bx_y.physicalQuantity());
                Ez(index)       = 0.25
                            * (-By_x(index)
                               - By_x(index[0], layout_->prevIndex(CenteringX[dirY], index[1]))
                               + Bx_y(index)
                               + Bx_y(layout_->prevIndex(CenteringY[dirX], index[0]), index[1]));
            }
            else if constexpr (dimension == 3)
            {
                auto CenteringX = layout_->centering(By_x.physicalQuantity());
                auto CenteringY = layout_->centering(Bx_y.physicalQuantity());
                Ez(index)
                    = 0.25
                      * (-By_x(index)
                         - By_x(index[0], layout_->prevIndex(CenteringX[dirY], index[1]), index[2])
                         + Bx_y(index)
                         + Bx_y(layout_->prevIndex(CenteringY[dirX], index[0]), index[1],
                                index[2]));
            }
        }
    }
};

} // namespace PHARE::core

#endif
