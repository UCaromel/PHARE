#ifndef PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP
#define PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

namespace PHARE::core
{




template<typename GridLayout>
class Ampere
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    Ampere(GridLayout const& layout)
        : layout_{layout}
    {
    }


    template<typename VecField>
    void operator()(VecField const& B, VecField& J)
    {
        // can't use structured bindings because
        //   "reference to local binding declared in enclosing function"
        auto& Jx = J(Component::X);
        auto& Jy = J(Component::Y);
        auto& Jz = J(Component::Z);

        // Point<std::uint32_t, dimension> shrink;
        //
        // for (size_t i = 0; i < dimension; ++i)
        // {
        //     shrink[i] = 1;
        // }
        //
        // layout_.evalOnShrinkedGhostBox(Jx, shrink,
        //                                [&](auto&... args) mutable { JxEq_(Jx, B, args...); });
        // layout_.evalOnShrinkedGhostBox(Jy, shrink,
        //                                [&](auto&... args) mutable { JyEq_(Jy, B, args...); });
        // layout_.evalOnShrinkedGhostBox(Jz, shrink,
        //                                [&](auto&... args) mutable { JzEq_(Jz, B, args...); });

        // Point<std::uint32_t, dimension> growX;
        // growX[dirX] += 1;
        // Point<std::uint32_t, dimension> growY;
        // if constexpr (dimension >= 2)
        //     growY[dirY] += 1;
        // Point<std::uint32_t, dimension> growZ;
        // if constexpr (dimension == 3)
        //     growZ[dirZ] += 1;
        //
        // layout_.evalOnBiggerBox(Jx, growX, [&](auto&... args) mutable { JxEq_(Jx, B, args...);
        // }); layout_.evalOnBiggerBox(Jy, growY, [&](auto&... args) mutable { JyEq_(Jy, B,
        // args...); }); layout_.evalOnBiggerBox(Jz, growZ, [&](auto&... args) mutable { JzEq_(Jz,
        // B, args...); });

        layout_.evalOnBox(Jx, [&](auto&... args) mutable { JxEq_(Jx, B, args...); });
        layout_.evalOnBox(Jy, [&](auto&... args) mutable { JyEq_(Jy, B, args...); });
        layout_.evalOnBox(Jz, [&](auto&... args) mutable { JzEq_(Jz, B, args...); });
    }


private:
    GridLayout layout_;


    template<typename VecField, typename Field, typename... Indexes>
    void JxEq_(Field& Jx, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [_, By, Bz] = B();

        if constexpr (dimension == 1)
            Jx(ijk...) = 0.0;

        if constexpr (dimension == 2)
            Jx(ijk...) = layout_.template deriv<Direction::Y, 4>(Bz, {ijk...});

        if constexpr (dimension == 3)
            Jx(ijk...) = layout_.template deriv<Direction::Y, 4>(Bz, {ijk...})
                         - layout_.template deriv<Direction::Z, 4>(By, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JyEq_(Field& Jy, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [Bx, By, Bz] = B();

        if constexpr (dimension == 1 || dimension == 2)
            Jy(ijk...) = -layout_.template deriv<Direction::X, 4>(Bz, {ijk...});

        if constexpr (dimension == 3)
            Jy(ijk...) = layout_.template deriv<Direction::Z, 4>(Bx, {ijk...})
                         - layout_.template deriv<Direction::X, 4>(Bz, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JzEq_(Field& Jz, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [Bx, By, Bz] = B();

        if constexpr (dimension == 1)
            Jz(ijk...) = layout_.template deriv<Direction::X, 4>(By, {ijk...});

        else
            Jz(ijk...) = layout_.template deriv<Direction::X, 4>(By, {ijk...})
                         - layout_.template deriv<Direction::Y, 4>(Bx, {ijk...});
    }
};

} // namespace PHARE::core
#endif
