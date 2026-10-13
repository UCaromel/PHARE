#ifndef PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP
#define PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <cstddef>
#include <cstdint>

namespace PHARE::core
{


enum class AmpereMode { ShrinkedGhost, GrownPhysical };

struct AmpereBox // structural type — usable as NTTP
{
    AmpereMode mode      = AmpereMode::ShrinkedGhost;
    std::uint32_t amount = 1;
};


template<typename GridLayout, AmpereBox box = AmpereBox{}>
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

        Point<std::uint32_t, dimension> amount;

        for (size_t i = 0; i < dimension; ++i)
        {
            amount[i] = box.amount;
        }

        auto eval = [&](auto& Jc, auto&& fn) {
            if constexpr (box.mode == AmpereMode::ShrinkedGhost)
                layout_.evalOnShrinkedGhostBox(Jc, amount, fn);
            else
                layout_.evalOnBiggerBox(Jc, amount, fn);
        };

        eval(Jx, [&](auto&... args) mutable { JxEq_(Jx, B, args...); });
        eval(Jy, [&](auto&... args) mutable { JyEq_(Jy, B, args...); });
        eval(Jz, [&](auto&... args) mutable { JzEq_(Jz, B, args...); });
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
            Jx(ijk...) = layout_.template deriv<Direction::Y>(Bz, {ijk...});

        if constexpr (dimension == 3)
            Jx(ijk...) = layout_.template deriv<Direction::Y>(Bz, {ijk...})
                         - layout_.template deriv<Direction::Z>(By, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JyEq_(Field& Jy, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [Bx, By, Bz] = B();

        if constexpr (dimension == 1 || dimension == 2)
            Jy(ijk...) = -layout_.template deriv<Direction::X>(Bz, {ijk...});

        if constexpr (dimension == 3)
            Jy(ijk...) = layout_.template deriv<Direction::Z>(Bx, {ijk...})
                         - layout_.template deriv<Direction::X>(Bz, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JzEq_(Field& Jz, VecField const& B, Indexes const&... ijk) const
    {
        auto const& [Bx, By, Bz] = B();

        if constexpr (dimension == 1)
            Jz(ijk...) = layout_.template deriv<Direction::X>(By, {ijk...});

        else
            Jz(ijk...) = layout_.template deriv<Direction::X>(By, {ijk...})
                         - layout_.template deriv<Direction::Y>(Bx, {ijk...});
    }
};

template<typename GridLayout, AmpereBox box = AmpereBox{}>
class AmperePV
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    explicit AmperePV(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename VecField>
    void operator()(VecField const& B, VecField& J)
    {
        auto& Jx = J(Component::X);
        auto& Jy = J(Component::Y);
        auto& Jz = J(Component::Z);

        Point<std::uint32_t, dimension> amount;
        for (std::size_t i = 0; i < dimension; ++i)
            amount[i] = box.amount;

        auto eval = [&](auto& Jc, auto&& fn) {
            if constexpr (box.mode == AmpereMode::ShrinkedGhost)
                layout_.evalOnShrinkedGhostBox(Jc, amount, fn);
            else
                layout_.evalOnBiggerBox(Jc, amount, fn);
        };

        eval(Jx, [&](auto&... args) mutable { JxEq_(Jx, B, args...); });
        eval(Jy, [&](auto&... args) mutable { JyEq_(Jy, B, args...); });
        eval(Jz, [&](auto&... args) mutable { JzEq_(Jz, B, args...); });
    }

    template<typename VecField>
    void operator()(VecField const& Bavg, VecField const& Bpv, VecField& J)
    {
        auto& Jx = J(Component::X);
        auto& Jy = J(Component::Y);
        auto& Jz = J(Component::Z);

        Point<std::uint32_t, dimension> amount;
        for (std::size_t i = 0; i < dimension; ++i)
            amount[i] = box.amount;

        auto eval = [&](auto& Jc, auto&& fn) {
            if constexpr (box.mode == AmpereMode::ShrinkedGhost)
                layout_.evalOnShrinkedGhostBox(Jc, amount, fn);
            else
                layout_.evalOnBiggerBox(Jc, amount, fn);
        };

        eval(Jx, [&](auto&... args) mutable { JxPvEq_(Jx, Bavg, Bpv, args...); });
        eval(Jy, [&](auto&... args) mutable { JyPvEq_(Jy, Bavg, Bpv, args...); });
        eval(Jz, [&](auto&... args) mutable { JzPvEq_(Jz, Bavg, Bpv, args...); });
    }

private:
    template<auto direction, typename Field>
    auto pvDeriv_(Field const& avg, Field const& pv, MeshIndex<dimension> index) const
    {
        return layout_.template deriv<direction, 4>(avg, index)
               + layout_.template deriv<direction, 2>(pv, index)
               - layout_.template deriv<direction, 2>(avg, index);
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JxPvEq_(Field& Jx, VecField const& Bavg, VecField const& Bpv,
                 Indexes const&... ijk) const
    {
        auto const& [_, Bya, Bza]   = Bavg();
        auto const& [_p, Byp, Bzp] = Bpv();

        if constexpr (dimension == 1)
            Jx(ijk...) = 0.0;
        if constexpr (dimension == 2)
            Jx(ijk...) = pvDeriv_<Direction::Y>(Bza, Bzp, {ijk...});
        if constexpr (dimension == 3)
            Jx(ijk...) = pvDeriv_<Direction::Y>(Bza, Bzp, {ijk...})
                         - pvDeriv_<Direction::Z>(Bya, Byp, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JyPvEq_(Field& Jy, VecField const& Bavg, VecField const& Bpv,
                 Indexes const&... ijk) const
    {
        auto const& [Bxa, Bya, Bza] = Bavg();
        auto const& [Bxp, Byp, Bzp] = Bpv();

        if constexpr (dimension == 1 || dimension == 2)
            Jy(ijk...) = -pvDeriv_<Direction::X>(Bza, Bzp, {ijk...});
        if constexpr (dimension == 3)
            Jy(ijk...) = pvDeriv_<Direction::Z>(Bxa, Bxp, {ijk...})
                         - pvDeriv_<Direction::X>(Bza, Bzp, {ijk...});
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JzPvEq_(Field& Jz, VecField const& Bavg, VecField const& Bpv,
                 Indexes const&... ijk) const
    {
        auto const& [Bxa, Bya, Bza] = Bavg();
        auto const& [Bxp, Byp, Bzp] = Bpv();

        if constexpr (dimension == 1)
            Jz(ijk...) = pvDeriv_<Direction::X>(Bya, Byp, {ijk...});
        else
            Jz(ijk...) = pvDeriv_<Direction::X>(Bya, Byp, {ijk...})
                         - pvDeriv_<Direction::Y>(Bxa, Bxp, {ijk...});
    }

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

    GridLayout layout_;
};

} // namespace PHARE::core
#endif
