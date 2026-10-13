#ifndef PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP
#define PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/differential_operators.hpp"

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

        auto const curlB = curl(layout_, B);

        eval(Jx, [&](auto const&... ijk) { Jx(ijk...) = curlB.x({ijk...}); });
        eval(Jy, [&](auto const&... ijk) { Jy(ijk...) = curlB.y({ijk...}); });
        eval(Jz, [&](auto const&... ijk) { Jz(ijk...) = curlB.z({ijk...}); });
    }


private:
    GridLayout layout_;
};

// AmperePV writes the point-value current on the ghost box shrunk by this many cells
inline constexpr std::uint32_t point_value_current_shrink = 2;

template<typename GridLayout>
class AmperePV
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    explicit AmperePV(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename VecField>
    void operator()(VecField const& Bavg, VecField const& Bpv, VecField& J)
    {
        auto& Jx = J(Component::X);
        auto& Jy = J(Component::Y);
        auto& Jz = J(Component::Z);

        Point<std::uint32_t, dimension> amount;
        for (std::size_t i = 0; i < dimension; ++i)
            amount[i] = point_value_current_shrink;

        auto eval = [&](auto& Jc, auto&& fn) { layout_.evalOnShrinkedGhostBox(Jc, amount, fn); };

        eval(Jx, [&](auto&... args) mutable { JxPvEq_(Jx, Bavg, Bpv, args...); });
        eval(Jy, [&](auto&... args) mutable { JyPvEq_(Jy, Bavg, Bpv, args...); });
        eval(Jz, [&](auto&... args) mutable { JzPvEq_(Jz, Bavg, Bpv, args...); });
    }

private:
    template<auto direction, typename Field>
    auto pvDeriv_(Field const& avg, Field const& pv, MeshIndex<dimension> index) const
    {
        constexpr auto d = static_cast<std::size_t>(direction);
        // as in Partial: a primal source is differentiated onto the dual node of the same index,
        // a dual one onto the primal node one index up
        int const first = layout_.centering(avg)[d] == QtyCentering::primal ? 0 : -1;
        auto const at   = [&](Field const& f, int const shift) {
            return detail::shifted<d>(f, index, first + shift);
        };
        auto const inv = layout_.inverseMeshSize(direction);

        auto const fourthOrderAvg
            = inv * (1.125 * (at(avg, 1) - at(avg, 0)) - (1. / 24.) * (at(avg, 2) - at(avg, -1)));
        return fourthOrderAvg + inv * (at(pv, 1) - at(pv, 0)) - inv * (at(avg, 1) - at(avg, 0));
    }

    template<typename VecField, typename Field, typename... Indexes>
    void JxPvEq_(Field& Jx, VecField const& Bavg, VecField const& Bpv, Indexes const&... ijk) const
    {
        auto const& [_, Bya, Bza]  = Bavg();
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
    void JyPvEq_(Field& Jy, VecField const& Bavg, VecField const& Bpv, Indexes const&... ijk) const
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
    void JzPvEq_(Field& Jz, VecField const& Bavg, VecField const& Bpv, Indexes const&... ijk) const
    {
        auto const& [Bxa, Bya, Bza] = Bavg();
        auto const& [Bxp, Byp, Bzp] = Bpv();

        if constexpr (dimension == 1)
            Jz(ijk...) = pvDeriv_<Direction::X>(Bya, Byp, {ijk...});
        else
            Jz(ijk...) = pvDeriv_<Direction::X>(Bya, Byp, {ijk...})
                         - pvDeriv_<Direction::Y>(Bxa, Bxp, {ijk...});
    }

    GridLayout layout_;
};

} // namespace PHARE::core
#endif
