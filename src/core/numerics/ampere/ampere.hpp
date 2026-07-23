#ifndef PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP
#define PHARE_CORE_NUMERICS_AMPERE_AMPERE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"

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



// Point-value (high-order) current operator — NOT the plain Ampere numerics operator above.
// The 2-arg form is a 4th-order curl; the 3-arg form is the split-order point-value current
// used by the MHD point-value world. Hybrid keeps consuming the 2nd-order Ampere.
template<typename GridLayout, AmpereBox box = AmpereBox{}>
class AmperePV
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    AmperePV(GridLayout const& layout)
        : layout_{layout}
    {
    }


    // 4th-order curl: J = curl_4(B)
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


    // Point-value current from the average AND point-value magnetic fields:
    //
    //     J_pv = curl_4(B_avg) + curl_2(B_pv - B_avg)
    //
    // Ghost arithmetic: B_pv is a LOCAL derived quantity (see PointValueHandler), valid only on
    // the ghost box shrunk by 1 in the directions transverse to each component. The exact route
    // curl_4(B_pv) reads +-2, so with g ghosts it can only produce J on g-3 layers — one short of
    // the g-2 (ShrinkedGhost,2) J box the Hall/resistive reconstruction consumes. Splitting
    //     curl_4(B_pv) = curl_4(B_avg) + curl_4(B_pv - B_avg)
    // and demoting ONLY the correction term to the 2nd-order curl (+-1) fixes the box: the main
    // term reads B_avg, which has full ghosts from the average-world schedules (-> g-2), and the
    // correction reads B_pv at +-1 (-> g-2 as well). Accuracy is preserved: the correction
    // B_pv - B_avg = -(1/24) lapl_T(B_avg) is O(h^2), so the demotion error
    // (curl_4 - curl_2)(O(h^2)) = O(h^2) * O(h^2) = O(h^4) — J_pv stays 4th-order.
    //
    // Alternative considered (NOT chosen, kept for the record): keep interlevel point-value
    // schedules — fill B_pv ghosts, plain curl_4(B_pv), then fill J_pv ghosts. Cost: extra refine
    // schedules plus old-state buffers for time interpolation of B_pv/J_pv coarse-fine ghosts.
    // Revisit if the local-computation route ever becomes limiting.
    template<typename VecField>
    void operator()(VecField const& Bavg, VecField const& Bpv, VecField& J)
    {
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

        eval(Jx, [&](auto&... args) mutable { JxPvEq_(Jx, Bavg, Bpv, args...); });
        eval(Jy, [&](auto&... args) mutable { JyPvEq_(Jy, Bavg, Bpv, args...); });
        eval(Jz, [&](auto&... args) mutable { JzPvEq_(Jz, Bavg, Bpv, args...); });
    }


private:
    GridLayout layout_;


    // split-order point-value derivative: 4th order on the average field, 2nd order on the
    // O(h^2) point-value correction (see the point-value operator() doc above).
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
};

} // namespace PHARE::core
#endif
