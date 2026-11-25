#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_HALL_FACE_PAIR_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_HALL_FACE_PAIR_HPP

#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/meta/meta_utilities.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace PHARE::core
{
// Runtime switches, read once per process.
//   PHARE_MHD_HALL_F12=1 : fourth-order face-pair Hall field (below) instead of the Hall terms in
//                          the Riemann solver and the UCT EMF.
//   PHARE_MHD_HALL_FAN=0 : no upwind whistler fan (default on for Hall runs with nu = 0).
inline bool envFlag_(char const* name, bool const fallback)
{
    char const* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::string_view{v} != "0";
}

inline bool hallFacePairF12()
{
    static bool const on = envFlag_("PHARE_MHD_HALL_F12", false);
    return on;
}

inline bool hallWhistlerFan()
{
    static bool const on = envFlag_("PHARE_MHD_HALL_FAN", true);
    return on;
}


/**
 * Fourth-order face-pair Hall field "F12", point value on the Yee edges.
 *
 *   M B̄_b   = (1 - Σ_{d⊥b} δ²_d / 12) B̄_b                         (face averages, one filter)
 *   J̃       = curl_Yee(M B̄)                                        (edges)
 *   J^s_a   = (1 + δ²_a / 24) J̃_a                                   (sharpened along its edge)
 *   W(a,b)  = I4_c[J^s_a] · B^pv_b / I4_b[ρ^pv]                     (on the face holding B_b)
 *   E^pv_c  = I4_a[W(a,b)] - I4_b[W(b,a)]                          ((c, a, b) cyclic)
 *
 * The point-value pipeline then averages E^pv_c along its edge with (1 + δ²_c / 24), so the
 * edge-average Hall field is the transpose of J^s's construction applied to the products and
 * Σ_edges Ē_H · J̃ = 0: the Hall field does no work on ½ B̄ᵀ M B̄.
 *
 * Reach (ghost layers, G = 6): M B̄ on the ghost box shrunk by 1, J̃ by 1 along its own axis and
 * 2 across, J^s by 2, E^pv by 4. E^pv is then valid 2 nodes beyond the physical box across its
 * edge and 1.5 cells along it, which is what CT (own axis + 1) and the point-value Etot flux
 * (faces grown 1, 4-point edge -> face) read.
 */
template<typename GridLayout>
class HallFacePairO4
{
    constexpr static auto dimension = GridLayout::dimension;
    using implT                     = GridLayout::implT;
    using InterpDir                 = implT::InterpDir;

public:
    static constexpr std::uint32_t filter_shrink  = 1;
    static constexpr std::uint32_t current_shrink = 2; // across the edge; 1 along it
    static constexpr std::uint32_t sharp_shrink   = 2;
    static constexpr std::uint32_t field_shrink   = 4;

    explicit HallFacePairO4(GridLayout const& layout)
        : layout_{layout}
    {
    }

    // Bavg: face averages; Bpv, rhopv: point values; MB, Jt, Js: scratch; E: written (point value)
    void operator()(auto& MB, auto& Jt, auto& Js, auto const& Bavg, auto const& Bpv,
                    auto const& rhopv, auto& E) const
    {
        filter_(MB, Bavg);
        current_(Jt, MB);
        sharpen_(Js, Jt);

        auto const amount = uniform_(field_shrink);
        layout_.evalOnShrinkedGhostBox(E(Component::X), amount, [&](auto const&... args) {
            E(Component::X)(args...) = edge_<0, 1, 2>(Js, Bpv, rhopv, {args...});
        });
        layout_.evalOnShrinkedGhostBox(E(Component::Y), amount, [&](auto const&... args) {
            E(Component::Y)(args...) = edge_<1, 2, 0>(Js, Bpv, rhopv, {args...});
        });
        layout_.evalOnShrinkedGhostBox(E(Component::Z), amount, [&](auto const&... args) {
            E(Component::Z)(args...) = edge_<2, 0, 1>(Js, Bpv, rhopv, {args...});
        });
    }

    // J̃ = curl_Yee(M B̄) alone, on the same box as in operator(); exposed for the zero-work check
    void current(auto& MB, auto& Jt, auto const& Bavg) const
    {
        filter_(MB, Bavg);
        current_(Jt, MB);
    }

private:
    static Point<std::uint32_t, dimension> uniform_(std::uint32_t const n)
    {
        Point<std::uint32_t, dimension> p;
        for (std::size_t i = 0; i < dimension; ++i)
            p[i] = n;
        return p;
    }

    template<std::size_t dir, InterpDir interp>
    static double interp4_(auto&& f, MeshIndex<dimension> const& idx)
    {
        constexpr auto wps = implT::template directionalInterp<dir, interp, 4>();
        double r           = 0.;
        for (auto const& wp : wps)
            r += wp.coef * f(idx + wp.indexes);
        return r;
    }

    void filter_(auto& MB, auto const& Bavg) const
    {
        auto const amount = uniform_(filter_shrink);
        for_N<3>([&](auto i) {
            constexpr auto b = static_cast<Direction>(i());
            auto const& Bb   = Bavg(static_cast<Component>(i()));
            auto& MBb        = MB(static_cast<Component>(i()));
            layout_.evalOnShrinkedGhostBox(MBb, amount, [&](auto const&... args) {
                MeshIndex<dimension> idx{args...};
                MBb(idx) = Bb(idx) - layout_.template transverseLapl<b>(Bb, idx) / 12.;
            });
        });
    }

    void current_(auto& Jt, auto const& MB) const
    {
        auto const& MBx = MB(Component::X);
        auto const& MBy = MB(Component::Y);
        auto const& MBz = MB(Component::Z);

        for_N<3>([&](auto i) {
            constexpr std::size_t a = i();
            Point<std::uint32_t, dimension> amount;
            for (std::size_t d = 0; d < dimension; ++d)
                amount[d] = (d == a) ? 1u : current_shrink;

            auto& Ja = Jt(static_cast<Component>(a));
            layout_.evalOnShrinkedGhostBox(Ja, amount, [&](auto const&... args) {
                MeshIndex<dimension> idx{args...};
                double j = 0.;
                if constexpr (a == 0)
                {
                    if constexpr (dimension >= 2)
                        j += layout_.template deriv<Direction::Y>(MBz, idx);
                    if constexpr (dimension == 3)
                        j -= layout_.template deriv<Direction::Z>(MBy, idx);
                }
                else if constexpr (a == 1)
                {
                    if constexpr (dimension == 3)
                        j += layout_.template deriv<Direction::Z>(MBx, idx);
                    j -= layout_.template deriv<Direction::X>(MBz, idx);
                }
                else
                {
                    j += layout_.template deriv<Direction::X>(MBy, idx);
                    if constexpr (dimension >= 2)
                        j -= layout_.template deriv<Direction::Y>(MBx, idx);
                }
                Ja(idx) = j;
            });
        });
    }

    void sharpen_(auto& Js, auto const& Jt) const
    {
        auto const amount = uniform_(sharp_shrink);
        for_N<3>([&](auto i) {
            constexpr std::size_t a = i();
            auto const& Ja          = Jt(static_cast<Component>(a));
            auto& Jsa               = Js(static_cast<Component>(a));
            layout_.evalOnShrinkedGhostBox(Jsa, amount, [&](auto const&... args) {
                MeshIndex<dimension> idx{args...};
                if constexpr (a < dimension)
                    Jsa(idx)
                        = Ja(idx)
                          + layout_.template directionalLapl<static_cast<Direction>(a)>(Ja, idx)
                                / 24.;
                else
                    Jsa(idx) = Ja(idx);
            });
        });
    }

    // W(a,b) on the face holding B_b: J^s_a moved along c, times B_b over ρ moved along b
    template<std::size_t c, std::size_t a, std::size_t b>
    static double product_(auto const& Js, auto const& Bpv, auto const& rhopv,
                           MeshIndex<dimension> const& face)
    {
        auto const& Jsa = Js(static_cast<Component>(a));
        auto const& Bb  = Bpv(static_cast<Component>(b));
        auto const j    = interp4_<c, InterpDir::PrimalToDual>(Jsa, face);
        auto const rho  = interp4_<b, InterpDir::DualToPrimal>(rhopv, face);
        return j * Bb(face) / rho;
    }

    // E^pv_c = I4_a[W(a,b)] - I4_b[W(b,a)], (c, a, b) cyclic
    template<std::size_t c, std::size_t a, std::size_t b>
    static double edge_(auto const& Js, auto const& Bpv, auto const& rhopv,
                        MeshIndex<dimension> const& edge)
    {
        auto const Wab = [&](auto const& face) {
            return product_<c, a, b>(Js, Bpv, rhopv, MeshIndex<dimension>{face});
        };
        auto const Wba = [&](auto const& face) {
            return product_<c, b, a>(Js, Bpv, rhopv, MeshIndex<dimension>{face});
        };
        return interp4_<a, InterpDir::DualToPrimal>(Wab, edge)
               - interp4_<b, InterpDir::DualToPrimal>(Wba, edge);
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
