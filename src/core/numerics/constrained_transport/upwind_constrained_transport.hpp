#ifndef PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP
#define PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP


#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/numerics/constrained_transport/upwind_edge_b.hpp"
#include "core/numerics/constrained_transport/edge_hall.hpp"
#include "core/utilities/index/index.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace PHARE::core
{

// experiment (2D, PHARE_HALL_ZERO_WORK / PHARE_HALL_FLUX): stencils of the zero-work Hall EMF and of
// its compact energy flux F_H (FH/derivation.md S6). C = face->cell average used by to_primitive,
// J~ = curl(C^T C B), Jn = node collocation of J~, W = Jn x Bn / rho_n at nodes, E_H = A^T W.
// Then sum_c B_cc . C(curl E_H) = 0, and Q_c = -B_cc . C(curl E_H)_c = -div F_H cell by cell.
// Array indices: Bx(i,j)@(i,j+1/2) By@(i+1/2,j) Bz,rho@(i+1/2,j+1/2) Ex@(i+1/2,j) Ey@(i,j+1/2)
// Ez@(i,j); E_H reaches +-2 around the edge, F_H 1 ghost layer; MHD ghost width is 4.
template<typename Field, typename RhoField>
struct ZeroWorkHall2D
{
    using I = std::size_t;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy;

    double bx(I i, I j) const { return 0.5 * (Bx(i, j) + Bx(i + 1, j)); }
    double by(I i, I j) const { return 0.5 * (By(i, j) + By(i, j + 1)); }
    double Btx(I i, I j) const { return 0.5 * (bx(i - 1, j) + bx(i, j)); }
    double Bty(I i, I j) const { return 0.5 * (by(i, j - 1) + by(i, j)); }
    double Jtx(I i, I j) const { return (Bz(i, j) - Bz(i, j - 1)) * iy; }
    double Jty(I i, I j) const { return -(Bz(i, j) - Bz(i - 1, j)) * ix; }
    double Jtz(I i, I j) const
    {
        return (Bty(i, j) - Bty(i - 1, j)) * ix - (Btx(i, j) - Btx(i, j - 1)) * iy;
    }
    std::array<double, 3> W(I i, I j) const
    {
        double const jx  = 0.5 * (Jtx(i - 1, j) + Jtx(i, j));
        double const jy  = 0.5 * (Jty(i, j - 1) + Jty(i, j));
        double const jz  = Jtz(i, j);
        double const bnx = 0.5 * (Bx(i, j - 1) + Bx(i, j));
        double const bny = 0.5 * (By(i - 1, j) + By(i, j));
        double const bnz = 0.25 * (Bz(i - 1, j - 1) + Bz(i, j - 1) + Bz(i - 1, j) + Bz(i, j));
        double const ir  = 4.0 / (rho(i - 1, j - 1) + rho(i, j - 1) + rho(i - 1, j) + rho(i, j));
        return {(jy * bnz - jz * bny) * ir, (jz * bnx - jx * bnz) * ir,
                (jx * bny - jy * bnx) * ir};
    }
    // F_H on the x-face (i, j+1/2) and the y-face (i+1/2, j): one-sided Poynting E_H x B~ (S),
    // C-commutator (G) and A-commutator (H) terms
    double Fx(I i, I j) const
    {
        double const dx = 1.0 / ix;
        auto const w0 = W(i, j), w1 = W(i, j + 1);
        return 0.5 * (w0[1] + w1[1]) * Bz(i - 1, j) - w0[2] * Bty(i - 1, j)
               + 0.5 * dx * iy * bx(i - 1, j) * (w1[2] - w0[2]) + 0.5 * dx * w0[0] * Jtx(i - 1, j);
    }
    double Fy(I i, I j) const
    {
        double const dy = 1.0 / iy;
        auto const w0 = W(i, j), w1 = W(i + 1, j);
        return w0[2] * Btx(i, j - 1) - 0.5 * (w0[0] + w1[0]) * Bz(i, j - 1)
               - 0.5 * dy * ix * by(i, j - 1) * (w1[2] - w0[2]) + 0.5 * dy * w0[1] * Jty(i, j - 1);
    }
};

auto make_zero_work_hall_2d(auto const& B, auto const& rho, auto const& layout)
{
    using Field_t = std::decay_t<decltype(B(Component::X))>;
    using Rho_t   = std::decay_t<decltype(rho)>;
    return ZeroWorkHall2D<Field_t, Rho_t>{B(Component::X),
                                          B(Component::Y),
                                          B(Component::Z),
                                          rho,
                                          layout.inverseMeshSize(Direction::X),
                                          layout.inverseMeshSize(Direction::Y)};
}

// 1D (d/dy = d/dz = 0): Bx is constant (div B = 0) and C^T C leaves By, Bz unchanged, so J~ = J,
// Jx = 0 and the node collocation A is the identity for Ey, Ez (primal = nodes). E_H = J x Bn / rho_n
// at nodes has W.J = 0 pointwise, and F_H reduces to the one-sided Poynting term (G = H = 0).
// Bx(i)@node i, By, Bz, rho@cell i+1/2, Ey, Ez@node i.
template<typename Field, typename RhoField>
struct ZeroWorkHall1D
{
    using I = std::size_t;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix;

    std::array<double, 3> W(I i) const
    {
        double const jy  = -(Bz(i) - Bz(i - 1)) * ix;
        double const jz  = (By(i) - By(i - 1)) * ix;
        double const bnx = Bx(i);
        double const bny = 0.5 * (By(i - 1) + By(i));
        double const bnz = 0.5 * (Bz(i - 1) + Bz(i));
        double const ir  = 2.0 / (rho(i - 1) + rho(i));
        return {(jy * bnz - jz * bny) * ir, jz * bnx * ir, -jy * bnx * ir};
    }
    double Ey(I i) const { return W(i)[1]; }
    double Ez(I i) const { return W(i)[2]; }
    double Fx(I i) const { return Ey(i) * Bz(i - 1) - Ez(i) * By(i - 1); }
};

// 3D (FH/derivation.md S3): A averages each edge component along its own axis to the nodes, node
// Bn = 4-face mean, rho_n = 8-cell mean. W is cached at nodes (each face flux reads it ~9 times).
// Bx(i,j,k)@(i,j+1/2,k+1/2) By@(i+1/2,j,k+1/2) Bz@(i+1/2,j+1/2,k) Ex@(i+1/2,j,k) Ey@(i,j+1/2,k)
// Ez@(i,j,k+1/2); F_d(idx) = E_a B~_b(idx-e_d) - E_b B~_a(idx-e_d) + h_d/2 b_d(idx-e_d)(curl E)_d
// + h_d/2 W_d J~_d(idx-e_d), cyclic (d,a,b).
template<typename Field, typename RhoField>
struct ZeroWorkHall3D
{
    using I = std::size_t;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy, iz;
    I nx = 0, ny = 0, nz = 0;
    std::vector<std::array<double, 3>> Wc;

    double bx(I i, I j, I k) const { return 0.5 * (Bx(i, j, k) + Bx(i + 1, j, k)); }
    double by(I i, I j, I k) const { return 0.5 * (By(i, j, k) + By(i, j + 1, k)); }
    double bz(I i, I j, I k) const { return 0.5 * (Bz(i, j, k) + Bz(i, j, k + 1)); }
    double Btx(I i, I j, I k) const { return 0.5 * (bx(i - 1, j, k) + bx(i, j, k)); }
    double Bty(I i, I j, I k) const { return 0.5 * (by(i, j - 1, k) + by(i, j, k)); }
    double Btz(I i, I j, I k) const { return 0.5 * (bz(i, j, k - 1) + bz(i, j, k)); }
    // experiment PHARE_ZW_RAWJ=1: raw Yee J = curl^T B instead of J~ = curl^T C^T C B in the node
    // collocation (W, E_H and F_H formulas unchanged; no longer zero-work)
    static inline bool const rawJ_ = std::getenv("PHARE_ZW_RAWJ") != nullptr;
    double Jtx(I i, I j, I k) const
    {
        if (rawJ_)
            return (Bz(i, j, k) - Bz(i, j - 1, k)) * iy - (By(i, j, k) - By(i, j, k - 1)) * iz;
        return (Btz(i, j, k) - Btz(i, j - 1, k)) * iy - (Bty(i, j, k) - Bty(i, j, k - 1)) * iz;
    }
    double Jty(I i, I j, I k) const
    {
        if (rawJ_)
            return (Bx(i, j, k) - Bx(i, j, k - 1)) * iz - (Bz(i, j, k) - Bz(i - 1, j, k)) * ix;
        return (Btx(i, j, k) - Btx(i, j, k - 1)) * iz - (Btz(i, j, k) - Btz(i - 1, j, k)) * ix;
    }
    double Jtz(I i, I j, I k) const
    {
        if (rawJ_)
            return (By(i, j, k) - By(i - 1, j, k)) * ix - (Bx(i, j, k) - Bx(i, j - 1, k)) * iy;
        return (Bty(i, j, k) - Bty(i - 1, j, k)) * ix - (Btx(i, j, k) - Btx(i, j - 1, k)) * iy;
    }
    std::array<double, 3> Wcompute(I i, I j, I k) const
    {
        double const jx  = 0.5 * (Jtx(i - 1, j, k) + Jtx(i, j, k));
        double const jy  = 0.5 * (Jty(i, j - 1, k) + Jty(i, j, k));
        double const jz  = 0.5 * (Jtz(i, j, k - 1) + Jtz(i, j, k));
        double const bnx = 0.25 * (Bx(i, j, k) + Bx(i, j - 1, k) + Bx(i, j, k - 1) + Bx(i, j - 1, k - 1));
        double const bny = 0.25 * (By(i, j, k) + By(i - 1, j, k) + By(i, j, k - 1) + By(i - 1, j, k - 1));
        double const bnz = 0.25 * (Bz(i, j, k) + Bz(i - 1, j, k) + Bz(i, j - 1, k) + Bz(i - 1, j - 1, k));
        double rs        = 0;
        for (I a = 0; a < 2; ++a)
            for (I b = 0; b < 2; ++b)
                for (I c = 0; c < 2; ++c)
                    rs += rho(i - a, j - b, k - c);
        double const ir = 8.0 / rs;
        return {(jy * bnz - jz * bny) * ir, (jz * bnx - jx * bnz) * ir, (jx * bny - jy * bnx) * ir};
    }
    void cache()
    {
        // nodes [2, n-3] per axis: domain nodes +-1 are what E_H and F_H read; W reaches B +-2
        nx = Bx.shape()[0];
        ny = By.shape()[1];
        nz = Bz.shape()[2];
        Wc.assign(nx * ny * nz, {0., 0., 0.});
        for (I i = 2; i + 3 <= nx; ++i)
            for (I j = 2; j + 3 <= ny; ++j)
                for (I k = 2; k + 3 <= nz; ++k)
                    Wc[(i * ny + j) * nz + k] = Wcompute(i, j, k);
    }
    std::array<double, 3> const& W(I i, I j, I k) const { return Wc[(i * ny + j) * nz + k]; }

    double Ex(I i, I j, I k) const { return 0.5 * (W(i, j, k)[0] + W(i + 1, j, k)[0]); }
    double Ey(I i, I j, I k) const { return 0.5 * (W(i, j, k)[1] + W(i, j + 1, k)[1]); }
    double Ez(I i, I j, I k) const { return 0.5 * (W(i, j, k)[2] + W(i, j, k + 1)[2]); }
    double cx(I i, I j, I k) const
    {
        return (Ez(i, j + 1, k) - Ez(i, j, k)) * iy - (Ey(i, j, k + 1) - Ey(i, j, k)) * iz;
    }
    double cy(I i, I j, I k) const
    {
        return (Ex(i, j, k + 1) - Ex(i, j, k)) * iz - (Ez(i + 1, j, k) - Ez(i, j, k)) * ix;
    }
    double cz(I i, I j, I k) const
    {
        return (Ey(i + 1, j, k) - Ey(i, j, k)) * ix - (Ex(i, j + 1, k) - Ex(i, j, k)) * iy;
    }
    double Fx(I i, I j, I k) const
    {
        return Ey(i, j, k) * Btz(i - 1, j, k) - Ez(i, j, k) * Bty(i - 1, j, k)
               + 0.5 / ix * (bx(i - 1, j, k) * cx(i, j, k) + W(i, j, k)[0] * Jtx(i - 1, j, k));
    }
    double Fy(I i, I j, I k) const
    {
        return Ez(i, j, k) * Btx(i, j - 1, k) - Ex(i, j, k) * Btz(i, j - 1, k)
               + 0.5 / iy * (by(i, j - 1, k) * cy(i, j, k) + W(i, j, k)[1] * Jty(i, j - 1, k));
    }
    double Fz(I i, I j, I k) const
    {
        return Ex(i, j, k) * Bty(i, j, k - 1) - Ey(i, j, k) * Btx(i, j, k - 1)
               + 0.5 / iz * (bz(i, j, k - 1) * cz(i, j, k) + W(i, j, k)[2] * Jtz(i, j, k - 1));
    }
};

// dimension dispatch: the zero-work Hall stencils of this patch
auto make_zero_work_hall(auto const& B, auto const& rho, auto const& layout)
{
    using Field_t   = std::decay_t<decltype(B(Component::X))>;
    using Rho_t     = std::decay_t<decltype(rho)>;
    constexpr auto d = std::decay_t<decltype(layout)>::dimension;
    if constexpr (d == 1)
        return ZeroWorkHall1D<Field_t, Rho_t>{B(Component::X), B(Component::Y), B(Component::Z), rho,
                                              layout.inverseMeshSize(Direction::X)};
    else if constexpr (d == 2)
        return make_zero_work_hall_2d(B, rho, layout);
    else
    {
        ZeroWorkHall3D<Field_t, Rho_t> zw{B(Component::X),
                                          B(Component::Y),
                                          B(Component::Z),
                                          rho,
                                          layout.inverseMeshSize(Direction::X),
                                          layout.inverseMeshSize(Direction::Y),
                                          layout.inverseMeshSize(Direction::Z)};
        zw.cache();
        return zw;
    }
}

// F_H on face d of index idx (uniform access for the FH1 check)
template<std::size_t d>
double hall_energy_flux_at(auto const& zw, auto const&... idx)
{
    if constexpr (d == 0)
        return zw.Fx(idx...);
    else if constexpr (d == 1)
        return zw.Fy(idx...);
    else
        return zw.Fz(idx...);
}

// experiment PHARE_HALL_FLUX=1: add F_H to the Etot face fluxes of one patch (1D, 2D, 3D)
void add_hall_energy_flux(auto& fluxes, auto const& B, auto const& rho, auto const& layout)
{
    constexpr auto d = std::decay_t<decltype(layout)>::dimension;
    auto const zw    = make_zero_work_hall(B, rho, layout);
    layout.evalOnBox(fluxes.Etot_fx, [&](auto... i) { fluxes.Etot_fx(i...) += zw.Fx(i...); });
    if constexpr (d >= 2)
        layout.evalOnBox(fluxes.Etot_fy, [&](auto... i) { fluxes.Etot_fy(i...) += zw.Fy(i...); });
    if constexpr (d == 3)
        layout.evalOnBox(fluxes.Etot_fz, [&](auto... i) { fluxes.Etot_fz(i...) += zw.Fz(i...); });
}

// the face-pair Hall flux of this patch (2D: 4 reflections, 3D: 8)
auto make_face_pair_flux(auto const& B, auto const& rho, auto const& layout)
{
    using Field_t    = std::decay_t<decltype(B(Component::X))>;
    using Rho_t      = std::decay_t<decltype(rho)>;
    constexpr auto d = std::decay_t<decltype(layout)>::dimension;
    static_assert(d >= 2);
    if constexpr (d == 2)
        return FacePairFluxSym<Field_t, Rho_t>{B(Component::X), B(Component::Y), B(Component::Z), rho,
                                               layout.inverseMeshSize(Direction::X),
                                               layout.inverseMeshSize(Direction::Y)};
    else
        return FacePairFluxSym3D<Field_t, Rho_t>{B(Component::X),
                                                 B(Component::Y),
                                                 B(Component::Z),
                                                 rho,
                                                 layout.inverseMeshSize(Direction::X),
                                                 layout.inverseMeshSize(Direction::Y),
                                                 layout.inverseMeshSize(Direction::Z)};
}

// experiment PHARE_HALL_FLUX_FACE=1 (2D, 3D): add the mirror-averaged face-pair Hall flux to Etot_f*.
// PHARE_HALL_FLUX_FACE_ONESIDED=1 adds the one-sided flux instead (same divergence, 1st-order face
// values: the FF-3 reflux control).
void add_hall_energy_flux_face(auto& fluxes, auto const& B, auto const& rho, auto const& layout)
{
    static bool const onesided = std::getenv("PHARE_HALL_FLUX_FACE_ONESIDED") != nullptr;
    constexpr auto d           = std::decay_t<decltype(layout)>::dimension;
    auto fs                    = make_face_pair_flux(B, rho, layout);
    // 3D: J, W, E of the face-pair field cached per patch; the 8 reflected frames read them at mirrored
    // indices (PHARE_FF_NOCACHE=1: recompute in every frame, the reference for the bitwise check)
    static bool const nocache = std::getenv("PHARE_FF_NOCACHE") != nullptr;
    using Cache_t = std::conditional_t<d == 3, FacePairCache3D, int>;
    Cache_t cache{};
    if constexpr (d == 3)
        if (!nocache)
        {
            cache.build(fs.hall(), rho.shape()[0], rho.shape()[1], rho.shape()[2]);
            fs.cache = &cache;
        }
    layout.evalOnBox(fluxes.Etot_fx, [&](auto... i) {
        fluxes.Etot_fx(i...) += onesided ? fs.Fx_onesided(i...) : fs.Fx(i...);
    });
    layout.evalOnBox(fluxes.Etot_fy, [&](auto... i) {
        fluxes.Etot_fy(i...) += onesided ? fs.Fy_onesided(i...) : fs.Fy(i...);
    });
    if constexpr (d == 3)
        layout.evalOnBox(fluxes.Etot_fz, [&](auto... i) {
            fluxes.Etot_fz(i...) += onesided ? fs.Fz_onesided(i...) : fs.Fz(i...);
        });
}

using UpwindConstrainedTransportInfo = OhmInfo;

template<typename GridLayout, template<typename> typename Reconstruction, bool Hall>
class UpwindConstrainedTransport : UpwindConstrainedTransportInfo
{
    using Super                     = UpwindConstrainedTransportInfo;
    using Reconstruction_t          = Reconstruction<GridLayout>;
    using EdgeB_t                   = UpwindEdgeB<GridLayout, Reconstruction>;
    constexpr static auto dimension = GridLayout::dimension;
    using Super::hyper_mode;

public:
    using Info_t = Super;

    UpwindConstrainedTransport(UpwindConstrainedTransportInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
        , is_dissipative_{info.isResistive() || info.isHyperResistive()}
    {
    }

    void operator()(auto& ct_state, auto const& dissipative_electric_state, auto& mhd_state) const
    {
        auto& E       = mhd_state.E;
        auto const& B = mhd_state.B;

        auto& Ex = E(Component::X);
        auto& Ey = E(Component::Y);
        auto& Ez = E(Component::Z);

        layout_.evalOnBox(Ex, [&](auto&... args) { ExEq_(ct_state, Ex, B, {args...}); });
        layout_.evalOnBox(Ey, [&](auto&... args) { EyEq_(ct_state, Ey, B, {args...}); });
        layout_.evalOnBox(Ez, [&](auto&... args) { EzEq_(ct_state, Ez, B, {args...}); });

        if constexpr (Hall)
            if (zeroWork_ && !hallOff_ && (parts_ & 4))
                addZeroWorkHall_(E, B, mhd_state.rho);
        if constexpr (Hall && dimension == 2)
            if (hallEdge_ && !hallOff_ && (parts_ & 4))
                addEdgeHall_(E, B, mhd_state.rho);
        if constexpr (Hall && dimension >= 2)
            if (hallFacePair_ && !hallOff_ && (parts_ & 4))
                addFacePairHall_(E, B, mhd_state.rho);

        if (is_dissipative_ && (parts_ & 8))
        {
            auto const& Ediss = dissipative_electric_state.E();
            for (auto c : {Component::X, Component::Y, Component::Z})
                layout_.evalOnBox(E(c), [&](auto&... args) { E(c)(args...) += Ediss(c)(args...); });
        }
    }

    // experiment: closure probe evaluates each part of E separately
    // bits: 1 advective (v x B), 2 fan diffusion, 4 Hall, 8 explicit eta/nu
    void setParts(int const parts) { parts_ = parts; }

private:
    void ExEq_(auto& ct_state, auto& Ex, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto const [aL, aR, dL, dR, BzL, BzR]
                = EdgeB_t::template states<Component::X, Component::Z>(ct_state, B, idx);

            auto FL = BzL * ct_state.vt_y(Component::Y)(idx)
                      - B(Component::Y)(idx) * ct_state.vt_y(Component::Z)(idx);
            auto FR = BzR * ct_state.vt_y(Component::Y)(idx)
                      - B(Component::Y)(idx) * ct_state.vt_y(Component::Z)(idx);

            if (parts_ == 15)
                Ex(idx) = -(aL * FL + aR * FR) + (dR * BzR - dL * BzL);
            else
                Ex(idx) = ((parts_ & 1) ? -(aL * FL + aR * FR) : 0.)
                          + ((parts_ & 2) ? (dR * BzR - dL * BzL) : 0.);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / ct_state.template getRhot<Direction::Y>()(idx);
                auto JxB_x_L = ct_state.template getJt<Direction::Y>()(Component::Y)(idx)*BzL
                               - ct_state.template getJt<Direction::Y>()(Component::Z)(idx)*B(
                                   Component::Y)(idx);
                auto JxB_x_R = ct_state.template getJt<Direction::Y>()(Component::Y)(idx)*BzR
                               - ct_state.template getJt<Direction::Y>()(Component::Z)(idx)*B(
                                   Component::Y)(idx);

                auto HallL = -JxB_x_L * invRho;
                auto HallR = -JxB_x_R * invRho;

                auto F_Bz_y = aL * HallL + aR * HallR;

                if ((parts_ & 4) && !altHall_)
                    Ex(idx) += -F_Bz_y;
            }
        }
        else if constexpr (dimension == 3)
        {
            auto const [aS, aN, dS, dN, BzS, BzN]
                = EdgeB_t::template states<Component::X, Component::Z>(ct_state, B, idx);
            auto const [aB, aT, dB, dT, ByB, ByT]
                = EdgeB_t::template states<Component::X, Component::Y>(ct_state, B, idx);

            auto [vyS, vyN] = Reconstruction_t::template reconstruct<Direction::Y>(
                ct_state.vt_z(Component::Y), idx);
            auto [vzB, vzT] = Reconstruction_t::template reconstruct<Direction::Z>(
                ct_state.vt_y(Component::Z), idx);

            if (parts_ == 15)
                Ex(idx) = (aB * vzB * ByB + aT * vzT * ByT) - (aS * vyS * BzS + aN * vyN * BzN)
                          - (dT * ByT - dB * ByB) + (dN * BzN - dS * BzS);
            else
                Ex(idx) = ((parts_ & 1) ? (aB * vzB * ByB + aT * vzT * ByT)
                                              - (aS * vyS * BzS + aN * vyN * BzN)
                                        : 0.)
                          + ((parts_ & 2) ? -(dT * ByT - dB * ByB) + (dN * BzN - dS * BzS) : 0.);

            if constexpr (Hall)
            {
                auto [jyS, jyN] = Reconstruction_t::template reconstruct<Direction::Y>(
                    ct_state.template getJt<Direction::Z>()(Component::Y), idx);
                auto [jzB, jzT] = Reconstruction_t::template reconstruct<Direction::Z>(
                    ct_state.template getJt<Direction::Y>()(Component::Z), idx);

                auto [rhoS, rhoN] = Reconstruction_t::template reconstruct<Direction::Y>(
                    ct_state.template getRhot<Direction::Z>(), idx);
                auto [rhoB, rhoT] = Reconstruction_t::template reconstruct<Direction::Z>(
                    ct_state.template getRhot<Direction::Y>(), idx);

                if ((parts_ & 4) && !altHall_)
                    Ex(idx) += -(aB * jzB * ByB / rhoB + aT * jzT * ByT / rhoT)
                               + (aS * jyS * BzS / rhoS + aN * jyN * BzN / rhoN);
            }
        }
    }

    void EyEq_(auto& ct_state, auto& Ey, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension <= 2)
        {
            auto const [aL, aR, dL, dR, BzL, BzR]
                = EdgeB_t::template states<Component::Y, Component::Z>(ct_state, B, idx);

            auto FL = BzL * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Z)(idx);
            auto FR = BzR * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Z)(idx);

            if (parts_ == 15)
                Ey(idx) = (aL * FL + aR * FR) - (dR * BzR - dL * BzL);
            else
                Ey(idx) = ((parts_ & 1) ? (aL * FL + aR * FR) : 0.)
                          - ((parts_ & 2) ? (dR * BzR - dL * BzL) : 0.);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / ct_state.template getRhot<Direction::X>()(idx);
                auto JxB_y_L = ct_state.template getJt<Direction::X>()(Component::Z)(idx)*B(
                                   Component::X)(idx)
                               - ct_state.template getJt<Direction::X>()(Component::X)(idx)*BzL;
                auto JxB_y_R = ct_state.template getJt<Direction::X>()(Component::Z)(idx)*B(
                                   Component::X)(idx)
                               - ct_state.template getJt<Direction::X>()(Component::X)(idx)*BzR;

                auto HallL = JxB_y_L * invRho;
                auto HallR = JxB_y_R * invRho;

                auto F_Bz_x = aL * HallL + aR * HallR;

                if ((parts_ & 4) && !altHall_)
                    Ey(idx) += F_Bz_x;
            }
        }
        else if constexpr (dimension == 3)
        {
            auto const [aW, aE, dW, dE, BzW, BzE]
                = EdgeB_t::template states<Component::Y, Component::Z>(ct_state, B, idx);
            auto const [aB, aT, dB, dT, BxB, BxT]
                = EdgeB_t::template states<Component::Y, Component::X>(ct_state, B, idx);

            auto [vxW, vxE] = Reconstruction_t::template reconstruct<Direction::X>(
                ct_state.vt_z(Component::X), idx);
            auto [vzB, vzT] = Reconstruction_t::template reconstruct<Direction::Z>(
                ct_state.vt_x(Component::Z), idx);

            if (parts_ == 15)
                Ey(idx) = (aW * vxW * BzW + aE * vxE * BzE) - (aB * vzB * BxB + aT * vzT * BxT)
                          - (dE * BzE - dW * BzW) + (dT * BxT - dB * BxB);
            else
                Ey(idx) = ((parts_ & 1) ? (aW * vxW * BzW + aE * vxE * BzE)
                                              - (aB * vzB * BxB + aT * vzT * BxT)
                                        : 0.)
                          + ((parts_ & 2) ? -(dE * BzE - dW * BzW) + (dT * BxT - dB * BxB) : 0.);

            if constexpr (Hall)
            {
                auto [jxW, jxE] = Reconstruction_t::template reconstruct<Direction::X>(
                    ct_state.template getJt<Direction::Z>()(Component::X), idx);
                auto [jzB, jzT] = Reconstruction_t::template reconstruct<Direction::Z>(
                    ct_state.template getJt<Direction::X>()(Component::Z), idx);
                auto [rhoW, rhoE] = Reconstruction_t::template reconstruct<Direction::X>(
                    ct_state.template getRhot<Direction::Z>(), idx);
                auto [rhoB, rhoT] = Reconstruction_t::template reconstruct<Direction::Z>(
                    ct_state.template getRhot<Direction::X>(), idx);
                if ((parts_ & 4) && !altHall_)
                    Ey(idx) += -(aW * jxW * BzW / rhoW + aE * jxE * BzE / rhoE)
                               + (aB * jzB * BxB / rhoB + aT * jzT * BxT / rhoT);
            }
        }
    }

    void EzEq_(auto& ct_state, auto& Ez, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 1)
        {
            auto const [aL, aR, dL, dR, ByL, ByR]
                = EdgeB_t::template states<Component::Z, Component::Y>(ct_state, B, idx);

            auto FL = ByL * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Y)(idx);
            auto FR = ByR * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Y)(idx);

            if (parts_ == 15)
                Ez(idx) = -(aL * FL + aR * FR) + (dR * ByR - dL * ByL);
            else
                Ez(idx) = ((parts_ & 1) ? -(aL * FL + aR * FR) : 0.)
                          + ((parts_ & 2) ? (dR * ByR - dL * ByL) : 0.);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / ct_state.template getRhot<Direction::X>()(idx);
                auto JxB_z_L = ct_state.template getJt<Direction::X>()(Component::X)(idx)*ByL
                               - ct_state.template getJt<Direction::X>()(Component::Y)(idx)*B(
                                   Component::X)(idx);
                auto JxB_z_R = ct_state.template getJt<Direction::X>()(Component::X)(idx)*ByR
                               - ct_state.template getJt<Direction::X>()(Component::Y)(idx)*B(
                                   Component::X)(idx);

                auto HallL = -JxB_z_L * invRho;
                auto HallR = -JxB_z_R * invRho;

                auto F_By_x = aL * HallL + aR * HallR;

                if ((parts_ & 4) && !altHall_)
                    Ez(idx) += -F_By_x;
            }
        }
        else if constexpr (dimension >= 2)
        {
            auto const [aW, aE, dW, dE, ByW, ByE]
                = EdgeB_t::template states<Component::Z, Component::Y>(ct_state, B, idx);
            auto const [aS, aN, dS, dN, BxS, BxN]
                = EdgeB_t::template states<Component::Z, Component::X>(ct_state, B, idx);

            auto [vyS, vyN] = Reconstruction_t::template reconstruct<Direction::Y>(
                ct_state.vt_x(Component::Y), idx);
            auto [vxW, vxE] = Reconstruction_t::template reconstruct<Direction::X>(
                ct_state.vt_y(Component::X), idx);

            if (parts_ == 15)
                Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                          + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);
            else
                Ez(idx) = ((parts_ & 1) ? -(aW * vxW * ByW + aE * vxE * ByE)
                                              + (aS * vyS * BxS + aN * vyN * BxN)
                                        : 0.)
                          + ((parts_ & 2) ? (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS) : 0.);

            if constexpr (Hall)
            {
                auto [jyS, jyN] = Reconstruction_t::template reconstruct<Direction::Y>(
                    ct_state.template getJt<Direction::X>()(Component::Y), idx);
                auto [jxW, jxE] = Reconstruction_t::template reconstruct<Direction::X>(
                    ct_state.template getJt<Direction::Y>()(Component::X), idx);

                auto [rhoS, rhoN] = Reconstruction_t::template reconstruct<Direction::Y>(
                    ct_state.template getRhot<Direction::X>(), idx);
                auto [rhoW, rhoE] = Reconstruction_t::template reconstruct<Direction::X>(
                    ct_state.template getRhot<Direction::Y>(), idx);

                if ((parts_ & 4) && !altHall_)
                    Ez(idx) += (aW * jxW * ByW / rhoW + aE * jxE * ByE / rhoE)
                               - (aS * jyS * BxS / rhoS + aN * jyN * BxN / rhoN);
            }
        }
    }

    // experiment PHARE_HALL_ZERO_WORK=1: Hall EMF with zero discrete work, E_H = A^T W (1D/2D/3D
    // stencils in ZeroWorkHall*, shared with the Hall energy flux F_H)
    void addZeroWorkHall_(auto& E, auto const& B, auto const& rho) const
    {
        auto const zw = make_zero_work_hall(B, rho, layout_);
        auto& Ex      = E(Component::X);
        auto& Ey      = E(Component::Y);
        auto& Ez      = E(Component::Z);
        if constexpr (dimension == 1)
        {
            // Ex is untouched, as in the upwind 1D path (it does not enter dB/dt or the x Poynting flux)
            layout_.evalOnBox(Ey, [&](auto i) { Ey(i) += zw.Ey(i); });
            layout_.evalOnBox(Ez, [&](auto i) { Ez(i) += zw.Ez(i); });
        }
        else if constexpr (dimension == 2)
        {
            layout_.evalOnBox(Ex, [&](auto i, auto j) {
                Ex(i, j) += 0.5 * (zw.W(i, j)[0] + zw.W(i + 1, j)[0]);
            });
            layout_.evalOnBox(Ey, [&](auto i, auto j) {
                Ey(i, j) += 0.5 * (zw.W(i, j)[1] + zw.W(i, j + 1)[1]);
            });
            layout_.evalOnBox(Ez, [&](auto i, auto j) { Ez(i, j) += zw.W(i, j)[2]; });
        }
        else
        {
            layout_.evalOnBox(Ex, [&](auto i, auto j, auto k) { Ex(i, j, k) += zw.Ex(i, j, k); });
            layout_.evalOnBox(Ey, [&](auto i, auto j, auto k) { Ey(i, j, k) += zw.Ey(i, j, k); });
            layout_.evalOnBox(Ez, [&](auto i, auto j, auto k) { Ez(i, j, k) += zw.Ez(i, j, k); });
        }
    }

    // experiment PHARE_HALL_EDGE=1 (2D, "option B"): E_H = (J x B)/rho on each Yee edge (edge_hall.hpp)
    void addEdgeHall_(auto& E, auto const& B, auto const& rho) const
    {
        using Field_t = std::decay_t<decltype(B(Component::X))>;
        using Rho_t   = std::decay_t<decltype(rho)>;
        EdgeHall2D<Field_t, Rho_t> const eh{B(Component::X), B(Component::Y), B(Component::Z), rho,
                                            layout_.inverseMeshSize(Direction::X),
                                            layout_.inverseMeshSize(Direction::Y)};
        auto& Ex = E(Component::X);
        auto& Ey = E(Component::Y);
        auto& Ez = E(Component::Z);
        layout_.evalOnBox(Ex, [&](auto i, auto j) { Ex(i, j) += eh.Ex(i, j); });
        layout_.evalOnBox(Ey, [&](auto i, auto j) { Ey(i, j) += eh.Ey(i, j); });
        layout_.evalOnBox(Ez, [&](auto i, auto j) { Ez(i, j) += eh.Ez(i, j); });
    }

    // experiment PHARE_HALL_FACEPAIR=1 (2D, 3D): igirault 622e2babb face-pair Hall field (edge_hall.hpp)
    void addFacePairHall_(auto& E, auto const& B, auto const& rho) const
    {
        using Field_t = std::decay_t<decltype(B(Component::X))>;
        using Rho_t   = std::decay_t<decltype(rho)>;
        auto& Ex      = E(Component::X);
        auto& Ey      = E(Component::Y);
        auto& Ez      = E(Component::Z);
        if constexpr (dimension == 2)
        {
            FacePairHall2D<Field_t, Rho_t> const fp{B(Component::X), B(Component::Y),
                                                    B(Component::Z), rho,
                                                    layout_.inverseMeshSize(Direction::X),
                                                    layout_.inverseMeshSize(Direction::Y)};
            layout_.evalOnBox(Ex, [&](auto i, auto j) { Ex(i, j) += fp.Ex(i, j); });
            layout_.evalOnBox(Ey, [&](auto i, auto j) { Ey(i, j) += fp.Ey(i, j); });
            layout_.evalOnBox(Ez, [&](auto i, auto j) { Ez(i, j) += fp.Ez(i, j); });
        }
        else
        {
            FacePairHall3D<Field_t, Rho_t> const fp{B(Component::X),
                                                    B(Component::Y),
                                                    B(Component::Z),
                                                    rho,
                                                    layout_.inverseMeshSize(Direction::X),
                                                    layout_.inverseMeshSize(Direction::Y),
                                                    layout_.inverseMeshSize(Direction::Z)};
            layout_.evalOnBox(Ex, [&](auto... i) { Ex(i...) += fp.Ex(i...); });
            layout_.evalOnBox(Ey, [&](auto... i) { Ey(i...) += fp.Ey(i...); });
            layout_.evalOnBox(Ez, [&](auto... i) { Ez(i...) += fp.Ez(i...); });
        }
    }

    static inline bool const zeroWork_ = std::getenv("PHARE_HALL_ZERO_WORK") != nullptr;
    static inline bool const hallFacePair_ = std::getenv("PHARE_HALL_FACEPAIR") != nullptr;
    static inline bool const hallEdge_ = std::getenv("PHARE_HALL_EDGE") != nullptr;
    // any replacement of the upwind Hall EMF; option B is 2D only, so it only replaces it in 2D
    // experiment PHARE_HALL_EMF_OFF=1: Hall build with no Hall EMF at all (Hall-off reference that keeps eta/nu)
    static inline bool const hallOff_ = std::getenv("PHARE_HALL_EMF_OFF") != nullptr;
    static inline bool const altHall_
        = zeroWork_ || (dimension == 2 && hallEdge_) || (dimension >= 2 && hallFacePair_) || hallOff_;

    GridLayout layout_;
    bool const is_dissipative_;
    int parts_ = 15;
};
} // namespace PHARE::core

#endif
