#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_LORENTZ_STRESS_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_LORENTZ_STRESS_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace PHARE::core
{
// experiment PHARE_LORENTZ_LC=1 (lorentz/derivation.md, "Implementation entry"; 2D, Rusanov, face
// energy): one discrete Lorentz force for momentum and energy.
//  - ideal EMF E^A = -vbar x Bbar: cell-velocity means around the edge, 2-point B means (no
//  upwinding)
//  - momentum flux: hydro Godunov flux + the conservative stress M (F_c = -div M is the adjoint of
//  E^A)
//  - Etot flux: P_A (central edge-B Poynting of the non-Hall E) + P_B (velocity-jump part of E^A)
// so that work_c(E^A) - v_c . F_c = div(P_A + P_B)_c per cell.
inline bool useLorentzLC()
{
    static bool const v = std::getenv("PHARE_LORENTZ_LC") != nullptr;
    return v;
}

// PHARE local index (primal and dual share the start index):
// Bx(i,j)@x-face (x_i, y_{j+1/2})  By(i,j)@y-face (x_{i+1/2}, y_j)  Bz, V(i,j)@cell
// Ey(i,j)@x-face, Ex(i,j)@y-face, Ez(i,j)@node (x_i, y_j); x-fluxes on x-faces, y-fluxes on
// y-faces.
template<typename BField, typename VField>
struct LorentzLC2D
{
    using I = std::size_t;
    BField const& Bx;
    BField const& By;
    BField const& Bz;
    VField const& Vx;
    VField const& Vy;
    VField const& Vz;
    double ix, iy;

    static double sq(double a) { return a * a; }

    double bxn(I i, I j) const { return 0.5 * (Bx(i, j - 1) + Bx(i, j)); } // node
    double byn(I i, I j) const { return 0.5 * (By(i - 1, j) + By(i, j)); } // node
    double bzx(I i, I j) const { return 0.5 * (Bz(i - 1, j) + Bz(i, j)); } // x-face
    double bzy(I i, I j) const { return 0.5 * (Bz(i, j - 1) + Bz(i, j)); } // y-face
    double Jz(I i, I j) const
    {
        return (By(i, j) - By(i - 1, j)) * ix - (Bx(i, j) - Bx(i, j - 1)) * iy;
    }

    // --- central ideal EMF E^A ---
    double Ey(I i, I j) const // x-face
    {
        double const vx = 0.5 * (Vx(i - 1, j) + Vx(i, j));
        double const vz = 0.5 * (Vz(i - 1, j) + Vz(i, j));
        return -(vz * Bx(i, j) - vx * bzx(i, j));
    }
    double Ex(I i, I j) const // y-face
    {
        double const vy = 0.5 * (Vy(i, j - 1) + Vy(i, j));
        double const vz = 0.5 * (Vz(i, j - 1) + Vz(i, j));
        return -(vy * bzy(i, j) - vz * By(i, j));
    }
    double Ez(I i, I j) const // node
    {
        double const vx = 0.25 * (Vx(i - 1, j - 1) + Vx(i, j - 1) + Vx(i - 1, j) + Vx(i, j));
        double const vy = 0.25 * (Vy(i - 1, j - 1) + Vy(i, j - 1) + Vy(i - 1, j) + Vy(i, j));
        return -(vx * byn(i, j) - vy * bxn(i, j));
    }

    // --- conservative stress M (S3b) ---
    double a4By(I i, I j) const // By at x-face (i,j)
    {
        return 0.25 * (By(i - 1, j) + By(i, j) + By(i - 1, j + 1) + By(i, j + 1));
    }
    double a4Bx(I i, I j) const // Bx at y-face (i,j)
    {
        return 0.25 * (Bx(i, j - 1) + Bx(i + 1, j - 1) + Bx(i, j) + Bx(i + 1, j));
    }
    double px(I i, I j) const { return Bx(i, j) * 0.5 * (Bx(i - 1, j) + Bx(i + 1, j)); }
    double py(I i, I j) const { return By(i, j) * 0.5 * (By(i, j - 1) + By(i, j + 1)); }
    double Qx(I i, I j) const { return Bx(i, j) * a4By(i, j); } // on x-faces
    double Qy(I i, I j) const { return By(i, j) * a4Bx(i, j); } // on y-faces

    std::array<double, 3> Mx(I i, I j) const // momentum flux through x-face (i,j)
    {
        double const a4By2
            = 0.25 * (sq(By(i - 1, j)) + sq(By(i, j)) + sq(By(i - 1, j + 1)) + sq(By(i, j + 1)));
        double const Bz2 = 0.5 * (sq(Bz(i - 1, j)) + sq(Bz(i, j)));
        double const Xx  = 0.125 * (px(i, j - 1) + 2. * px(i, j) + px(i, j + 1));
        return {0.5 * (a4By2 + Bz2) - Xx,
                -0.25 * (Qy(i - 1, j) + Qy(i, j) + Qy(i - 1, j + 1) + Qy(i, j + 1)),
                -Bx(i, j) * bzx(i, j)};
    }
    std::array<double, 3> My(I i, I j) const // momentum flux through y-face (i,j)
    {
        double const a4Bx2
            = 0.25 * (sq(Bx(i, j - 1)) + sq(Bx(i + 1, j - 1)) + sq(Bx(i, j)) + sq(Bx(i + 1, j)));
        double const Bz2 = 0.5 * (sq(Bz(i, j - 1)) + sq(Bz(i, j)));
        double const Xy  = 0.125 * (py(i - 1, j) + 2. * py(i, j) + py(i + 1, j));
        return {-0.25 * (Qx(i, j - 1) + Qx(i + 1, j - 1) + Qx(i, j) + Qx(i + 1, j)),
                0.5 * (a4Bx2 + Bz2) - Xy, -By(i, j) * bzy(i, j)};
    }

    // --- P_B: velocity-jump part of the exact energy flux of E^A (S2c) ---
    struct Arms
    {
        double N, S, E, W;
    };
    Arms arms(I i, I j) const // node (i,j); N -> x-face (i,j), S -> x-face (i,j-1), E -> y-face
                              // (i,j), W -> y-face (i-1,j)
    {
        double const J = 0.25 * Jz(i, j), bx = bxn(i, j), by = byn(i, j);
        auto q           = [&](I a, I b) { return J * (-(Vx(a, b) * by - Vy(a, b) * bx)); };
        double const qSW = q(i - 1, j - 1), qSE = q(i, j - 1), qNW = q(i - 1, j), qNE = q(i, j);
        double const tau = qNE - qNW + qSW - qSE;
        double const dx = 1. / ix, dy = 1. / iy;
        return {dx * (0.5 * (qNE - qNW) - tau / 8.), dx * (0.5 * (qSE - qSW) + tau / 8.),
                dy * (0.5 * (qNE - qSE) - tau / 8.), dy * (0.5 * (qNW - qSW) + tau / 8.)};
    }
    double PBx(I i, I j) const
    {
        double const dEy
            = -((Vz(i, j) - Vz(i - 1, j)) * Bx(i, j) - (Vx(i, j) - Vx(i - 1, j)) * bzx(i, j));
        return -0.25 * (Bz(i, j) - Bz(i - 1, j)) * dEy + arms(i, j).N + arms(i, j + 1).S;
    }
    double PBy(I i, I j) const
    {
        double const dEx
            = -((Vy(i, j) - Vy(i, j - 1)) * bzy(i, j) - (Vz(i, j) - Vz(i, j - 1)) * By(i, j));
        return 0.25 * (Bz(i, j) - Bz(i, j - 1)) * dEx + arms(i, j).E + arms(i + 1, j).W;
    }
};

// add M to the momentum fluxes and P_B to the Etot fluxes of one patch (P_A is added by the
// PoyntingCorrection with the central edge B)
void add_lorentz_lc_fluxes(auto& fluxes, auto const& B, auto const& V, auto const& layout)
{
    using GL = std::decay_t<decltype(layout)>;
    static_assert(GL::dimension == 2);
    auto const& Bx = B(Component::X);
    auto const& Vx = V(Component::X);
    using BF       = std::decay_t<decltype(Bx)>;
    using VF       = std::decay_t<decltype(Vx)>;
    LorentzLC2D<BF, VF> const lc{Bx,
                                 B(Component::Y),
                                 B(Component::Z),
                                 Vx,
                                 V(Component::Y),
                                 V(Component::Z),
                                 layout.inverseMeshSize(Direction::X),
                                 layout.inverseMeshSize(Direction::Y)};
    auto& mx = fluxes.rhoV_fx;
    auto& my = fluxes.rhoV_fy;
    layout.evalOnBox(fluxes.Etot_fx, [&](auto i, auto j) {
        auto const m = lc.Mx(i, j);
        mx(Component::X)(i, j) += m[0];
        mx(Component::Y)(i, j) += m[1];
        mx(Component::Z)(i, j) += m[2];
        fluxes.Etot_fx(i, j) += lc.PBx(i, j);
    });
    layout.evalOnBox(fluxes.Etot_fy, [&](auto i, auto j) {
        auto const m = lc.My(i, j);
        my(Component::X)(i, j) += m[0];
        my(Component::Y)(i, j) += m[1];
        my(Component::Z)(i, j) += m[2];
        fluxes.Etot_fy(i, j) += lc.PBy(i, j);
    });
}

} // namespace PHARE::core

#endif
