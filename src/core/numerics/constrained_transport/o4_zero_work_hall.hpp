#ifndef PHARE_CORE_NUMERICS_O4_ZERO_WORK_HALL_HPP
#define PHARE_CORE_NUMERICS_O4_ZERO_WORK_HALL_HPP

// EXPERIMENT (exp/o4-hall-energy): zero-work fourth-order Hall EMF, 2D.
// Spec: plans/low-beta-blast-sweep/2026-11-06-hall-zero-work/O4/derivation.md S6, numpy mirror
// hall_E_O4() in O4/sympy/s02b_numeric_convergence.py. Every operator below is a line of that
// mirror; array index = PHARE local index (primal and dual share the start index).
//
//   b = K Bbar (cells), Bt = K^T b (faces), Jt = curl_h^T Bt (edges),
//   Jn = L Jt, Bn, rho_n (nodes), W = Jn x Bn / rho_n, Ebar_H = L^T W (edge averages; Ez = Wz).
//
// Out-of-range reads return NaN, so a reach violation shows up as NaN on the consumer box.

#include "core/numerics/constrained_transport/o4_poynting_tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace PHARE::core::o4zw
{
struct Grid2
{
    int nx = 0, ny = 0;
    std::vector<double> a;

    Grid2() = default;
    Grid2(int nx_, int ny_)
        : nx{nx_}
        , ny{ny_}
        , a(static_cast<std::size_t>(nx_ * ny_), std::numeric_limits<double>::quiet_NaN())
    {
    }

    double& operator()(int i, int j) { return a[static_cast<std::size_t>(i * ny + j)]; }
    double operator()(int i, int j) const { return a[static_cast<std::size_t>(i * ny + j)]; }
    double get(int i, int j) const
    {
        if (i < 0 || j < 0 || i >= nx || j >= ny)
            return std::numeric_limits<double>::quiet_NaN();
        return (*this)(i, j);
    }
    // f(idx + a e_x + b e_y), numpy's s(f, a, b)
    double s(int i, int j, int da, int db) const { return get(i + da, j + db); }
};

template<typename Fn>
Grid2 make(Grid2 const& like, Fn&& fn)
{
    Grid2 out{like.nx, like.ny};
    for (int i = 0; i < like.nx; ++i)
        for (int j = 0; j < like.ny; ++j)
            out(i, j) = fn(i, j);
    return out;
}

inline double D2(Grid2 const& f, int ax, int i, int j)
{
    return ax == 0 ? f.s(i, j, 1, 0) - 2 * f(i, j) + f.s(i, j, -1, 0)
                   : f.s(i, j, 0, 1) - 2 * f(i, j) + f.s(i, j, 0, -1);
}

using Stencil = std::array<std::pair<int, double>, 4>;

// face -> cell, cell i <- faces i-1..i+2
inline constexpr Stencil W4{{{-1, -1. / 16}, {0, 9. / 16}, {1, 9. / 16}, {2, -1. / 16}}};
// cell/face -> node, node i <- i-2..i+1
inline constexpr Stencil W4D{{{-2, -1. / 16}, {-1, 9. / 16}, {0, 9. / 16}, {1, -1. / 16}}};
// L: node i <- edges i-2..i+1
inline constexpr Stencil LEN{{{-2, -1. / 24}, {-1, 13. / 24}, {0, 13. / 24}, {1, -1. / 24}}};

inline double st(Grid2 const& f, int ax, Stencil const& w, int i, int j)
{
    double r = 0;
    for (auto const& [o, c] : w)
        r += c * (ax == 0 ? f.s(i, j, o, 0) : f.s(i, j, 0, o));
    return r;
}
inline double stT(Grid2 const& f, int ax, Stencil const& w, int i, int j)
{
    double r = 0;
    for (auto const& [o, c] : w)
        r += c * (ax == 0 ? f.s(i, j, -o, 0) : f.s(i, j, 0, -o));
    return r;
}

struct HallO4Result
{
    Grid2 Ex, Ey, Ez;    // Ebar_H (edge averages, Ez node point value)
    Grid2 Jtx, Jty, Jtz; // Jt_M = curl_h^T K^T K Bbar (edges)
    Grid2 bx, by, bz;    // b = K Bbar (cells)
    Grid2 Btx, Bty, Btz; // Bt = K^T K Bbar (faces)
    Grid2 Wx, Wy, Wz;    // node W
    double pairWorkMax = 0, pairWorkScale = 0; // face-pair route only
};

// Inputs: face averages Bx, By, cell average Bz and rho (2D), local index; dx, dy mesh sizes.
inline HallO4Result hallE(Grid2 const& Bx, Grid2 const& By, Grid2 const& Bz, Grid2 const& rho,
                          double dx, double dy)
{
    HallO4Result r;
    // K = P4 . N : face averages -> cell point values (what p_pv sees)
    auto Bxp = make(Bx, [&](int i, int j) { return Bx(i, j) - D2(Bx, 1, i, j) / 24; });
    auto Byp = make(By, [&](int i, int j) { return By(i, j) - D2(By, 0, i, j) / 24; });
    auto Bzp = make(
        Bz, [&](int i, int j) { return Bz(i, j) - (D2(Bz, 0, i, j) + D2(Bz, 1, i, j)) / 24; });
    r.bx = make(Bx, [&](int i, int j) { return st(Bxp, 0, W4, i, j); });
    r.by = make(By, [&](int i, int j) { return st(Byp, 1, W4, i, j); });
    r.bz = Bzp;
    // K^T : cell -> face (P4^T then N)
    auto tx = make(Bx, [&](int i, int j) { return stT(r.bx, 0, W4, i, j); });
    auto ty = make(By, [&](int i, int j) { return stT(r.by, 1, W4, i, j); });
    r.Btx   = make(Bx, [&](int i, int j) { return tx(i, j) - D2(tx, 1, i, j) / 24; });
    r.Bty   = make(By, [&](int i, int j) { return ty(i, j) - D2(ty, 0, i, j) / 24; });
    r.Btz   = make(Bz, [&](int i, int j) {
        return r.bz(i, j) - (D2(r.bz, 0, i, j) + D2(r.bz, 1, i, j)) / 24;
    });
    // Jt_M = curl_h^T Bt (backward), edges
    r.Jtx = make(Bx, [&](int i, int j) { return (r.Btz(i, j) - r.Btz.s(i, j, 0, -1)) / dy; });
    r.Jty = make(Bx, [&](int i, int j) { return -(r.Btz(i, j) - r.Btz.s(i, j, -1, 0)) / dx; });
    r.Jtz = make(Bx, [&](int i, int j) {
        return (r.Bty(i, j) - r.Bty.s(i, j, -1, 0)) / dx
               - (r.Btx(i, j) - r.Btx.s(i, j, 0, -1)) / dy;
    });
    // node values
    auto Jnx  = make(Bx, [&](int i, int j) { return st(r.Jtx, 0, LEN, i, j); });
    auto Jny  = make(Bx, [&](int i, int j) { return st(r.Jty, 1, LEN, i, j); });
    auto& Jnz = r.Jtz;
    auto Bnx  = make(Bx, [&](int i, int j) { return st(Bxp, 1, W4D, i, j); });
    auto Bny  = make(Bx, [&](int i, int j) { return st(Byp, 0, W4D, i, j); });
    auto Bzx  = make(Bx, [&](int i, int j) { return st(Bzp, 0, W4D, i, j); });
    auto Bnz  = make(Bx, [&](int i, int j) { return st(Bzx, 1, W4D, i, j); });
    auto rp   = make(
        rho, [&](int i, int j) { return rho(i, j) - (D2(rho, 0, i, j) + D2(rho, 1, i, j)) / 24; });
    auto rx = make(Bx, [&](int i, int j) { return st(rp, 0, W4D, i, j); });
    auto rn = make(Bx, [&](int i, int j) { return st(rx, 1, W4D, i, j); });
    r.Wx    = make(Bx, [&](int i, int j) {
        return (Jny(i, j) * Bnz(i, j) - Jnz(i, j) * Bny(i, j)) / rn(i, j);
    });
    r.Wy    = make(Bx, [&](int i, int j) {
        return (Jnz(i, j) * Bnx(i, j) - Jnx(i, j) * Bnz(i, j)) / rn(i, j);
    });
    r.Wz    = make(Bx, [&](int i, int j) {
        return (Jnx(i, j) * Bny(i, j) - Jny(i, j) * Bnx(i, j)) / rn(i, j);
    });
    // L^T: edge averages (Ez node point value in 2D)
    r.Ex = make(Bx, [&](int i, int j) { return stT(r.Wx, 0, LEN, i, j); });
    r.Ey = make(Bx, [&](int i, int j) { return stT(r.Wy, 1, LEN, i, j); });
    r.Ez = r.Wz;
    return r;
}


// ---------------------------------------------------------------------------------------------
// Closure-consistent O4 Poynting flux, point-value face flux F_pv for the pipeline
// (O4/pipeline/derivation.md consolidation; numpy reference pipeline_flux() in
// O4/pipeline/sympy/q03_numeric_gates.py with chain()/fluxes()/mirror_average() of
// O4/poynting/sympy/p04_accuracy.py). Single mesh size h (dx = dy).
//
// centerings (x, y): Bx (p,d) By (d,p) Bz (d,d) | Ex (d,p) Ey (p,d) Ez (p,p) | Fx (p,d) Fy (d,p)

struct Chain
{
    Grid2 b[3], Bt[3], Jt[3], X[3], E[3];
    Grid2 Q, w4;
};

inline Chain chain(Grid2 const* B, Grid2 const* E, double h, bool withQ)
{
    Chain c;
    auto const& Bx = B[0];
    auto const& By = B[1];
    auto const& Bz = B[2];
    auto Bxp  = make(Bx, [&](int i, int j) { return Bx(i, j) - D2(Bx, 1, i, j) / 24; });
    auto Byp  = make(By, [&](int i, int j) { return By(i, j) - D2(By, 0, i, j) / 24; });
    auto Bzp  = make(
        Bz, [&](int i, int j) { return Bz(i, j) - (D2(Bz, 0, i, j) + D2(Bz, 1, i, j)) / 24; });
    c.b[0]    = make(Bx, [&](int i, int j) { return st(Bxp, 0, W4, i, j); });
    c.b[1]    = make(Bx, [&](int i, int j) { return st(Byp, 1, W4, i, j); });
    c.b[2]    = Bzp;
    auto tx   = make(Bx, [&](int i, int j) { return stT(c.b[0], 0, W4, i, j); });
    auto ty   = make(Bx, [&](int i, int j) { return stT(c.b[1], 1, W4, i, j); });
    c.Bt[0]   = make(Bx, [&](int i, int j) { return tx(i, j) - D2(tx, 1, i, j) / 24; });
    c.Bt[1]   = make(Bx, [&](int i, int j) { return ty(i, j) - D2(ty, 0, i, j) / 24; });
    c.Bt[2]   = make(Bx, [&](int i, int j) {
        return c.b[2](i, j) - (D2(c.b[2], 0, i, j) + D2(c.b[2], 1, i, j)) / 24;
    });
    auto& Bt  = c.Bt;
    c.Jt[0]   = make(Bx, [&](int i, int j) { return (Bt[2](i, j) - Bt[2].s(i, j, 0, -1)) / h; });
    c.Jt[1]   = make(Bx, [&](int i, int j) { return -(Bt[2](i, j) - Bt[2].s(i, j, -1, 0)) / h; });
    c.Jt[2]   = make(Bx, [&](int i, int j) {
        return ((Bt[1](i, j) - Bt[1].s(i, j, -1, 0)) - (Bt[0](i, j) - Bt[0].s(i, j, 0, -1))) / h;
    });
    for (int k = 0; k < 3; ++k)
        c.E[k] = E[k];
    auto const& e = c.E;
    c.X[0] = make(Bx, [&](int i, int j) { return (e[2].s(i, j, 0, 1) - e[2](i, j)) / h; });
    c.X[1] = make(Bx, [&](int i, int j) { return -(e[2].s(i, j, 1, 0) - e[2](i, j)) / h; });
    c.X[2] = make(Bx, [&](int i, int j) {
        return ((e[1].s(i, j, 1, 0) - e[1](i, j)) - (e[0].s(i, j, 0, 1) - e[0](i, j))) / h;
    });
    if (withQ)
    {
        auto const& X = c.X;
        auto kx0 = make(Bx, [&](int i, int j) { return X[0](i, j) - D2(X[0], 1, i, j) / 24; });
        auto kx1 = make(Bx, [&](int i, int j) { return X[1](i, j) - D2(X[1], 0, i, j) / 24; });
        auto KX0 = make(Bx, [&](int i, int j) { return st(kx0, 0, W4, i, j); });
        auto KX1 = make(Bx, [&](int i, int j) { return st(kx1, 1, W4, i, j); });
        auto KX2 = make(Bx, [&](int i, int j) {
            return X[2](i, j) - (D2(X[2], 0, i, j) + D2(X[2], 1, i, j)) / 24;
        });
        c.Q = make(Bx, [&](int i, int j) {
            return -(c.b[0](i, j) * KX0(i, j) + c.b[1](i, j) * KX1(i, j)
                     + c.b[2](i, j) * KX2(i, j));
        });
        auto const& Jt = c.Jt;
        auto om0 = make(Bx, [&](int i, int j) {
            return e[0](i, j) * Jt[0](i, j)
                   + (e[0](i, j) * D2(Jt[0], 0, i, j) - D2(e[0], 0, i, j) * Jt[0](i, j)) / 24;
        });
        auto om1 = make(Bx, [&](int i, int j) {
            return e[1](i, j) * Jt[1](i, j)
                   + (e[1](i, j) * D2(Jt[1], 1, i, j) - D2(e[1], 1, i, j) * Jt[1](i, j)) / 24;
        });
        auto om2  = make(Bx, [&](int i, int j) { return e[2](i, j) * Jt[2](i, j); });
        auto om2x = make(Bx, [&](int i, int j) { return st(om2, 0, W4, i, j); });
        c.w4      = make(Bx, [&](int i, int j) {
            return st(om0, 1, W4, i, j) + st(om1, 0, W4, i, j) + st(om2x, 1, W4, i, j);
        });
    }
    return c;
}

// F_E4 = F_E + Psi (p04_accuracy.fluxes, second output)
inline std::array<Grid2, 2> fluxE4(Chain const& c, double h)
{
    auto const& E  = c.E;
    auto const& Bt = c.Bt;
    auto const& Jt = c.Jt;
    Grid2 const* env[6] = {&c.b[0], &c.b[1], &c.b[2], &c.X[0], &c.X[1], &c.X[2]};
    auto ev = [&](auto const& tab, int i, int j) {
        double r = 0;
        for (auto const& row : tab)
            r += row.c * env[row.f1]->s(i, j, row.o1x, row.o1y)
                 * env[row.f2]->s(i, j, row.o2x, row.o2y);
        return r;
    };
    Grid2 om[3];
    om[0] = make(E[0], [&](int i, int j) {
        return E[0](i, j) * Jt[0](i, j)
               + (E[0](i, j) * D2(Jt[0], 0, i, j) - D2(E[0], 0, i, j) * Jt[0](i, j)) / 24;
    });
    om[1] = make(E[0], [&](int i, int j) {
        return E[1](i, j) * Jt[1](i, j)
               + (E[1](i, j) * D2(Jt[1], 1, i, j) - D2(E[1], 1, i, j) * Jt[1](i, j)) / 24;
    });
    om[2] = make(E[0], [&](int i, int j) { return E[2](i, j) * Jt[2](i, j); });
    auto pi1 = [&](auto const& tab, int i, int j) {
        double r = 0;
        for (auto const& row : tab)
            r += row.c * om[row.f].s(i, j, row.ox, row.oy);
        return r;
    };
    std::array<Grid2, 2> F;
    F[0] = make(E[0], [&](int i, int j) {
        double const S  = E[1](i, j) * Bt[2].s(i, j, -1, 0) - E[2](i, j) * Bt[1].s(i, j, -1, 0);
        double const Fo = h * ev(GK0, i, j) + S;
        double const pi = pi1(PI00, i, j) + pi1(PI10, i, j) + pi1(PI20, i, j);
        double const gd = E[0].s(i, j, -1, 0) * Jt[0](i, j) - E[0](i, j) * Jt[0].s(i, j, -1, 0);
        return Fo - h * (pi + gd / 24);
    });
    F[1] = make(E[0], [&](int i, int j) {
        double const S  = E[2](i, j) * Bt[0].s(i, j, 0, -1) - E[0](i, j) * Bt[2].s(i, j, 0, -1);
        double const Fo = h * ev(GK1, i, j) + S;
        double const pi = pi1(PI01, i, j) + pi1(PI11, i, j) + pi1(PI21, i, j);
        double const gd = E[1].s(i, j, 0, -1) * Jt[1](i, j) - E[1](i, j) * Jt[1].s(i, j, 0, -1);
        return Fo - h * (pi + gd / 24);
    });
    return F;
}

// reflection x -> (NX-1) - x along ax: primal p -> NX-1-p, dual d -> NX-2-d
inline Grid2 flip(Grid2 const& f, int ax, bool primal, double sign)
{
    int const n = ax == 0 ? f.nx : f.ny;
    return make(f, [&](int i, int j) {
        int const k  = ax == 0 ? i : j;
        int const km = primal ? n - 1 - k : n - 2 - k;
        return sign * (ax == 0 ? f.get(km, j) : f.get(i, km));
    });
}

// primal flags per axis
inline constexpr bool cenB[3][2] = {{true, false}, {false, true}, {false, false}};
inline constexpr bool cenE[3][2] = {{false, true}, {true, false}, {true, true}};
inline constexpr bool cenF[2][2] = {{true, false}, {false, true}};

inline std::array<Grid2, 2> mirrorAverage(Grid2 const* B, Grid2 const* E, double h)
{
    std::array<Grid2, 2> acc;
    int const axesList[4][2] = {{-1, -1}, {0, -1}, {1, -1}, {0, 1}};
    for (auto const& axes : axesList)
    {
        Grid2 Bm[3] = {B[0], B[1], B[2]};
        Grid2 Em[3] = {E[0], E[1], E[2]};
        for (int a : axes)
        {
            if (a < 0)
                continue;
            for (int k = 0; k < 3; ++k)
            {
                double const sB = (k == 1 - a || k == 2) ? -1. : 1.;
                double const sE = (k == a) ? -1. : 1.;
                Bm[k]           = flip(Bm[k], a, cenB[k][a], sB);
                Em[k]           = flip(Em[k], a, cenE[k][a], sE);
            }
        }
        auto Fm = fluxE4(chain(Bm, Em, h, false), h);
        for (int r = 1; r >= 0; --r)
        {
            int const a = axes[r];
            if (a < 0)
                continue;
            for (int d = 0; d < 2; ++d)
                Fm[d] = flip(Fm[d], a, cenF[d][a], d == a ? -1. : 1.);
        }
        for (int d = 0; d < 2; ++d)
        {
            if (acc[d].a.empty())
                acc[d] = Fm[d];
            else
                for (std::size_t n = 0; n < acc[d].a.size(); ++n)
                    acc[d].a[n] += Fm[d].a[n];
        }
    }
    for (auto& f : acc)
        for (auto& v : f.a)
            v /= 4;
    return acc;
}

// phi_h (phi_table.py) with node_val / node_der of q03_numeric_gates.py
inline Grid2 nodeVal(Grid2 const& f, bool const* dualAx, int skip)
{
    Grid2 out = f;
    for (int a = 0; a < 2; ++a)
        if (dualAx[a] && a != skip)
        {
            Grid2 const prev = out;
            out              = make(f, [&](int i, int j) {
                return 0.5 * (prev(i, j) + (a == 0 ? prev.get(i - 1, j) : prev.get(i, j - 1)));
            });
        }
    return out;
}
inline Grid2 nodeDer(Grid2 const& f, bool const* dualAx, int a)
{
    if (dualAx[a])
    {
        auto d = make(f, [&](int i, int j) {
            return f(i, j) - (a == 0 ? f.get(i - 1, j) : f.get(i, j - 1));
        });
        return nodeVal(d, dualAx, a);
    }
    auto v = nodeVal(f, dualAx, -1);
    return make(f, [&](int i, int j) {
        return a == 0 ? 0.5 * (v.get(i + 1, j) - v.get(i - 1, j))
                      : 0.5 * (v.get(i, j + 1) - v.get(i, j - 1));
    });
}

inline Grid2 phiH(Grid2 const* E, Grid2 const* B)
{
    // dual axes: E_c dual along c; B_c dual along the others
    bool const dE[3][2] = {{true, false}, {false, true}, {false, false}};
    bool const dB[3][2] = {{false, true}, {true, false}, {true, true}};
    struct T
    {
        double co;
        bool onE;
        int a, p, q;
    };
    T const terms[8] = {{1. / 24, true, 0, 0, 2},  {-1. / 24, false, 0, 0, 2},
                        {1. / 24, true, 1, 1, 2},  {-1. / 24, false, 1, 1, 2},
                        {-0.125, true, 0, 2, 0},   {-0.125, false, 0, 2, 0},
                        {-0.125, true, 1, 2, 1},   {-0.125, false, 1, 2, 1}};
    Grid2 phi = make(E[0], [](int, int) { return 0.; });
    for (auto const& t : terms)
    {
        auto f1 = t.onE ? nodeDer(E[t.p], dE[t.p], t.a) : nodeVal(E[t.p], dE[t.p], -1);
        auto f2 = t.onE ? nodeVal(B[t.q], dB[t.q], -1) : nodeDer(B[t.q], dB[t.q], t.a);
        for (std::size_t n = 0; n < phi.a.size(); ++n)
            phi.a[n] += t.co * f1.a[n] * f2.a[n];
    }
    return phi;
}

// F_pv = (1 + D2_n/24) F_sym - curl_h phi_h  (to be added to the point-value Etot fluxes)
inline std::array<Grid2, 2> pipelineFpv(Grid2 const* B, Grid2 const* E, double h)
{
    auto Fs  = mirrorAverage(B, E, h);
    auto ph  = phiH(E, B);
    auto Fpx = make(Fs[0], [&](int i, int j) {
        return Fs[0](i, j) + D2(Fs[0], 0, i, j) / 24 - (ph.s(i, j, 0, 1) - ph(i, j));
    });
    auto Fpy = make(Fs[1], [&](int i, int j) {
        return Fs[1](i, j) + D2(Fs[1], 1, i, j) / 24 + (ph.s(i, j, 1, 0) - ph(i, j));
    });
    return {Fpx, Fpy};
}


// ---------------------------------------------------------------------------------------------
// EXPERIMENT PHARE_O4_HALL_FP: O4 face-pair Hall EMF (O4/facepair/derivation.md S2; numpy mirror
// facepair_E in O4/facepair/sympy/s02d_numeric.py). M = K^T K (default) or N^T N (nn = true).
struct FacePairResult
{
    Grid2 Ex, Ey, Ez;
    Grid2 Jt[3];
    double pairWorkMax = 0, pairWorkScale = 0; // max |W_ab j^b_c - W_cb j^b_a| and max |W j|
};

inline FacePairResult facepairE(Grid2 const& Bx, Grid2 const& By, Grid2 const& Bz, Grid2 const& rho,
                                double h, bool nn)
{
    FacePairResult r;
    Grid2 Nb[3];
    Nb[0] = make(Bx, [&](int i, int j) { return Bx(i, j) - D2(Bx, 1, i, j) / 24; });
    Nb[1] = make(By, [&](int i, int j) { return By(i, j) - D2(By, 0, i, j) / 24; });
    Nb[2] = make(
        Bz, [&](int i, int j) { return Bz(i, j) - (D2(Bz, 0, i, j) + D2(Bz, 1, i, j)) / 24; });
    Grid2 Bt[3];
    if (!nn)
    {
        auto b0 = make(Bx, [&](int i, int j) { return st(Nb[0], 0, W4, i, j); });
        auto b1 = make(Bx, [&](int i, int j) { return st(Nb[1], 1, W4, i, j); });
        auto tx = make(Bx, [&](int i, int j) { return stT(b0, 0, W4, i, j); });
        auto ty = make(Bx, [&](int i, int j) { return stT(b1, 1, W4, i, j); });
        Bt[0]   = make(Bx, [&](int i, int j) { return tx(i, j) - D2(tx, 1, i, j) / 24; });
        Bt[1]   = make(Bx, [&](int i, int j) { return ty(i, j) - D2(ty, 0, i, j) / 24; });
        auto& b2 = Nb[2];
        Bt[2]    = make(Bx, [&](int i, int j) {
            return b2(i, j) - (D2(b2, 0, i, j) + D2(b2, 1, i, j)) / 24;
        });
    }
    else
    {
        Bt[0] = make(Bx, [&](int i, int j) { return Nb[0](i, j) - D2(Nb[0], 1, i, j) / 24; });
        Bt[1] = make(Bx, [&](int i, int j) { return Nb[1](i, j) - D2(Nb[1], 0, i, j) / 24; });
        Bt[2] = make(Bx, [&](int i, int j) {
            return Nb[2](i, j) - (D2(Nb[2], 0, i, j) + D2(Nb[2], 1, i, j)) / 24;
        });
    }
    auto& Jt = r.Jt;
    Jt[0] = make(Bx, [&](int i, int j) { return (Bt[2](i, j) - Bt[2].s(i, j, 0, -1)) / h; });
    Jt[1] = make(Bx, [&](int i, int j) { return -(Bt[2](i, j) - Bt[2].s(i, j, -1, 0)) / h; });
    Jt[2] = make(Bx, [&](int i, int j) {
        return ((Bt[1](i, j) - Bt[1].s(i, j, -1, 0)) - (Bt[0](i, j) - Bt[0].s(i, j, 0, -1))) / h;
    });
    auto avg = [&](Grid2 const& f, int c) {
        if (c >= 2)
            return f;
        return make(f, [&](int i, int j) { return f(i, j) + D2(f, c, i, j) / 24; });
    };
    auto P = [&](Grid2 const& f, int b, int c) {
        int const t = 3 - b - c;
        if (t >= 2)
            return avg(f, c);
        auto m = make(f, [&](int i, int j) { return st(f, t, W4D, i, j); });
        return avg(m, c);
    };
    auto PT = [&](Grid2 const& f, int b, int c) {
        int const t = 3 - b - c;
        auto a      = avg(f, c);
        if (t >= 2)
            return a;
        return make(f, [&](int i, int j) { return stT(a, t, W4D, i, j); });
    };
    auto rp = make(
        rho, [&](int i, int j) { return rho(i, j) - (D2(rho, 0, i, j) + D2(rho, 1, i, j)) / 24; });
    Grid2 rf[3];
    rf[0] = make(rp, [&](int i, int j) { return st(rp, 0, W4D, i, j); });
    rf[1] = make(rp, [&](int i, int j) { return st(rp, 1, W4D, i, j); });
    rf[2] = rp;
    Grid2 jf[3][3]; // jf[a][b] = j^b_a on face b
    Grid2 W[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            if (a != b)
            {
                jf[a][b] = PT(Jt[a], b, a);
                W[a][b]  = make(Bx, [&](int i, int j) {
                    return jf[a][b](i, j) * Nb[b](i, j) / rf[b](i, j);
                });
            }
    Grid2* E[3]       = {&r.Ex, &r.Ey, &r.Ez};
    int const cab[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    for (auto const& t : cab)
    {
        int const c = t[0], a = t[1], b = t[2];
        auto p1     = P(W[a][b], b, c);
        auto p2     = P(W[b][a], a, c);
        *E[c]       = make(Bx, [&](int i, int j) { return p1(i, j) - p2(i, j); });
    }
    // per-face pair work on face b: W_ab j^b_c - W_cb j^b_a (zero by construction, round-off)
    for (int b = 0; b < 3; ++b)
    {
        int const a = (b + 1) % 3, c = (b + 2) % 3;
        for (std::size_t n = 0; n < W[a][b].a.size(); ++n)
        {
            double const x = W[a][b].a[n] * jf[c][b].a[n], y = W[c][b].a[n] * jf[a][b].a[n];
            if (std::isfinite(x) && std::isfinite(y))
            {
                r.pairWorkMax   = std::max(r.pairWorkMax, std::abs(x - y));
                r.pairWorkScale = std::max(r.pairWorkScale, std::abs(x));
            }
        }
    }
    return r;
}

// EXPERIMENT PHARE_O4_P_NN: O4 face energy 1/2 sum_d P4_d[(N B_d)^2] (the squares interpolated)
inline bool pressureNN()
{
    static bool const v = [] {
        auto const* e = std::getenv("PHARE_O4_P_NN");
        return e && std::atoi(e) != 0;
    }();
    return v;
}
inline bool hallFacePair()
{
    static bool const v = [] {
        auto const* e = std::getenv("PHARE_O4_HALL_FP");
        return e && std::atoi(e) != 0;
    }();
    return v;
}

// the Hall EMF actually used: node zero-work (default) or face-pair (PHARE_O4_HALL_FP)
inline HallO4Result hallESelected(Grid2 const& Bx, Grid2 const& By, Grid2 const& Bz,
                                  Grid2 const& rho, double dx, double dy)
{
    if (!hallFacePair())
        return hallE(Bx, By, Bz, rho, dx, dy);
    if (std::abs(dx - dy) > 1e-12 * dx)
        throw std::runtime_error("O4 face-pair Hall EMF needs dx == dy");
    auto fp = facepairE(Bx, By, Bz, rho, dx, pressureNN());
    HallO4Result r;
    r.Ex            = std::move(fp.Ex);
    r.Ey            = std::move(fp.Ey);
    r.Ez            = std::move(fp.Ez);
    r.Jtx           = std::move(fp.Jt[0]);
    r.Jty           = std::move(fp.Jt[1]);
    r.Jtz           = std::move(fp.Jt[2]);
    r.pairWorkMax   = fp.pairWorkMax;
    r.pairWorkScale = fp.pairWorkScale;
    return r;
}

// EXPERIMENT PHARE_O4_HALL_EMF_SCALE=s: multiply the zero-work Hall EMF (0: no Hall EMF, control)
inline double hallEmfScale()
{
    static double const s = [] {
        auto const* e = std::getenv("PHARE_O4_HALL_EMF_SCALE");
        return e ? std::atof(e) : 1.0;
    }();
    return s;
}

inline bool envFlag(char const* name)
{
    auto const* s = std::getenv(name);
    return s && std::atoi(s) != 0;
}
inline int envInt(char const* name, int dflt)
{
    auto const* s = std::getenv(name);
    return s ? std::atoi(s) : dflt;
}

} // namespace PHARE::core::o4zw

#endif
