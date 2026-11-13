#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_EDGE_HALL_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_EDGE_HALL_HPP

#include <cstddef>
#include <limits>
#include <vector>

namespace PHARE::core
{
// experiment PHARE_HALL_EDGE=1 ("option B", 2D): E_H = (J x B)/rho formed directly on each Yee edge,
// J = raw Yee curl B, and J, B, rho brought to the edge by 2- or 4-point averages (zero_work_check.py
// E_naive). Array indices: Bx(i,j)@(i,j+1/2) By@(i+1/2,j) Bz,rho@(i+1/2,j+1/2) Ex@(i+1/2,j)
// Ey@(i,j+1/2) Ez@(i,j); Jx, Jy, Jz on the Ex, Ey, Ez edges. Field types only need operator()(i, j).
template<typename Field, typename RhoField>
struct EdgeHall2D
{
    using I = std::size_t;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy;

    double Jx(I i, I j) const { return (Bz(i, j) - Bz(i, j - 1)) * iy; }
    double Jy(I i, I j) const { return -(Bz(i, j) - Bz(i - 1, j)) * ix; }
    double Jz(I i, I j) const
    {
        return (By(i, j) - By(i - 1, j)) * ix - (Bx(i, j) - Bx(i, j - 1)) * iy;
    }
    double Ex(I i, I j) const
    {
        double const jy = 0.25 * (Jy(i, j) + Jy(i + 1, j) + Jy(i, j - 1) + Jy(i + 1, j - 1));
        double const jz = 0.5 * (Jz(i, j) + Jz(i + 1, j));
        double const bz = 0.5 * (Bz(i, j) + Bz(i, j - 1));
        double const by = By(i, j);
        double const r  = 0.5 * (rho(i, j) + rho(i, j - 1));
        return (jy * bz - jz * by) / r;
    }
    double Ey(I i, I j) const
    {
        double const jx = 0.25 * (Jx(i, j) + Jx(i - 1, j) + Jx(i, j + 1) + Jx(i - 1, j + 1));
        double const jz = 0.5 * (Jz(i, j) + Jz(i, j + 1));
        double const bz = 0.5 * (Bz(i, j) + Bz(i - 1, j));
        double const bx = Bx(i, j);
        double const r  = 0.5 * (rho(i, j) + rho(i - 1, j));
        return (jz * bx - jx * bz) / r;
    }
    double Ez(I i, I j) const
    {
        double const jx = 0.5 * (Jx(i, j) + Jx(i - 1, j));
        double const jy = 0.5 * (Jy(i, j) + Jy(i, j - 1));
        double const bx = 0.5 * (Bx(i, j) + Bx(i, j - 1));
        double const by = 0.5 * (By(i, j) + By(i - 1, j));
        double const r  = 0.25 * (rho(i, j) + rho(i - 1, j) + rho(i, j - 1) + rho(i - 1, j - 1));
        return (jx * by - jy * bx) / r;
    }
};

// experiment PHARE_HALL_FACEPAIR=1 (2D): port of igirault fix/hall-yee-edge-efield 622e2babb hall_:
// each pair J_a B_b / rho is formed on the face storing B_b (J_a and rho projected there) and
// averaged to the E edge; raw J = Yee curl B. Matches scripts/ivan.py E_ivan.
template<typename Field, typename RhoField>
struct FacePairHall2D
{
    using I = std::size_t;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy;

    double Jx(I i, I j) const { return (Bz(i, j) - Bz(i, j - 1)) * iy; }
    double Jy(I i, I j) const { return -(Bz(i, j) - Bz(i - 1, j)) * ix; }
    double Jz(I i, I j) const
    {
        return (By(i, j) - By(i - 1, j)) * ix - (Bx(i, j) - Bx(i, j - 1)) * iy;
    }
    double rY(I i, I j) const { return 0.5 * (rho(i, j) + rho(i, j - 1)); } // By face (i+1/2, j)
    double rX(I i, I j) const { return 0.5 * (rho(i, j) + rho(i - 1, j)); } // Bx face (i, j+1/2)
    // Bz face (cell centre): Jy Bz / rho and Jx Bz / rho
    double F1(I i, I j) const { return 0.5 * (Jy(i, j) + Jy(i + 1, j)) * Bz(i, j) / rho(i, j); }
    double F2(I i, I j) const { return 0.5 * (Jx(i, j) + Jx(i, j + 1)) * Bz(i, j) / rho(i, j); }
    // By face = Ex site: Jz By / rho and Jx By / rho
    double G1(I i, I j) const { return 0.5 * (Jz(i, j) + Jz(i + 1, j)) * By(i, j) / rY(i, j); }
    double G2(I i, I j) const { return Jx(i, j) * By(i, j) / rY(i, j); }
    // Bx face = Ey site: Jz Bx / rho and Jy Bx / rho
    double H1(I i, I j) const { return 0.5 * (Jz(i, j) + Jz(i, j + 1)) * Bx(i, j) / rX(i, j); }
    double H2(I i, I j) const { return Jy(i, j) * Bx(i, j) / rX(i, j); }

    double Ex(I i, I j) const { return 0.5 * (F1(i, j) + F1(i, j - 1)) - G1(i, j); }
    double Ey(I i, I j) const { return H1(i, j) - 0.5 * (F2(i, j) + F2(i - 1, j)); }
    double Ez(I i, I j) const
    {
        return 0.5 * (G2(i, j) + G2(i - 1, j)) - 0.5 * (H2(i, j) + H2(i, j - 1));
    }
};

// local reflection of an accessor about a face/cell: index i -> (flip ? c - i : i), value * sign
template<typename A>
struct ReflectedField
{
    using I = std::size_t;
    A const& a;
    double sign;
    bool fx, fy;
    I cx, cy;
    double operator()(I i, I j) const { return sign * a(fx ? cx - i : i, fy ? cy - j : j); }
};

// experiment PHARE_HALL_FLUX_FACE=1 (2D): exact local Hall energy flux for the face-pair EMF with the
// face energy 1/2 sum_d avg(B_d^2) (FHface/derivation.md, Consolidation): one-sided
//   Fx = 1/2 dx iy Bx (Ez(j+1) - Ez) + Ey Bz(i-1) - Ez By(i-1) + 1/2 dx (Jy Wxz(i-1) - Jz Wxy(i-1))
//   Fy = -1/2 dy ix By (Ez(i+1) - Ez) + Ez Bx(j-1) - Ex Bz(j-1) + 1/2 dy (Jz Wyx(j-1) - Jx Wyz(j-1))
// (Wxz = F2, Wyz = F1, Wxy = G2, Wyx = H2 of FacePairHall2D), then averaged over the 4 grid
// reflections about the face (B a pseudovector; the normal flux component flips), which keeps
// Q + div F = 0 and gives 2nd-order face values for refluxing.
template<typename Field, typename RhoField>
struct FacePairFluxSym
{
    using I  = std::size_t;
    using RF = ReflectedField<Field>;
    using RR = ReflectedField<RhoField>;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy;

    static double Fx1(auto const& f, double ix, double iy, I i, I j)
    {
        double const dx = 1.0 / ix;
        return 0.5 * dx * iy * f.Bx(i, j) * (f.Ez(i, j + 1) - f.Ez(i, j))
               + f.Ey(i, j) * f.Bz(i - 1, j) - f.Ez(i, j) * f.By(i - 1, j)
               + 0.5 * dx * (f.Jy(i, j) * f.F2(i - 1, j) - f.Jz(i, j) * f.G2(i - 1, j));
    }
    static double Fy1(auto const& f, double ix, double iy, I i, I j)
    {
        double const dy = 1.0 / iy;
        return -0.5 * dy * ix * f.By(i, j) * (f.Ez(i + 1, j) - f.Ez(i, j))
               + f.Ez(i, j) * f.Bx(i, j - 1) - f.Ex(i, j) * f.Bz(i, j - 1)
               + 0.5 * dy * (f.Jz(i, j) * f.H2(i, j - 1) - f.Jx(i, j) * f.F1(i, j - 1));
    }
    // frame (rx, ry): reflect about the face. dir 0 = x-face (i, j+1/2), 1 = y-face (i+1/2, j)
    template<int dir>
    double frame_(bool rx, bool ry, I i, I j) const
    {
        // centres: x-face: x primal p -> 2i - p, x dual d -> 2i - 1 - d; y dual d -> 2j - d, y primal -> 2j + 1 - p
        //          y-face: x dual d -> 2i - d, x primal -> 2i + 1 - p; y primal p -> 2j - p, y dual -> 2j - 1 - d
        I const cxp = dir == 0 ? 2 * i : 2 * i + 1, cxd = dir == 0 ? 2 * i - 1 : 2 * i;
        I const cyp = dir == 0 ? 2 * j + 1 : 2 * j, cyd = dir == 0 ? 2 * j : 2 * j - 1;
        // pseudovector signs: reflecting x flips By, Bz; reflecting y flips Bx, Bz
        double const sx = ry ? -1. : 1., sy = rx ? -1. : 1., sz = (rx != ry) ? -1. : 1.;
        RF bx{Bx, sx, rx, ry, cxp, cyd}, by{By, sy, rx, ry, cxd, cyp}, bz{Bz, sz, rx, ry, cxd, cyd};
        RR r{rho, 1., rx, ry, cxd, cyd};
        FacePairHall2D<RF, RR> const f{bx, by, bz, r, ix, iy};
        if constexpr (dir == 0)
            return (rx ? -1. : 1.) * Fx1(f, ix, iy, i, j);
        else
            return (ry ? -1. : 1.) * Fy1(f, ix, iy, i, j);
    }
    double Fx(I i, I j) const
    {
        return 0.25 * (frame_<0>(false, false, i, j) + frame_<0>(true, false, i, j)
                       + frame_<0>(false, true, i, j) + frame_<0>(true, true, i, j));
    }
    double Fy(I i, I j) const
    {
        return 0.25 * (frame_<1>(false, false, i, j) + frame_<1>(true, false, i, j)
                       + frame_<1>(false, true, i, j) + frame_<1>(true, true, i, j));
    }
    double Fx_onesided(I i, I j) const { return frame_<0>(false, false, i, j); }
    double Fy_onesided(I i, I j) const { return frame_<1>(false, false, i, j); }
};

// experiment PHARE_HALL_FACEPAIR=1 (3D): the 3D face-pair Hall field. Pair field W_ab = J_a
// projected on the B_b face, times B_b / rho_b (2-point face density); E_c = P_{B_b->E_c}(W_ab) -
// P_{B_a->E_c}(W_ba), (c, a, b) cyclic; raw J = Yee curl B. Bx(i,j,k)@(i,j+1/2,k+1/2)
// By@(i+1/2,j,k+1/2) Bz@(i+1/2,j+1/2,k) rho@(i+1/2,j+1/2,k+1/2) Ex@(i+1/2,j,k) Ey@(i,j+1/2,k)
// Ez@(i,j,k+1/2). Reduces to FacePairHall2D when nothing depends on z. Matches facepair3d/ref3d.py
// E_hall.
template<typename Field, typename RhoField>
struct FacePairHall3D
{
    using I = std::size_t;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy, iz;

    double Jx(I i, I j, I k) const
    {
        return (Bz(i, j, k) - Bz(i, j - 1, k)) * iy - (By(i, j, k) - By(i, j, k - 1)) * iz;
    }
    double Jy(I i, I j, I k) const
    {
        return (Bx(i, j, k) - Bx(i, j, k - 1)) * iz - (Bz(i, j, k) - Bz(i - 1, j, k)) * ix;
    }
    double Jz(I i, I j, I k) const
    {
        return (By(i, j, k) - By(i - 1, j, k)) * ix - (Bx(i, j, k) - Bx(i, j - 1, k)) * iy;
    }
    double rX(I i, I j, I k) const { return 0.5 * (rho(i, j, k) + rho(i - 1, j, k)); }
    double rY(I i, I j, I k) const { return 0.5 * (rho(i, j, k) + rho(i, j - 1, k)); }
    double rZ(I i, I j, I k) const { return 0.5 * (rho(i, j, k) + rho(i, j, k - 1)); }
    // W_ab: J_a on the B_b face
    double Wyx(I i, I j, I k) const
    {
        return 0.5 * (Jy(i, j, k) + Jy(i, j, k + 1)) * Bx(i, j, k) / rX(i, j, k);
    }
    double Wzx(I i, I j, I k) const
    {
        return 0.5 * (Jz(i, j, k) + Jz(i, j + 1, k)) * Bx(i, j, k) / rX(i, j, k);
    }
    double Wxy(I i, I j, I k) const
    {
        return 0.5 * (Jx(i, j, k) + Jx(i, j, k + 1)) * By(i, j, k) / rY(i, j, k);
    }
    double Wzy(I i, I j, I k) const
    {
        return 0.5 * (Jz(i, j, k) + Jz(i + 1, j, k)) * By(i, j, k) / rY(i, j, k);
    }
    double Wxz(I i, I j, I k) const
    {
        return 0.5 * (Jx(i, j, k) + Jx(i, j + 1, k)) * Bz(i, j, k) / rZ(i, j, k);
    }
    double Wyz(I i, I j, I k) const
    {
        return 0.5 * (Jy(i, j, k) + Jy(i + 1, j, k)) * Bz(i, j, k) / rZ(i, j, k);
    }

    double Ex(I i, I j, I k) const
    {
        return 0.5 * (Wyz(i, j, k) + Wyz(i, j - 1, k)) - 0.5 * (Wzy(i, j, k) + Wzy(i, j, k - 1));
    }
    double Ey(I i, I j, I k) const
    {
        return 0.5 * (Wzx(i, j, k) + Wzx(i, j, k - 1)) - 0.5 * (Wxz(i, j, k) + Wxz(i - 1, j, k));
    }
    double Ez(I i, I j, I k) const
    {
        return 0.5 * (Wxy(i, j, k) + Wxy(i - 1, j, k)) - 0.5 * (Wyx(i, j, k) + Wyx(i, j - 1, k));
    }
};

// 3D local reflection: index -> (flip ? c - index : index) per axis, value * sign
template<typename A>
struct ReflectedField3D
{
    using I = std::size_t;
    A const& a;
    double sign;
    bool fx, fy, fz;
    I cx, cy, cz;
    double operator()(I i, I j, I k) const
    {
        return sign * a(fx ? cx - i : i, fy ? cy - j : j, fz ? cz - k : k);
    }
};

// per-patch cache of the face-pair field (raw J on edges, the 6 pair fields W on faces, E on edges), on
// the cell-centred array extent n_a = rho.shape()[a]. Valid ranges: J [1, n), W [1, n - 1), E [2, n - 1)
// per axis; NaN elsewhere, so a read outside them shows.
struct Array3
{
    using I = std::size_t;
    I n1 = 0, n2 = 0;
    std::vector<double> v;
    double operator()(I i, I j, I k) const { return v[(i * n1 + j) * n2 + k]; }
    double& at(I i, I j, I k) { return v[(i * n1 + j) * n2 + k]; }
};

struct FacePairCache3D
{
    using I = std::size_t;
    Array3 Jx, Jy, Jz, Wyx, Wzx, Wxy, Wzy, Wxz, Wyz, Ex, Ey, Ez;

    template<typename H>
    void build(H const& h, I n0, I n1, I n2)
    {
        double const nan = std::numeric_limits<double>::quiet_NaN();
        for (Array3* a : {&Jx, &Jy, &Jz, &Wyx, &Wzx, &Wxy, &Wzy, &Wxz, &Wyz, &Ex, &Ey, &Ez})
        {
            a->n1 = n1;
            a->n2 = n2;
            a->v.assign(n0 * n1 * n2, nan);
        }
        auto loop = [&](I lo, I hi0, I hi1, I hi2, auto&& f) {
            for (I i = lo; i < hi0; ++i)
                for (I j = lo; j < hi1; ++j)
                    for (I k = lo; k < hi2; ++k)
                        f(i, j, k);
        };
        loop(1, n0, n1, n2, [&](I i, I j, I k) {
            Jx.at(i, j, k) = h.Jx(i, j, k);
            Jy.at(i, j, k) = h.Jy(i, j, k);
            Jz.at(i, j, k) = h.Jz(i, j, k);
        });
        // W from the cached J (same arithmetic as FacePairHall3D::W**)
        loop(1, n0 - 1, n1 - 1, n2 - 1, [&](I i, I j, I k) {
            Wyx.at(i, j, k) = 0.5 * (Jy(i, j, k) + Jy(i, j, k + 1)) * h.Bx(i, j, k) / h.rX(i, j, k);
            Wzx.at(i, j, k) = 0.5 * (Jz(i, j, k) + Jz(i, j + 1, k)) * h.Bx(i, j, k) / h.rX(i, j, k);
            Wxy.at(i, j, k) = 0.5 * (Jx(i, j, k) + Jx(i, j, k + 1)) * h.By(i, j, k) / h.rY(i, j, k);
            Wzy.at(i, j, k) = 0.5 * (Jz(i, j, k) + Jz(i + 1, j, k)) * h.By(i, j, k) / h.rY(i, j, k);
            Wxz.at(i, j, k) = 0.5 * (Jx(i, j, k) + Jx(i, j + 1, k)) * h.Bz(i, j, k) / h.rZ(i, j, k);
            Wyz.at(i, j, k) = 0.5 * (Jy(i, j, k) + Jy(i + 1, j, k)) * h.Bz(i, j, k) / h.rZ(i, j, k);
        });
        loop(2, n0 - 1, n1 - 1, n2 - 1, [&](I i, I j, I k) {
            Ex.at(i, j, k) = 0.5 * (Wyz(i, j, k) + Wyz(i, j - 1, k)) - 0.5 * (Wzy(i, j, k) + Wzy(i, j, k - 1));
            Ey.at(i, j, k) = 0.5 * (Wzx(i, j, k) + Wzx(i, j, k - 1)) - 0.5 * (Wxz(i, j, k) + Wxz(i - 1, j, k));
            Ez.at(i, j, k) = 0.5 * (Wxy(i, j, k) + Wxy(i - 1, j, k)) - 0.5 * (Wyx(i, j, k) + Wyx(i, j - 1, k));
        });
    }
};

// experiment PHARE_HALL_FLUX_FACE=1 (3D): exact local Hall energy flux of FacePairHall3D against
// the face energy (FHface/derivation.md S1; 3D H tables of s01_face_flux.H_tables(3)), cyclic (d,
// a, b):
//   F_d = h_d (1/2 B_d (curl E)_d + 1/2 J_a W_db(-e_d) - 1/2 J_b W_da(-e_d)) + E_a B_b(-e_d) - E_b
//   B_a(-e_d),
// then averaged over the 8 grid reflections about the face (B a pseudovector, the normal component
// of F flips). All 8 are needed for 2nd-order face values; a 2D-like subset of 4 gives order ~1.2
// (facepair3d/check3d.py).
template<typename Field, typename RhoField>
struct FacePairFluxSym3D
{
    using I  = std::size_t;
    using RF = ReflectedField3D<Field>;
    using RR = ReflectedField3D<RhoField>;
    Field const& Bx;
    Field const& By;
    Field const& Bz;
    RhoField const& rho;
    double ix, iy, iz;
    FacePairCache3D const* cache = nullptr;

    FacePairHall3D<Field, RhoField> hall() const { return {Bx, By, Bz, rho, ix, iy, iz}; }

    // a reflected frame read from the cache: J, E polar (component c flips iff axis c is reflected),
    // B pseudo (component b flips for each reflected axis other than b), W_ab = J_a B_b / rho
    using RA = ReflectedField3D<Array3>;
    struct CachedView
    {
        RF Bx, By, Bz;
        RA Jx, Jy, Jz, Wyx, Wzx, Wxy, Wzy, Wxz, Wyz, Ex, Ey, Ez;
        double ix, iy, iz;
    };

    template<int dir>
    static double F1(auto const& f, I i, I j, I k)
    {
        if constexpr (dir == 0)
        {
            double const X = (f.Ez(i, j + 1, k) - f.Ez(i, j, k)) * f.iy
                             - (f.Ey(i, j, k + 1) - f.Ey(i, j, k)) * f.iz;
            return (0.5 * f.Bx(i, j, k) * X + 0.5 * f.Jy(i, j, k) * f.Wxz(i - 1, j, k)
                    - 0.5 * f.Jz(i, j, k) * f.Wxy(i - 1, j, k))
                       / f.ix
                   + f.Ey(i, j, k) * f.Bz(i - 1, j, k) - f.Ez(i, j, k) * f.By(i - 1, j, k);
        }
        else if constexpr (dir == 1)
        {
            double const X = (f.Ex(i, j, k + 1) - f.Ex(i, j, k)) * f.iz
                             - (f.Ez(i + 1, j, k) - f.Ez(i, j, k)) * f.ix;
            return (0.5 * f.By(i, j, k) * X + 0.5 * f.Jz(i, j, k) * f.Wyx(i, j - 1, k)
                    - 0.5 * f.Jx(i, j, k) * f.Wyz(i, j - 1, k))
                       / f.iy
                   + f.Ez(i, j, k) * f.Bx(i, j - 1, k) - f.Ex(i, j, k) * f.Bz(i, j - 1, k);
        }
        else
        {
            double const X = (f.Ey(i + 1, j, k) - f.Ey(i, j, k)) * f.ix
                             - (f.Ex(i, j + 1, k) - f.Ex(i, j, k)) * f.iy;
            return (0.5 * f.Bz(i, j, k) * X + 0.5 * f.Jx(i, j, k) * f.Wzy(i, j, k - 1)
                    - 0.5 * f.Jy(i, j, k) * f.Wzx(i, j, k - 1))
                       / f.iz
                   + f.Ex(i, j, k) * f.By(i, j, k - 1) - f.Ey(i, j, k) * f.Bx(i, j, k - 1);
        }
    }
    // frame (rx, ry, rz) reflected about the dir-face of index (i, j, k). Reflection centres: along
    // the normal axis primal p -> 2n - p, dual d -> 2n - 1 - d; along a tangential axis dual d ->
    // 2t - d, primal p -> 2t + 1 - p. Reflecting axis a flips the B components other than a.
    template<int dir>
    double frame_(bool rx, bool ry, bool rz, I i, I j, I k) const
    {
        I const id[3] = {i, j, k};
        I cp[3], cd[3];
        for (int a = 0; a < 3; ++a)
        {
            cp[a] = a == dir ? 2 * id[a] : 2 * id[a] + 1;
            cd[a] = a == dir ? 2 * id[a] - 1 : 2 * id[a];
        }
        double const sx = (ry != rz) ? -1. : 1., sy = (rx != rz) ? -1. : 1.,
                     sz = (rx != ry) ? -1. : 1.;
        RF const bx{Bx, sx, rx, ry, rz, cp[0], cd[1], cd[2]};
        RF const by{By, sy, rx, ry, rz, cd[0], cp[1], cd[2]};
        RF const bz{Bz, sz, rx, ry, rz, cd[0], cd[1], cp[2]};
        RR const r{rho, 1., rx, ry, rz, cd[0], cd[1], cd[2]};
        FacePairHall3D<RF, RR> const f{bx, by, bz, r, ix, iy, iz};
        bool const rn = dir == 0 ? rx : dir == 1 ? ry : rz;
        return (rn ? -1. : 1.) * F1<dir>(f, i, j, k);
    }
    template<int dir>
    double frameCached_(bool rx, bool ry, bool rz, I i, I j, I k) const
    {
        I const id[3]  = {i, j, k};
        bool const r[3] = {rx, ry, rz};
        I cp[3], cd[3];
        for (int a = 0; a < 3; ++a)
        {
            cp[a] = a == dir ? 2 * id[a] : 2 * id[a] + 1;
            cd[a] = a == dir ? 2 * id[a] - 1 : 2 * id[a];
        }
        auto sB = [&](int b) {
            double s = 1;
            for (int a = 0; a < 3; ++a)
                if (a != b && r[a])
                    s = -s;
            return s;
        };
        auto sJ = [&](int c) { return r[c] ? -1. : 1.; };
        // centring: polar (J, E) of component c dual along c; face quantities of component b primal along b
        auto cE = [&](Array3 const& a, int c, double s) {
            return RA{a, s, rx, ry, rz, c == 0 ? cd[0] : cp[0], c == 1 ? cd[1] : cp[1], c == 2 ? cd[2] : cp[2]};
        };
        auto cF = [&](Array3 const& a, int b, double s) {
            return RA{a, s, rx, ry, rz, b == 0 ? cp[0] : cd[0], b == 1 ? cp[1] : cd[1], b == 2 ? cp[2] : cd[2]};
        };
        auto const& C = *cache;
        CachedView const f{RF{Bx, sB(0), rx, ry, rz, cp[0], cd[1], cd[2]},
                           RF{By, sB(1), rx, ry, rz, cd[0], cp[1], cd[2]},
                           RF{Bz, sB(2), rx, ry, rz, cd[0], cd[1], cp[2]},
                           cE(C.Jx, 0, sJ(0)),
                           cE(C.Jy, 1, sJ(1)),
                           cE(C.Jz, 2, sJ(2)),
                           cF(C.Wyx, 0, sJ(1) * sB(0)),
                           cF(C.Wzx, 0, sJ(2) * sB(0)),
                           cF(C.Wxy, 1, sJ(0) * sB(1)),
                           cF(C.Wzy, 1, sJ(2) * sB(1)),
                           cF(C.Wxz, 2, sJ(0) * sB(2)),
                           cF(C.Wyz, 2, sJ(1) * sB(2)),
                           cE(C.Ex, 0, sJ(0)),
                           cE(C.Ey, 1, sJ(1)),
                           cE(C.Ez, 2, sJ(2)),
                           ix,
                           iy,
                           iz};
        bool const rn = dir == 0 ? rx : dir == 1 ? ry : rz;
        return (rn ? -1. : 1.) * F1<dir>(f, i, j, k);
    }
    template<int dir>
    double F(I i, I j, I k) const
    {
        double s = 0;
        for (int m = 0; m < 8; ++m)
            s += cache ? frameCached_<dir>(m & 1, m & 2, m & 4, i, j, k)
                       : frame_<dir>(m & 1, m & 2, m & 4, i, j, k);
        return 0.125 * s;
    }
    double Fx(I i, I j, I k) const { return F<0>(i, j, k); }
    double Fy(I i, I j, I k) const { return F<1>(i, j, k); }
    double Fz(I i, I j, I k) const { return F<2>(i, j, k); }
    double Fx_onesided(I i, I j, I k) const { return frame_<0>(false, false, false, i, j, k); }
    double Fy_onesided(I i, I j, I k) const { return frame_<1>(false, false, false, i, j, k); }
    double Fz_onesided(I i, I j, I k) const { return frame_<2>(false, false, false, i, j, k); }
};
} // namespace PHARE::core

#endif
