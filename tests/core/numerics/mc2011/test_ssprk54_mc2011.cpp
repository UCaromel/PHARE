/*
  Tests of the SSPRK(5,4) temporal coarse-fine reconstruction kernels
  (core/numerics/mc2011/mc2011_reconstruction.hpp), the temporal half of fourth-order AMR
  ghost fills in the style of McCorquodale and Colella (2011).

  Part 1 is a pure-ODE check on the Brusselator (nonlinear, two components), with errors measured
  locally over one coarse step from a reference state:

  1. The continuous-extension and split coefficients of the SSPRK54 struct solve their
     defining algebraic systems.
  2. Fine-stage ghost reconstruction at substep start chi, dt_f = dt/2:
        U^(i) = ytilde(chi) + dt_f c_i u'(chi) + dt_f^2 (Ac)_i u''(chi)
                + dt_f^3 [ (A c^2)_i / 2 * f''f^2 + (A A c)_i * (f')^2 f ]
     with f''f^2 = u''' - (f')^2 f, u''' = (6/dt^2) sum_i beta3_i k_i, and the split
     (f')^2 f = sum_i w_i k_i / dt^2 (PHARE stores k_i = f(Y_i), hence /dt^2 and not /dt^3).
     The local error must converge with slope 4 for stages 2..5 and chi in {0, 1/2}.

  Part 2 tests the kernels themselves: the coefficient tables against the tableau they come
  from, the dense-output reconstruction, the split terms, and the stage back-solve used to
  recover the stage derivatives from the stored stage states.

  References:
   - Spiteri, Ruuth, "A new class of optimal high-order strong-stability-preserving time
     discretization methods", SIAM J. Numer. Anal. 40 (2002) 469-491 (SSPRK(5,4) tableau)
   - McCorquodale, Colella, "A high-order finite-volume method for conservation laws on
     locally refined grids", Commun. Appl. Math. Comput. Sci. 6 (2011) 1-25
   - Gottlieb, Ketcheson, Shu, "High order strong stability preserving time discretizations",
     J. Sci. Comput. 38 (2009) 251-289 (Shu-Osher <-> Butcher conversion)
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "phare_core.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/faraday/faraday.hpp"
#include "core/numerics/finite_volume_euler/finite_volume_euler_per_field.hpp"
#include "core/numerics/mc2011/mc2011_reconstruction.hpp"
#include "core/utilities/point/point.hpp"

#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

using namespace PHARE::core;

namespace
{

using SSPRK54 = mc2011::SSPRK54;

using Vec = std::array<double, 2>;

Vec operator+(Vec const& a, Vec const& b)
{
    return {a[0] + b[0], a[1] + b[1]};
}
Vec operator-(Vec const& a, Vec const& b)
{
    return {a[0] - b[0], a[1] - b[1]};
}
Vec operator*(double s, Vec const& a)
{
    return {s * a[0], s * a[1]};
}
double maxAbs(Vec const& a)
{
    return std::max(std::abs(a[0]), std::abs(a[1]));
}


// Butcher form of the Shu-Osher tableau, rebuilt independently from the w coefficients, to
// check the tabulated SSPRK54 quantities against the tableau they derive from.
struct Tableau
{
    double A[5][5]{}, b[5]{}, c[5]{};
    double Ac[5]{}, Ac2[5]{}, AAc[5]{}; // A*c, A*c^2, A*(A*c)

    Tableau()
    {
        A[1][0] = SSPRK54::w0;
        for (int j = 0; j < 5; ++j)
            A[2][j] = SSPRK54::w11 * A[1][j];
        A[2][1] += SSPRK54::w12;
        for (int j = 0; j < 5; ++j)
            A[3][j] = SSPRK54::w21 * A[2][j];
        A[3][2] += SSPRK54::w22;
        for (int j = 0; j < 5; ++j)
            A[4][j] = SSPRK54::w31 * A[3][j];
        A[4][3] += SSPRK54::w32;
        for (int j = 0; j < 5; ++j)
            b[j] = SSPRK54::w40 * A[2][j] + SSPRK54::w41 * A[3][j] + SSPRK54::w43 * A[4][j];
        b[3] += SSPRK54::w42;
        b[4] += SSPRK54::w44;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                c[i] += A[i][j];
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
            {
                Ac[i] += A[i][j] * c[j];
                Ac2[i] += A[i][j] * c[j] * c[j];
            }
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                AAc[i] += A[i][j] * Ac[j];
    }
};

Tableau const rk{};


// Tabulated decompositions of SSPRK54, read from the kernel header:
//   beta1, beta2, beta3: continuous-extension weights (beta3 is defined as b - beta1 - beta2),
//   wsplit: weights of the split (f')^2 f = sum_i wsplit_i k_i / dt^2, the minimum-norm solution
//   of M w = e4 with M = [1; c; c^2; Ac] (4x5, rank 4).
constexpr auto& beta1  = SSPRK54::beta1;
constexpr auto& beta2  = SSPRK54::beta2;
constexpr auto& wsplit = SSPRK54::wsplit;

struct CE
{
    double b1[5], b2[5], b3[5]; // beta3 := b - beta1 - beta2, so the endpoint is consistent

    CE()
    {
        for (int i = 0; i < 5; ++i)
        {
            b1[i] = beta1[i];
            b2[i] = beta2[i];
            b3[i] = rk.b[i] - beta1[i] - beta2[i];
        }
    }
};

CE const ce{};


// Brusselator: f0 = A + y0^2 y1 - (B+1) y0, f1 = B y0 - y0^2 y1
constexpr double Abr = 1.0, Bbr = 3.0;

Vec f(Vec const& y)
{
    return {Abr + y[0] * y[0] * y[1] - (Bbr + 1) * y[0], Bbr * y[0] - y[0] * y[0] * y[1]};
}
Vec const yInit{1.2, 2.9};


struct StepData
{
    std::array<Vec, 5> Y, k;
    Vec ynew;
};

StepData stages(Vec y, double dt)
{
    StepData s;
    for (int i = 0; i < 5; ++i)
    {
        s.Y[i] = y;
        for (int j = 0; j < i; ++j)
            s.Y[i] = s.Y[i] + (dt * rk.A[i][j]) * s.k[j];
        s.k[i] = f(s.Y[i]);
    }
    s.ynew = y;
    for (int j = 0; j < 5; ++j)
        s.ynew = s.ynew + (dt * rk.b[j]) * s.k[j];
    return s;
}

Vec refSolve(Vec y, double T, int nsub = 4096)
{
    if (T == 0.0)
        return y;
    double const h = T / nsub;
    for (int i = 0; i < nsub; ++i)
        y = stages(y, h).ynew;
    return y;
}

Vec weighted(double const (&w)[5], std::array<Vec, 5> const& k)
{
    Vec r{0, 0};
    for (int i = 0; i < 5; ++i)
        r = r + w[i] * k[i];
    return r;
}


// average of the last `n` successive log2 error ratios
double tailSlope(std::vector<double> const& errs, int n = 3)
{
    double s     = 0;
    int const sz = static_cast<int>(errs.size());
    for (int i = sz - n; i < sz; ++i)
        s += std::log2(errs[i - 1] / errs[i]);
    return s / n;
}

} // namespace


TEST(SSPRK54Tableau, ceAndSplitCoefficientsSolveTheirDefiningSystems)
{
    // M x = r for M = [1; c; c^2; Ac] and
    //   beta1 -> (1,0,0,0), beta2 -> (0,1/2,0,0), beta3 -> (0,0,1/3,1/6), w -> (0,0,0,1)
    auto Mdot = [&](double const(&x)[5]) {
        std::array<double, 4> r{};
        for (int i = 0; i < 5; ++i)
        {
            r[0] += x[i];
            r[1] += x[i] * rk.c[i];
            r[2] += x[i] * rk.c[i] * rk.c[i];
            r[3] += x[i] * rk.Ac[i];
        }
        return r;
    };
    double constexpr tol = 1e-13;
    auto const r1 = Mdot(ce.b1), r2 = Mdot(ce.b2), r3 = Mdot(ce.b3), rw = Mdot(wsplit);
    std::array<double, 4> const t1{1, 0, 0, 0}, t2{0, 1.0 / 2, 0, 0}, t3{0, 0, 1.0 / 3, 1.0 / 6},
        tw{0, 0, 0, 1};
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(r1[i], t1[i], tol);
        EXPECT_NEAR(r2[i], t2[i], tol);
        EXPECT_NEAR(r3[i], t3[i], tol); // holds to truncation because beta3 = b - beta1 - beta2
        EXPECT_NEAR(rw[i], tw[i], tol);
    }
}




TEST(StageReconstruction, fineStageGhostsConvergeAtOrder4)
{
    // Full pipeline: coarse-step k's only -> CE + derivatives + split -> fine stages
    // at substep start chi, dt_f = dt/2 (ratio-2 subcycling), vs true fine stages
    // started from the reference solution at chi.
    for (double const chi : {0.0, 0.5})
    {
        std::vector<double> errs;
        for (int p = 2; p <= 7; ++p)
        {
            double const dt = std::pow(2.0, -p), dtf = dt / 2;
            auto const s    = stages(yInit, dt);
            Vec const uchi  = refSolve(yInit, chi * dt);
            auto const fine = stages(uchi, dtf);

            double bt[5], btp[5], btpp[5];
            for (int i = 0; i < 5; ++i)
            {
                bt[i]   = ce.b1[i] * chi + ce.b2[i] * chi * chi + ce.b3[i] * chi * chi * chi;
                btp[i]  = ce.b1[i] + 2 * ce.b2[i] * chi + 3 * ce.b3[i] * chi * chi;
                btpp[i] = 2 * ce.b2[i] + 6 * ce.b3[i] * chi;
            }
            Vec const u    = yInit + dt * weighted(bt, s.k);
            Vec const up   = weighted(btp, s.k);
            Vec const upp  = (1 / dt) * weighted(btpp, s.k);
            Vec const uppp = (6 / (dt * dt)) * weighted(ce.b3, s.k);
            Vec const fpfp = (1 / (dt * dt)) * weighted(wsplit, s.k);
            Vec const ffpp = uppp - fpfp;

            double err = 0;
            for (int i = 1; i < 5; ++i) // stage 1 is uchi itself
            {
                Vec const rec = u + (dtf * rk.c[i]) * up + (dtf * dtf * rk.Ac[i]) * upp
                                + (dtf * dtf * dtf) * (0.5 * rk.Ac2[i] * ffpp + rk.AAc[i] * fpfp);
                err = std::max(err, maxAbs(rec - fine.Y[i]));
            }
            errs.push_back(err);
        }
        EXPECT_NEAR(tailSlope(errs), 4.0, 0.2) << "chi = " << chi;
    }
}


namespace
{

// A coherent pure-MHD profile purely to name the 2D Yee-MHD GridLayout_t type below.
// {2, 1} (dimension, interp_order) is a *Hybrid* profile (reconstruction_type stays
// MHDOff), so PHARE_Types<...>::MHD is an incoherent-MHD-axes type there; naming
// MHD::GridLayout_t off it hits the deliberate absent-MHDOff-specialization canary
// (ReconstructionNghosts<MHDOff> is intentionally never defined). interp_order=0 keeps
// this profile pure MHD; the reconstruction/timestepper/order values below are
// otherwise arbitrary valid choices (these tests build GridLayout directly from
// dl/cells/origin, not from this profile).
constexpr PHARE::SimOpts kMHD2DOpts{2,
                                    0,
                                    0,
                                    PHARE::MHDOpts::MHDOrder::O2,
                                    PHARE::MHDOpts::ReconstructionType::WENOZ,
                                    PHARE::MHDOpts::SlopeLimiterType::None,
                                    PHARE::MHDOpts::RiemannSolverType::Rusanov};

// Smooth, index-dependent fill so k = -divF is nonzero and spatially varying.
template<typename... Idx>
double fval(double const a, double const fx, double const fy, Idx const&... ij)
{
    auto const idx = std::array<double, sizeof...(ij)>{static_cast<double>(ij)...};
    return a + fx * std::sin(0.3 * idx[0]) + fy * std::cos(0.2 * idx[1]);
}

} // namespace




TEST(SSPRK54Coefficients, butcherWeightsAndAbscissaeMatchTableau)
{
    // b and c in SSPRK54 are Shu-Osher collapses written out by hand; the
    // Tableau rebuilds them through the full A-matrix recursion.
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(SSPRK54::b[i], rk.b[i], 1e-13) << "b[" << i << "]";
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(SSPRK54::c[i], rk.c[i + 1], 1e-13) << "c[" << i << "]";
}


TEST(SSPRK54Coefficients, betaSumsReproduceButcherWeights)
{
    // beta3 := b - beta1 - beta2 by construction, so the CE endpoint weights
    // b_i(1) = beta1 + beta2 + beta3 recover b to rounding.
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(SSPRK54::beta1[i] + SSPRK54::beta2[i] + SSPRK54::beta3[i], SSPRK54::b[i], 1e-15)
            << "i = " << i;
}


TEST(SSPRK54Coefficients, gammaTablesMatchTableau)
{
    // gamma1=c_i, gamma2=(Ac)_i, gamma3=(Ac^2)_i/2, gamma4=(AAc)_i for the four
    // intermediate stages (SSPRK54 index s = Butcher index s+1). The stage rows
    // are constexpr Shu-Osher recursions, within ~1 ULP of the tableau
    // quantities, hence the 1e-15 tolerance.
    double constexpr tol = 1e-15;
    for (int s = 0; s < 4; ++s)
    {
        EXPECT_NEAR(SSPRK54::gamma1[s], rk.c[s + 1], tol) << "gamma1[" << s << "]";
        EXPECT_NEAR(SSPRK54::gamma2[s], rk.Ac[s + 1], tol) << "gamma2[" << s << "]";
        EXPECT_NEAR(SSPRK54::gamma3[s], 0.5 * rk.Ac2[s + 1], tol) << "gamma3[" << s << "]";
        EXPECT_NEAR(SSPRK54::gamma4[s], rk.AAc[s + 1], tol) << "gamma4[" << s << "]";
    }

    // Index 4 (the final blended state) holds the EXACT order-condition values
    // {1, 1/2, 1/6, 1/6}; the tableau b-combos reproduce them only to the
    // residual of the 15-digit tableau literals (~5e-16), hence the looser bound.
    double sb{}, sbc{}, sbc2{}, sbAc{};
    for (int j = 0; j < 5; ++j)
    {
        sb += rk.b[j];
        sbc += rk.b[j] * rk.c[j];
        sbc2 += rk.b[j] * rk.c[j] * rk.c[j];
        sbAc += rk.b[j] * rk.Ac[j];
    }
    double constexpr tol4 = 2.22e-15;
    EXPECT_NEAR(SSPRK54::gamma1[4], sb, tol4);
    EXPECT_NEAR(SSPRK54::gamma2[4], sbc, tol4);
    EXPECT_NEAR(SSPRK54::gamma3[4], 0.5 * sbc2, tol4);
    EXPECT_NEAR(SSPRK54::gamma4[4], sbAc, tol4);
}


TEST(SSPRK54Coefficients, fourthOrderConditions)
{
    // The 8 order conditions for 4th order, evaluated on the Butcher form
    // rebuilt from the w's. The Spiteri-Ruuth literals as printed (15 digits) satisfy them to
    // <= 4.84e-16 in exact rational arithmetic (15-digit rounding noise, not
    // source truncation); double evaluation adds less than its own size
    // (worst 5.6e-16). Tolerance = 2.22e-15 = 4x the worst observed residual.
    double constexpr tol = 2.22e-15;
    double sb{}, sbc{}, sbc2{}, sbc3{}, sbAc{}, sbcAc{}, sbAc2{}, sbAAc{};
    for (int j = 0; j < 5; ++j)
    {
        sb += rk.b[j];
        sbc += rk.b[j] * rk.c[j];
        sbc2 += rk.b[j] * rk.c[j] * rk.c[j];
        sbc3 += rk.b[j] * rk.c[j] * rk.c[j] * rk.c[j];
        sbAc += rk.b[j] * rk.Ac[j];
        sbcAc += rk.b[j] * rk.c[j] * rk.Ac[j];
        sbAc2 += rk.b[j] * rk.Ac2[j];
        sbAAc += rk.b[j] * rk.AAc[j];
    }
    EXPECT_NEAR(sb, 1.0, tol);
    EXPECT_NEAR(sbc, 1.0 / 2.0, tol);
    EXPECT_NEAR(sbc2, 1.0 / 3.0, tol);
    EXPECT_NEAR(sbAc, 1.0 / 6.0, tol);
    EXPECT_NEAR(sbc3, 1.0 / 4.0, tol);
    EXPECT_NEAR(sbcAc, 1.0 / 8.0, tol);
    EXPECT_NEAR(sbAc2, 1.0 / 12.0, tol);
    EXPECT_NEAR(sbAAc, 1.0 / 24.0, tol);
}


TEST(Reconstruct, endpointReproducesSSPRKUpdate)
{
    // chi=1, dtFine=0: reconstruct == y_n + dtCoarse * sum_i b_i k_i (the
    // coarse step's own update), independent of the split terms.
    std::array<std::array<double, 5>, 3> const kSets{{{0.7, -1.3, 2.1, 0.4, -0.9},
                                                      {1e-3, 2e-3, -5e-4, 3e-3, -1e-3},
                                                      {-4.2, 0.0, 7.7, -2.5, 1.1}}};
    double const yN       = 1.234;
    double const dtCoarse = 0.05;

    for (auto const& k : kSets)
    {
        double expected = yN;
        for (int i = 0; i < 5; ++i)
            expected += dtCoarse * SSPRK54::b[i] * k[i];

        double const rec
            = mc2011::reconstruct(yN, k, /*splitA=*/7.7, /*splitB=*/-3.3, /*chi=*/1.0, dtCoarse,
                                  /*dtFine=*/0.0, /*stageIndex=*/2);
        EXPECT_NEAR(rec, expected, 1e-14 * std::abs(expected));
    }
}


TEST(Reconstruct, dtFineZeroIsStageIndependent)
{
    // Every gamma term is dtFine-multiplied, so dtFine=0 yields the pure CE
    // value ~y(chi): identical for all stage indices and any split values.
    std::array<double, 5> const k{0.7, -1.3, 2.1, 0.4, -0.9};
    double const yN       = -0.6;
    double const dtCoarse = 0.02;

    for (double const chi : {0.0, 0.37, 0.5, 1.0})
    {
        double const ref = mc2011::reconstruct(yN, k, 1.0, 2.0, chi, dtCoarse, 0.0, 0);
        for (std::size_t stage = 1; stage < 5; ++stage)
        {
            double const rec = mc2011::reconstruct(yN, k, 100.0 * stage, -50.0 * stage, chi,
                                                   dtCoarse, 0.0, stage);
            EXPECT_EQ(rec, ref) << "chi = " << chi << ", stage = " << stage;
        }
    }
}


TEST(SplitTerms, matchesDefiningCombinations)
{
    // splitA = sum_i wsplit_i k_i / dt^2, splitB = (6/dt^2) sum_i beta3_i k_i - splitA.
    std::array<double, 5> const k{0.7, -1.3, 2.1, 0.4, -0.9};
    double const dt     = 0.05;
    double const invDt2 = 1.0 / (dt * dt);

    double s{}, u3{};
    for (int i = 0; i < 5; ++i)
    {
        s += SSPRK54::wsplit[i] * k[i];
        u3 += SSPRK54::beta3[i] * k[i];
    }

    auto const [splitA, splitB] = mc2011::splitTerms(k, invDt2);
    EXPECT_DOUBLE_EQ(splitA, s * invDt2);
    EXPECT_DOUBLE_EQ(splitB, 6.0 * invDt2 * u3 - splitA);
}


/*
  Back-solve tests: the stage derivatives k_i recovered from the stored stage states must match
  the ones computed directly from the fluxes, and the chi = 1 reconstruction must return the
  stored final state.

  Emulates one full SSPRK(5,4) coarse sweep exactly as SSPRK4_5Integrator
  performs it -- same production kernels, same fp forms:
    - stage updates: FiniteVolumeEulerPerField with the pre-multiplied
      Shu-Osher sub-increment w_i*dt (stage 1 distinct-output, stages 2-4
      in-place after the RKUtils combination);
    - combinations: `sum = 0.0; sum += w_a*y_a; sum += w_b*y_b` per cell on
      evalOnBox, mirroring RKUtils::RKstep_ left-to-right accumulation;
    - final blend: butcher form, U0 - dt*div(sum_i b_i F_i) with the b-weighted
      flux accumulation of accumulateButcherFluxes_ (SSPRK54::b holds the very
      same w-product collapses the integrator passes).
  Each stage gets its own synthetic flux pair, standing in for F(Y_i).
*/
class StateBackSolve : public ::testing::Test
{
protected:
    using Layout = typename PHARE::core::PHARE_Types<kMHD2DOpts>::MHD::GridLayout_t;
    using FvE    = FiniteVolumeEulerPerField<Layout>;

    static constexpr double dt = 0.01;

    Layout const layout{{{0.1, 0.1}}, {{20, 20}}, Point{0.0, 0.0}};

    UsableFieldMHD<2> U0{"U0", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y1{"Y1", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y2{"Y2", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y3{"Y3", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y4{"Y4", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> UNP1{"UNP1", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> zero{"zero", layout, MHDQuantity::Scalar::rho}; // never written; stays 0

    UsableFieldMHD<2> k1cap{"k1cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k2cap{"k2cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k3cap{"k3cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k4cap{"k4cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k5cap{"k5cap", layout, MHDQuantity::Scalar::rho};

    // Independent oracle for rows 1-4 (StateBackSolve.rowsOneToFourMatchDirectFluxDivergence):
    // -divF_i evaluated directly from stage i's own flux pair, structurally independent of
    // backSolve's (Y_i - combo)/(w_i*dt) inversion algebra.
    UsableFieldMHD<2> k1Direct{"k1Direct", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k2Direct{"k2Direct", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k3Direct{"k3Direct", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k4Direct{"k4Direct", layout, MHDQuantity::Scalar::rho};

    UsableFieldMHD<2> F1x{"F1x", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> F2x{"F2x", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> F3x{"F3x", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> F4x{"F4x", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> F5x{"F5x", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> F1y{"F1y", layout, MHDQuantity::Scalar::ScalarFlux_y};
    UsableFieldMHD<2> F2y{"F2y", layout, MHDQuantity::Scalar::ScalarFlux_y};
    UsableFieldMHD<2> F3y{"F3y", layout, MHDQuantity::Scalar::ScalarFlux_y};
    UsableFieldMHD<2> F4y{"F4y", layout, MHDQuantity::Scalar::ScalarFlux_y};
    UsableFieldMHD<2> F5y{"F5y", layout, MHDQuantity::Scalar::ScalarFlux_y};
    UsableFieldMHD<2> Faccx{"Faccx", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> Faccy{"Faccy", layout, MHDQuantity::Scalar::ScalarFlux_y};

    void SetUp() override
    {
        layout.evalOnGhostBox(
            U0, [&](auto const&... args) { U0(args...) = fval(1.5, 0.4, 0.3, args...); });

        auto fillFlux = [&](auto& F, double const a, double const fx, double const fy) {
            layout.evalOnGhostBox(
                F, [&](auto const&... args) { F(args...) = fval(a, fx, fy, args...); });
        };
        fillFlux(F1x, 0.20, 0.70, 0.50);
        fillFlux(F1y, -0.10, 0.60, 0.80);
        fillFlux(F2x, 0.15, 0.65, 0.45);
        fillFlux(F2y, -0.05, 0.55, 0.75);
        fillFlux(F3x, 0.25, 0.60, 0.55);
        fillFlux(F3y, -0.15, 0.50, 0.70);
        fillFlux(F4x, 0.10, 0.75, 0.40);
        fillFlux(F4y, 0.00, 0.65, 0.85);
        fillFlux(F5x, 0.30, 0.55, 0.60);
        fillFlux(F5y, -0.20, 0.45, 0.65);

        // Shu-Osher combination, fp form of RKUtils::RKstep_
        auto comb = [&](auto& out, double const wa, auto const& a, double const wb, auto const& b) {
            layout.evalOnBox(out, [&](auto const&... args) {
                double sum = 0.0;
                sum += wa * a(args...);
                sum += wb * b(args...);
                out(args...) = sum;
            });
        };

        FvE{layout, SSPRK54::w0 * dt}(U0, Y1, F1x, F1y);

        comb(Y2, SSPRK54::w10, U0, SSPRK54::w11, Y1);
        FvE{layout, SSPRK54::w12 * dt}(Y2, Y2, F2x, F2y);

        comb(Y3, SSPRK54::w20, U0, SSPRK54::w21, Y2);
        FvE{layout, SSPRK54::w22 * dt}(Y3, Y3, F3x, F3y);

        comb(Y4, SSPRK54::w30, U0, SSPRK54::w31, Y3);
        FvE{layout, SSPRK54::w32 * dt}(Y4, Y4, F4x, F4y);

        // Butcher-form final blend: Un+1 = U0 - dt*div(sum_i b_i F_i)
        layout.evalOnGhostBox(Faccx, [&](auto const&... args) {
            Faccx(args...) = SSPRK54::b[0] * F1x(args...) + SSPRK54::b[1] * F2x(args...)
                             + SSPRK54::b[2] * F3x(args...) + SSPRK54::b[3] * F4x(args...)
                             + SSPRK54::b[4] * F5x(args...);
        });
        layout.evalOnGhostBox(Faccy, [&](auto const&... args) {
            Faccy(args...) = SSPRK54::b[0] * F1y(args...) + SSPRK54::b[1] * F2y(args...)
                             + SSPRK54::b[2] * F3y(args...) + SSPRK54::b[3] * F4y(args...)
                             + SSPRK54::b[4] * F5y(args...);
        });
        FvE{layout, dt}(U0, UNP1, std::as_const(Faccx), std::as_const(Faccy));

        // Reference k's for rows 1-4, from backSolve's own formula (Y_i - combo)/(w_i*dt)
        // with the same weights and operand order.
        layout.evalOnBox(U0, [&](auto const&... args) {
            k1cap(args...) = (Y1(args...) - U0(args...)) / (SSPRK54::w0 * dt);
            k2cap(args...)
                = (Y2(args...) - (SSPRK54::w10 * U0(args...) + SSPRK54::w11 * Y1(args...)))
                  / (SSPRK54::w12 * dt);
            k3cap(args...)
                = (Y3(args...) - (SSPRK54::w20 * U0(args...) + SSPRK54::w21 * Y2(args...)))
                  / (SSPRK54::w22 * dt);
            k4cap(args...)
                = (Y4(args...) - (SSPRK54::w30 * U0(args...) + SSPRK54::w31 * Y3(args...)))
                  / (SSPRK54::w32 * dt);
        });

        // Row 5 reference is -divF(F5x,F5y) (dt-independent): dt=1, U=zero makes
        // FiniteVolumeEulerPerField return the raw divergence (Unew = 0 - 1*divF = -divF).
        FvE{layout, 1.0}(zero, k5cap, F5x, F5y);

        // Independent oracle for rows 1-4: the same dt=1,U=zero raw-divergence trick
        // as row 5 above, applied to each stage's own flux pair (Fix,Fiy) instead of
        // inverting Y_i. It never reads Y1..Y4/U0 and never uses backSolve's
        // (Y_i - combo)/(w_i*dt) formula, so it shares no algebra with backSolve().
        FvE{layout, 1.0}(zero, k1Direct, F1x, F1y);
        FvE{layout, 1.0}(zero, k2Direct, F2x, F2y);
        FvE{layout, 1.0}(zero, k3Direct, F3x, F3y);
        FvE{layout, 1.0}(zero, k4Direct, F4x, F4y);
    }
};



TEST_F(StateBackSolve, rowsOneToFourMatchDirectFluxDivergence)
{
    // Independent check for rows 1-4: k1Direct..k4Direct (SetUp() above) are -divF_i
    // evaluated directly from each stage's own flux pair via FiniteVolumeEulerPerField's
    // dt=1,U=zero raw-divergence trick -- the same technique row 5 already used below --
    // rather than inverting Y_i with backSolve's own (Y_i - combo)/(w_i*dt) formula. In
    // exact arithmetic the two are equal, since Y_i was itself built as
    // combo_i - (w_i*dt)*divF_i, so (Y_i - combo_i)/(w_i*dt) == -divF_i algebraically.
    //
    // They are not bit-identical in floating point: forming Y_i multiplies divF_i by the
    // small coefficient w_i*dt (~4e-3 to ~5.5e-3 here, dt=0.01) and adds it to combo_i
    // (O(1) in magnitude), then backSolve's inversion subtracts combo_i back off and
    // divides by that same small w_i*dt -- a subtract-small-from-large-then-divide
    // pattern that amplifies the O(eps) rounding of the Y_i-combo_i subtraction by
    // 1/(w_i*dt) ~ O(1e2-1e3). This is the same order of cancellation amplification
    // also behind the 1e-10*scale tolerance on row 5 below (there the amplification is
    // 1/(b5*dt) ~ 4e2), so the same bound is used here.
    double maxAbsK[4] = {0.0, 0.0, 0.0, 0.0};
    layout.evalOnBox(U0, [&](auto const&... args) {
        auto const k = mc2011::backSolve(U0(args...), Y1(args...), Y2(args...), Y3(args...),
                                         Y4(args...), UNP1(args...), dt);
        double const direct[4]
            = {k1Direct(args...), k2Direct(args...), k3Direct(args...), k4Direct(args...)};
        for (std::size_t i = 0; i < 4; ++i)
        {
            double const scale = std::max({1.0, std::abs(direct[i]), std::abs(k[i])});
            EXPECT_NEAR(k[i], direct[i], 1e-10 * scale) << "row " << (i + 1);
            maxAbsK[i] = std::max(maxAbsK[i], std::abs(direct[i]));
        }
    });
    // Each stage individually, not just their aggregate max: a bug that zeroes exactly
    // one row (e.g. a wrong flux pair wired to one FvE call) must not hide behind the
    // other three rows being nonzero.
    EXPECT_GT(maxAbsK[0], 0.0) << "row 1";
    EXPECT_GT(maxAbsK[1], 0.0) << "row 2";
    EXPECT_GT(maxAbsK[2], 0.0) << "row 3";
    EXPECT_GT(maxAbsK[3], 0.0) << "row 4";
}


TEST_F(StateBackSolve, butcherResidualK5MatchesDirectCapture)
{
    // k5-hat is the butcher residual; the reference is -divF(F5x,F5y) computed
    // directly. Equal in exact arithmetic (div is linear in the fluxes), but
    // the blend divergences the b-weighted flux SUM, so fp equality is only
    // approximate -- rounding in Un+1 is amplified by 1/(b5*dt) ~ 4e2 here.
    double maxAbsK5 = 0.0;
    layout.evalOnBox(U0, [&](auto const&... args) {
        auto const k       = mc2011::backSolve(U0(args...), Y1(args...), Y2(args...), Y3(args...),
                                               Y4(args...), UNP1(args...), dt);
        double const scale = std::max({1.0, std::abs(k5cap(args...)), std::abs(k[4])});
        EXPECT_NEAR(k[4], k5cap(args...), 1e-10 * scale);
        maxAbsK5 = std::max(maxAbsK5, std::abs(k[4]));
    });
    EXPECT_GT(maxAbsK5, 0.0);
}


TEST_F(StateBackSolve, chiOneEndpointReproducesStoredUnp1)
{
    // The residual k5-hat makes the chi=1, dtFine=0 reconstruction return the
    // stored Un+1 exactly in exact arithmetic. In fp it is NOT bit-for-bit:
    // reconstruct accumulates the endpoint as
    // y0 + sum_i fl(dtC * b_i(1) * k_i) with b_i(1) = fl(beta1+beta2+beta3),
    // while the residual was solved from unp1 = y0 + fl(dtC * sum_i b_i k_i)
    // with b_i directly -- different groupings, so the round-trip carries a few
    // rounding steps (measured <= 2 ULP on this sweep; a
    // bit-for-bit result would require the two fp paths to coincide, which
    // they do not). The tight
    // relative tolerance below bounds that; a regression in the coefficient
    // wiring (wrong b_i, wrong beta sum, wrong stage index) shows up orders of
    // magnitude above it.
    double const invDt2   = 1.0 / (dt * dt);
    double constexpr rtol = 4.0 * std::numeric_limits<double>::epsilon();
    layout.evalOnBox(U0, [&](auto const&... args) {
        auto const k = mc2011::backSolve(U0(args...), Y1(args...), Y2(args...), Y3(args...),
                                         Y4(args...), UNP1(args...), dt);
        auto const [splitA, splitB] = mc2011::splitTerms(k, invDt2);
        double const rec = mc2011::reconstruct(U0(args...), k, splitA, splitB, /*chi=*/1.0, dt,
                                               /*dtFine=*/0.0, /*stageIndex=*/4);
        EXPECT_NEAR(rec, UNP1(args...), rtol * std::abs(UNP1(args...)));
    });
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
