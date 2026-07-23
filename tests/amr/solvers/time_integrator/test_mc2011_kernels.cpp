/*
  Tests of the PRODUCTION MC2011/SSPRK(5,4) temporal-reconstruction kernels
  (core/numerics/mc2011/mc2011_reconstruction.hpp).

  Unlike test_main.cpp in this directory -- a standalone ODE re-derivation
  sharing only copy-pasted literals with production -- these tests exercise
  the shipped code, so a regression in the kernels fails here.

  The k-capture overloads (FiniteVolumeEulerPerField, Faraday) are no longer
  on the production path -- the messenger back-solves the stage derivatives
  from the persisted stage states (core::mc2011::backSolve) -- but they are
  KEPT as the oracle for the StateBackSolve gate tests below: gate G1 requires
  the back-solved rows 1-4 to be bit-identical to what the capture form used
  to produce (state-backsolve derivation.md S5.9).

  The KCapture tests were originally the oracle for bug F1 (2026-07-17 diff
  audit): SSPRK stages 2-5 pass the same field as U and Unew, so a k-capture
  that reads U after the in-place write yields k == 0 identically. They fill
  real MHD fields on a Yee-MHD layout and require the aliased and distinct k
  captures to agree (and be nonzero).
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

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

// Butcher form of the Shu-Osher tableau, rebuilt independently from the w
// coefficients (conversion as in test_main.cpp's Tableau struct), to check
// the frozen SSPRK54 tables against their defining tableau quantities.
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


// Smooth, index-dependent fill so k = -divF is nonzero and spatially varying.
template<typename... Idx>
double fval(double const a, double const fx, double const fy, Idx const&... ij)
{
    auto const idx = std::array<double, sizeof...(ij)>{static_cast<double>(ij)...};
    return a + fx * std::sin(0.3 * idx[0]) + fy * std::cos(0.2 * idx[1]);
}

} // namespace



TEST(KCapture, finiteVolumeEulerAliasedMatchesDistinct)
{
    using Layout = typename PHARE::core::PHARE_Types<PHARE::SimOpts{2, 1}>::MHD::GridLayout_t;
    Layout const layout{{{0.1, 0.1}}, {{20, 20}}, Point{0.0, 0.0}};

    UsableFieldMHD<2> U{"U", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Ualias{"Ualias", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Unew{"Unew", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> kDistinct{"kDistinct", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> kAliased{"kAliased", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Fx{"Fx", layout, MHDQuantity::Scalar::ScalarFlux_x};
    UsableFieldMHD<2> Fy{"Fy", layout, MHDQuantity::Scalar::ScalarFlux_y};

    layout.evalOnGhostBox(U, [&](auto const&... args) {
        U(args...)      = fval(1.5, 0.4, 0.3, args...);
        Ualias(args...) = U(args...);
    });
    layout.evalOnGhostBox(Fx, [&](auto const&... args) { Fx(args...) = fval(0.2, 0.7, 0.5, args...); });
    layout.evalOnGhostBox(Fy, [&](auto const&... args) { Fy(args...) = fval(-0.1, 0.6, 0.8, args...); });

    double const dt = 0.01;
    FiniteVolumeEulerPerField<Layout> const fvEuler{layout, dt};

    fvEuler(U, Unew, kDistinct, Fx, Fy);        // distinct output (stage-1 pattern)
    fvEuler(Ualias, Ualias, kAliased, Fx, Fy);  // in-place (stages 2-5 pattern)

    double maxAbsK = 0.0;
    layout.evalOnBox(kDistinct, [&](auto const&... args) {
        EXPECT_DOUBLE_EQ(kAliased(args...), kDistinct(args...));
        maxAbsK = std::max(maxAbsK, std::abs(kDistinct(args...)));
    });
    EXPECT_GT(maxAbsK, 0.0); // k == 0 everywhere is exactly the F1 failure mode
}


TEST(KCapture, faradayAliasedMatchesDistinct)
{
    using Layout = typename PHARE::core::PHARE_Types<PHARE::SimOpts{2, 1}>::MHD::GridLayout_t;
    Layout const layout{{{0.1, 0.1}}, {{20, 20}}, Point{0.0, 0.0}};

    UsableVecFieldMHD<2> B{"B", layout, MHDQuantity::Vector::B};
    UsableVecFieldMHD<2> Balias{"Balias", layout, MHDQuantity::Vector::B};
    UsableVecFieldMHD<2> Bnew{"Bnew", layout, MHDQuantity::Vector::B};
    UsableVecFieldMHD<2> kB{"kB", layout, MHDQuantity::Vector::B};
    UsableVecFieldMHD<2> kBalias{"kBalias", layout, MHDQuantity::Vector::B};
    UsableVecFieldMHD<2> E{"E", layout, MHDQuantity::Vector::E};

    for (auto const component : {Component::X, Component::Y, Component::Z})
    {
        auto& Bc      = B(component);
        auto& Baliasc = Balias(component);
        auto& Ec      = E(component);
        layout.evalOnGhostBox(Bc, [&](auto const&... args) {
            Bc(args...)      = fval(0.8, 0.25, 0.35, args...);
            Baliasc(args...) = Bc(args...);
        });
        layout.evalOnGhostBox(Ec, [&](auto const&... args) {
            Ec(args...) = fval(0.1, 0.45, 0.55, args...);
        });
    }

    double const dt = 0.01;
    Faraday<Layout> faraday{layout};

    faraday(B, E, Bnew, kB, dt);              // distinct output (stage-1 pattern)
    faraday(Balias, E, Balias, kBalias, dt);  // in-place (stages 2-5 pattern)

    // Bx is untouched in 1D only; in 2D all three components evolve.
    for (auto const component : {Component::X, Component::Y, Component::Z})
    {
        double maxAbsK = 0.0;
        layout.evalOnBox(kB(component), [&](auto const&... args) {
            EXPECT_DOUBLE_EQ(kBalias(component)(args...), kB(component)(args...));
            maxAbsK = std::max(maxAbsK, std::abs(kB(component)(args...)));
        });
        EXPECT_GT(maxAbsK, 0.0);
    }
}


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
        EXPECT_NEAR(SSPRK54::beta1[i] + SSPRK54::beta2[i] + SSPRK54::beta3[i], SSPRK54::b[i],
                    1e-15)
            << "i = " << i;
}


TEST(SSPRK54Coefficients, gammaTablesMatchTableau)
{
    // gamma1=c_i, gamma2=(Ac)_i, gamma3=(Ac^2)_i/2, gamma4=(AAc)_i for the four
    // intermediate stages (SSPRK54 index s = Butcher index s+1). The stage rows
    // are constexpr Shu-Osher recursions (phase3-gamma-table S3.1) within
    // ~1 ULP of the tableau quantities (S3.3), hence the 1e-15 tolerance.
    double constexpr tol = 1e-15;
    for (int s = 0; s < 4; ++s)
    {
        EXPECT_NEAR(SSPRK54::gamma1[s], rk.c[s + 1], tol) << "gamma1[" << s << "]";
        EXPECT_NEAR(SSPRK54::gamma2[s], rk.Ac[s + 1], tol) << "gamma2[" << s << "]";
        EXPECT_NEAR(SSPRK54::gamma3[s], 0.5 * rk.Ac2[s + 1], tol) << "gamma3[" << s << "]";
        EXPECT_NEAR(SSPRK54::gamma4[s], rk.AAc[s + 1], tol) << "gamma4[" << s << "]";
    }

    // Index 4 (the final blended state) holds the EXACT order-condition values
    // {1, 1/2, 1/6, 1/6} (S0.3); the tableau b-combos reproduce them only to
    // the printed-literal residual (~5e-16, S1.2), hence the looser bound.
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
    // rebuilt from the w's. The printed Spiteri-Ruuth literals satisfy them to
    // <= 4.84e-16 in exact rational arithmetic (phase3-gamma-table S1.2 —
    // 15-digit rounding noise, not source truncation); double evaluation adds
    // less than its own size (S4.1, worst 5.6e-16). Tolerance = 2.22e-15 =
    // 4x the worst observed residual (S4).
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
    double const yN = 1.234;
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
    double const yN = -0.6;
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
    double const dt = 0.05;
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
  Gates G1/G2 of the state-backsolve rework (derivation.md S5.9).

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
  Each stage gets its own synthetic flux pair, standing in for F(Y_i). The
  retired k-capture overloads provide the reference k's.
*/
class StateBackSolve : public ::testing::Test
{
protected:
    using Layout = typename PHARE::core::PHARE_Types<PHARE::SimOpts{2, 1}>::MHD::GridLayout_t;
    using FvE    = FiniteVolumeEulerPerField<Layout>;

    static constexpr double dt = 0.01;

    Layout const layout{{{0.1, 0.1}}, {{20, 20}}, Point{0.0, 0.0}};

    UsableFieldMHD<2> U0{"U0", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y1{"Y1", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y2{"Y2", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y3{"Y3", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y4{"Y4", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> Y5{"Y5", layout, MHDQuantity::Scalar::rho}; // k5-capture scratch output
    UsableFieldMHD<2> UNP1{"UNP1", layout, MHDQuantity::Scalar::rho};

    UsableFieldMHD<2> k1cap{"k1cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k2cap{"k2cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k3cap{"k3cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k4cap{"k4cap", layout, MHDQuantity::Scalar::rho};
    UsableFieldMHD<2> k5cap{"k5cap", layout, MHDQuantity::Scalar::rho};

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
            layout.evalOnGhostBox(F,
                                  [&](auto const&... args) { F(args...) = fval(a, fx, fy, args...); });
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
        auto comb = [&](auto& out, double const wa, auto const& a, double const wb,
                        auto const& b) {
            layout.evalOnBox(out, [&](auto const&... args) {
                double sum = 0.0;
                sum += wa * a(args...);
                sum += wb * b(args...);
                out(args...) = sum;
            });
        };

        FvE{layout, SSPRK54::w0 * dt}(U0, Y1, k1cap, F1x, F1y);

        comb(Y2, SSPRK54::w10, U0, SSPRK54::w11, Y1);
        FvE{layout, SSPRK54::w12 * dt}(Y2, Y2, k2cap, F2x, F2y);

        comb(Y3, SSPRK54::w20, U0, SSPRK54::w21, Y2);
        FvE{layout, SSPRK54::w22 * dt}(Y3, Y3, k3cap, F3x, F3y);

        comb(Y4, SSPRK54::w30, U0, SSPRK54::w31, Y3);
        FvE{layout, SSPRK54::w32 * dt}(Y4, Y4, k4cap, F4x, F4y);

        // Retired stage-5 extraction pattern: distinct-output euler on Y5(=Y4 here),
        // full dt, so k5cap = -divF(Y5). Y5 the field is only a scratch output.
        FvE{layout, dt}(Y4, Y5, k5cap, F5x, F5y);

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
        // as_const so overload resolution picks the plain (no k-capture) overload
        FvE{layout, dt}(U0, UNP1, std::as_const(Faccx), std::as_const(Faccy));
    }
};


TEST_F(StateBackSolve, rowsOneToFourBitIdenticalToRetiredCapture) // gate G1
{
    // backSolve rows 1-4 mirror the capture's operands exactly ((Unew - u0)
    // divided by the pre-multiplied w_i*dt, combinations retraced in RKUtils
    // order), so bit identity -- EXPECT_EQ, not EXPECT_NEAR -- is the contract.
    double maxAbsK = 0.0;
    layout.evalOnBox(U0, [&](auto const&... args) {
        auto const k = mc2011::backSolve(U0(args...), Y1(args...), Y2(args...), Y3(args...),
                                         Y4(args...), UNP1(args...), dt);
        EXPECT_EQ(k[0], k1cap(args...));
        EXPECT_EQ(k[1], k2cap(args...));
        EXPECT_EQ(k[2], k3cap(args...));
        EXPECT_EQ(k[3], k4cap(args...));
        for (auto const& ki : k)
            maxAbsK = std::max(maxAbsK, std::abs(ki));
    });
    EXPECT_GT(maxAbsK, 0.0);
}


TEST_F(StateBackSolve, butcherResidualK5MatchesDirectCapture) // gate G1, row 5
{
    // k5-hat is the butcher residual; the retired capture computed -divF(Y5)
    // directly. Equal in exact arithmetic (div is linear in the fluxes), but
    // the blend divergences the b-weighted flux SUM, so fp equality is only
    // approximate -- rounding in Un+1 is amplified by 1/(b5*dt) ~ 4e2 here.
    double maxAbsK5 = 0.0;
    layout.evalOnBox(U0, [&](auto const&... args) {
        auto const k = mc2011::backSolve(U0(args...), Y1(args...), Y2(args...), Y3(args...),
                                         Y4(args...), UNP1(args...), dt);
        double const scale = std::max({1.0, std::abs(k5cap(args...)), std::abs(k[4])});
        EXPECT_NEAR(k[4], k5cap(args...), 1e-10 * scale);
        maxAbsK5 = std::max(maxAbsK5, std::abs(k[4]));
    });
    EXPECT_GT(maxAbsK5, 0.0);
}


TEST_F(StateBackSolve, chiOneEndpointReproducesStoredUnp1) // gate G2
{
    // The residual k5-hat makes the chi=1, dtFine=0 reconstruction return the
    // stored Un+1 exactly in exact arithmetic (derivation.md S5.6). In fp it is
    // NOT bit-for-bit: reconstruct accumulates the endpoint as
    // y0 + sum_i fl(dtC * b_i(1) * k_i) with b_i(1) = fl(beta1+beta2+beta3),
    // while the residual was solved from unp1 = y0 + fl(dtC * sum_i b_i k_i)
    // with b_i directly -- different groupings, so the round-trip carries a few
    // rounding steps (measured <= 2 ULP on this sweep, 2026-07-19; S5.9's
    // bit-for-bit wording assumed the two fp paths coincide). The tight
    // relative tolerance below is the gate; a regression in the coefficient
    // wiring (wrong b_i, wrong beta sum, wrong stage index) shows up orders of
    // magnitude above it.
    double const invDt2 = 1.0 / (dt * dt);
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
