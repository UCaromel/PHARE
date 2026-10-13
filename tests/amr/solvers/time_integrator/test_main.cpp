/*
  Pure-ODE validation of the stage-consistent temporal coarse-fine reconstruction
  derived for PHARE's SSPRK(5,4) integrator (Spiteri-Ruuth), the temporal half of
  McCorquodale-Colella 2011-style 4th-order AMR ghost fills.

  What is validated, on the Brusselator (nonlinear, 2-component, no scalar-ODE
  coincidences), all errors LOCAL (one coarse step from a reference state):

  1. The Shu-Osher -> Butcher tableau conversion satisfies all 8 order-4 conditions.
  2. Ketcheson-Loczi-Jangabylova-Kusmanov 2017 (arXiv:1605.02429) Eq 24: the unique
     2nd-order SSP dense output for this method. Baseline anchor: local slope 3.
  3. The derived cubic 3rd-order continuous extension (CE)
        ytilde(t_n + th*dt) = y_n + dt sum_i b_i(th) k_i,
        b_i(th) = beta1_i th + beta2_i th^2 + beta3_i th^3,
     endpoint-consistent (b_i(1) = b_i bit-level via beta3 := b - beta1 - beta2):
     local slopes u:4, u':3, u'':2, u''':1, and CE(1) == the SSPRK update.
  4. The elementary-differential split (f')^2 f = sum_i w_i k_i / dt^2 + O(dt).
     NOTE the /dt^2 scaling: PHARE stores k_i = f(Y_i) (CarpetX convention), NOT
     k_i = h f(Y_i) (Mongwane convention, which would give /dt^3).
  5. Fine-stage ghost reconstruction at substep start chi, dt_f = dt/2:
        U^(i) = ytilde(chi) + dt_f c_i u'(chi) + dt_f^2 (Ac)_i u''(chi)
                + dt_f^3 [ (A c^2)_i / 2 * f''f^2 + (A A c)_i * (f')^2 f ]
     with f''f^2 = u''' - (f')^2 f, u''' = (6/dt^2) sum_i beta3_i k_i.
     Gate: O(dt^4), i.e. local slope 4, stages 2..5, chi in {0, 1/2}.

  Coefficients beta1, beta2, w below are frozen outputs of the exact-rational
  (SymPy) derivation; construction recipe and proofs in
  ~/.claude/plans/ucaromel/higher-order-refinement/2026-07-10-mc2011-ssprk-temporal/
  (step0-tableau-rank.md, step1-continuous-extension.md, step2-stage-assembly.md).
  Their defining algebraic systems are re-verified in double precision here.

  References:
   - Spiteri, Ruuth, SIAM J. Numer. Anal. 40 (2002) 469-491 (SSPRK(5,4) tableau)
   - Ketcheson, Loczi, Jangabylova, Kusmanov, arXiv:1605.02429 (SSP dense output;
     Thm 3: SSP dense output capped at order 2; Eq 24; sec 4.4 covers this method)
   - McCorquodale, Colella, CAMCoS 6 (2011) 1-25 (stage-consistent C-F ghosts)
   - Gottlieb, Ketcheson, Shu, J. Sci. Comput. 38 (2009) (Shu-Osher <-> Butcher)
*/

#include <array>
#include <cmath>
#include <vector>

#include "core/numerics/mc2011/mc2011_reconstruction.hpp"

#include "gtest/gtest.h"

namespace
{

using SSPRK54 = PHARE::core::mc2011::SSPRK54;

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


// Shu-Osher coefficients, from the production single source of truth
// (core/numerics/mc2011/mc2011_reconstruction.hpp) — no local literal copies.
constexpr double w0 = SSPRK54::w0;
constexpr double w11 = SSPRK54::w11, w12 = SSPRK54::w12;
constexpr double w21 = SSPRK54::w21, w22 = SSPRK54::w22;
constexpr double w31 = SSPRK54::w31, w32 = SSPRK54::w32;
constexpr double w40 = SSPRK54::w40, w41 = SSPRK54::w41, w42 = SSPRK54::w42;
constexpr double w43 = SSPRK54::w43, w44 = SSPRK54::w44;

struct Tableau
{
    double A[5][5]{}, b[5]{}, c[5]{};
    double Ac[5]{}, Ac2[5]{}, AAc[5]{}; // A*c, A*c^2, A*(A*c)

    Tableau()
    {
        A[1][0] = w0;
        for (int j = 0; j < 5; ++j)
            A[2][j] = w11 * A[1][j];
        A[2][1] += w12;
        for (int j = 0; j < 5; ++j)
            A[3][j] = w21 * A[2][j];
        A[3][2] += w22;
        for (int j = 0; j < 5; ++j)
            A[4][j] = w31 * A[3][j];
        A[4][3] += w32;
        for (int j = 0; j < 5; ++j)
            b[j] = w40 * A[2][j] + w41 * A[3][j] + w43 * A[4][j];
        b[3] += w42;
        b[4] += w44;
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


// Frozen derivation outputs, from the production single source of truth.
// Recipe: beta1 = M^T (M M^T)^-1 e1 + l1*n, beta2 = M^T (M M^T)^-1 (0,1/2,0,0)^T
// + l2*n, n = null(M), l1 = -57/2500, l2 = 37/200 (min-negativity member);
// w = min-norm solution of M w = e4, M = [1; c; c^2; Ac] (4x5, rank 4).
constexpr auto& beta1  = SSPRK54::beta1;
constexpr auto& beta2  = SSPRK54::beta2;
constexpr auto& wsplit = SSPRK54::wsplit;

struct CE
{
    double b1[5], b2[5], b3[5]; // beta3 := b - beta1 - beta2 => endpoint bit-consistent

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


// Brusselator: f0 = A + yInit^2 y1 - (B+1) yInit, f1 = B yInit - yInit^2 y1
constexpr double Abr = 1.0, Bbr = 3.0;

Vec f(Vec const& y)
{
    return {Abr + y[0] * y[0] * y[1] - (Bbr + 1) * y[0], Bbr * y[0] - y[0] * y[0] * y[1]};
}
Vec Jv(Vec const& y, Vec const& v) // Jacobian-vector product
{
    return {(2 * y[0] * y[1] - (Bbr + 1)) * v[0] + y[0] * y[0] * v[1],
            (Bbr - 2 * y[0] * y[1]) * v[0] - y[0] * y[0] * v[1]};
}
Vec Hvv(Vec const& y, Vec const& v) // Hessian contraction f''(y)(v,v)
{
    double const h = 2 * y[1] * v[0] * v[0] + 4 * y[0] * v[0] * v[1];
    return {h, -h};
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



// The 8 order-4 conditions are asserted (with a derived 2.22e-15 tolerance) in
// test_mc2011_kernels.cpp::SSPRK54Coefficients.fourthOrderConditions against
// the same production struct; no duplicate check here.
TEST(SSPRK54Tableau, ceAndSplitCoefficientsSolveTheirDefiningSystems)
{
    // M x = r for M = [1; c; c^2; Ac] and
    //   beta1 -> (1,0,0,0), beta2 -> (0,1/2,0,0), beta3 -> (0,0,1/3,1/6), w -> (0,0,0,1)
    auto Mdot = [&](double const (&x)[5]) {
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


TEST(KetchesonEq24Baseline, sspDenseOutputHasLocalSlope3)
{
    // Eq 24: bbar_1(th) = th - (1 - b_1) th^2, bbar_j(th) = b_j th^2 (j >= 2).
    // Order-2 dense output => one-step (local) error O(dt^3).
    double const th = 0.37;
    std::vector<double> errs;
    for (int p = 3; p <= 9; ++p)
    {
        double const dt = std::pow(2.0, -p);
        auto const s    = stages(yInit, dt);
        double bt[5];
        bt[0] = th - (1 - rk.b[0]) * th * th;
        for (int j = 1; j < 5; ++j)
            bt[j] = rk.b[j] * th * th;
        Vec const u = yInit + dt * weighted(bt, s.k);
        errs.push_back(maxAbs(u - refSolve(yInit, th * dt)));
    }
    EXPECT_NEAR(tailSlope(errs), 3.0, 0.1);
}


TEST(DerivedContinuousExtension, valueAndDerivativesConvergeAtOrders4321)
{
    double const th = 0.37;
    std::vector<double> eu, eup, eupp, euppp;
    for (int p = 3; p <= 9; ++p)
    {
        double const dt = std::pow(2.0, -p);
        auto const s    = stages(yInit, dt);
        double bt[5], btp[5], btpp[5];
        for (int i = 0; i < 5; ++i)
        {
            bt[i]   = ce.b1[i] * th + ce.b2[i] * th * th + ce.b3[i] * th * th * th;
            btp[i]  = ce.b1[i] + 2 * ce.b2[i] * th + 3 * ce.b3[i] * th * th;
            btpp[i] = 2 * ce.b2[i] + 6 * ce.b3[i] * th;
        }
        Vec const u    = yInit + dt * weighted(bt, s.k);
        Vec const up   = weighted(btp, s.k);
        Vec const upp  = (1 / dt) * weighted(btpp, s.k);
        Vec const uppp = (6 / (dt * dt)) * weighted(ce.b3, s.k);

        // truth from analytic elementary differentials at the reference point
        Vec const yr = refSolve(yInit, th * dt);
        Vec const fr = f(yr);
        eu.push_back(maxAbs(u - yr));
        eup.push_back(maxAbs(up - fr));
        eupp.push_back(maxAbs(upp - Jv(yr, fr)));
        euppp.push_back(maxAbs(uppp - (Hvv(yr, fr) + Jv(yr, Jv(yr, fr)))));
    }
    EXPECT_NEAR(tailSlope(eu), 4.0, 0.15);
    EXPECT_NEAR(tailSlope(eup), 3.0, 0.1);
    EXPECT_NEAR(tailSlope(eupp), 2.0, 0.1);
    EXPECT_NEAR(tailSlope(euppp), 1.0, 0.15); // converges to 1 from above
}


TEST(DerivedContinuousExtension, endpointReproducesTheSSPRKUpdate)
{
    for (int p = 3; p <= 6; ++p)
    {
        double const dt = std::pow(2.0, -p);
        auto const s    = stages(yInit, dt);
        double b1[5];
        for (int i = 0; i < 5; ++i)
            b1[i] = ce.b1[i] + ce.b2[i] + ce.b3[i]; // = b_i to <= 1 ulp by construction
        Vec const u = yInit + dt * weighted(b1, s.k);
        EXPECT_LE(maxAbs(u - s.ynew), 1e-14 * maxAbs(s.ynew));
    }
}


TEST(SplitTerm, recoversFPrimeSquaredFAtSlope1)
{
    // (f')^2 f = sum_i w_i k_i / dt^2 + O(dt) -- k_i = f(Y_i) convention => /dt^2
    Vec const truth = Jv(yInit, Jv(yInit, f(yInit)));
    std::vector<double> errs;
    for (int p = 2; p <= 8; ++p)
    {
        double const dt = std::pow(2.0, -p);
        auto const s    = stages(yInit, dt);
        errs.push_back(maxAbs((1 / (dt * dt)) * weighted(wsplit, s.k) - truth));
    }
    EXPECT_NEAR(tailSlope(errs), 1.0, 0.1);
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
                                + (dtf * dtf * dtf)
                                      * (0.5 * rk.Ac2[i] * ffpp + rk.AAc[i] * fpfp);
                err = std::max(err, maxAbs(rec - fine.Y[i]));
            }
            errs.push_back(err);
        }
        EXPECT_NEAR(tailSlope(errs), 4.0, 0.2) << "chi = " << chi;
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
