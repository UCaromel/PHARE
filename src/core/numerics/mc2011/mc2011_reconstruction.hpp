#ifndef PHARE_CORE_NUMERICS_MC2011_MC2011_RECONSTRUCTION_HPP
#define PHARE_CORE_NUMERICS_MC2011_MC2011_RECONSTRUCTION_HPP

#include <array>
#include <cstddef>

namespace PHARE::core::mc2011
{

// Single source of truth for every SSPRK(5,4)/MC2011 temporal-reconstruction
// coefficient. Consumed by SSPRK4_5Integrator (Shu-Osher stage sweep + Tier 1
// split terms) and MHDMessenger (Tier 2/3 coarse-fine ghost assembly).
//
// Derivation: full-derivation.md (MC2011 temporal reconstruction); frozen
// literals are exact-rational (SymPy) outputs evaluated to double, locked
// in-tree by tests/amr/solvers/time_integrator/test_mc2011_kernels.cpp.
//
// References:
//  - Spiteri, Ruuth, SIAM J. Numer. Anal. 40 (2002) 469-491 (SSPRK(5,4) tableau)
//  - McCorquodale, Colella, CAMCoS 6 (2011) 1-25 (stage-consistent C-F ghosts)
struct SSPRK54
{
    // Shu-Osher tableau (Spiteri-Ruuth), 15-digit published literals.
    static constexpr double w0{0.391752226571890};
    static constexpr double w10{0.444370493651235};
    static constexpr double w11{0.555629506348765};
    static constexpr double w12{0.368410593050371};
    static constexpr double w20{0.620101851488403};
    static constexpr double w21{0.379898148511597};
    static constexpr double w22{0.251891774271694};
    static constexpr double w30{0.178079954393132};
    static constexpr double w31{0.821920045606868};
    static constexpr double w32{0.544974750228521};
    static constexpr double w40{0.517231671970585};
    static constexpr double w41{0.096059710526147};
    static constexpr double w42{0.063692468666290};
    static constexpr double w43{0.386708617503268};
    static constexpr double w44{0.226007483236906};

    // Stage abscissae c_i (row sums of the Shu-Osher tableau): the physical
    // time t_n + c[i]*dt that intermediate state Y_{i+2} approximates. Match
    // Spiteri-Ruuth SSPRK(5,4) c = [0.39175, 0.58608, 0.47454, 0.93501].
    static constexpr double c1{w0};
    static constexpr double c2{w11 * c1 + w12};
    static constexpr double c3{w21 * c2 + w22};
    static constexpr double c4{w31 * c3 + w32};
    static constexpr double c[4]{c1, c2, c3, c4};

    // Butcher weights b_i (Shu-Osher -> Butcher collapse), kept at working
    // precision so beta3 below reproduces the coarse step's bit-exact endpoint
    // rather than relying on a separately truncated frozen literal.
    static constexpr double b[5]{
        w0 * w11 * w21 * w31 * w43 + w0 * w11 * w21 * w41 + w0 * w11 * w40,
        w12 * w21 * w31 * w43 + w12 * w21 * w41 + w12 * w40, w22 * w31 * w43 + w22 * w41,
        w32 * w43 + w42, w44};

    // Frozen derivation outputs (exact-rational recipe evaluated to double, 17
    // digits): the cubic continuous extension (CE)
    //   ~y(chi) = y_n + dt sum_i b_i(chi) k_i,
    //   b_i(t)  = beta1_i t + beta2_i t^2 + beta3_i t^3,
    // and the elementary-differential split weights wsplit
    // ((f')^2 f = sum_i wsplit_i k_i / dt^2 + O(dt)).
    static constexpr double beta1[5]{+1.00238904846721044e+00, -1.02334729999065607e-02,
                                     +1.54105578630404257e-02, -4.45543398153150968e-03,
                                     -3.11069934881285509e-03};
    static constexpr double beta2[5]{-1.73049925661196125e+00, +1.29013264577362285e+00,
                                     -1.29093852347305887e-01, +1.00377088053292152e+00,
                                     -4.34310417347277322e-01};
    static constexpr double wsplit[5]{+3.69581767513502468e-01, -5.00865888704944240e+00,
                                      -1.57474970044687246e+00, +6.35203824800890260e+00,
                                      -1.38211428026090349e-01};

    // beta3 := b - beta1 - beta2, recomputed here (not frozen) so the CE's
    // endpoint (chi=1) reproduces the SSPRK update bit-for-bit (design
    // decision 9, MC2011 temporal wiring).
    static constexpr double beta3[5]{
        b[0] - beta1[0] - beta2[0], b[1] - beta1[1] - beta2[1], b[2] - beta1[2] - beta2[2],
        b[3] - beta1[3] - beta2[3], b[4] - beta1[4] - beta2[4]};

    // Per-stage geometric constants (full-derivation.md Table S5.1; structure
    // proven generically in phase3-gamma-table S0, recursion forms in S3):
    // gamma1=c_i, gamma2=(Ac)_i, gamma3=1/2(Ac^2)_i, gamma4=(AAc)_i. Index
    // 0..3 = the four intermediate states (Butcher stages Y2..Y5); index 4 =
    // the final blended state, whose entries are the EXACT order-condition
    // values {1, 1/2, 1/6, 1/6} (S0.3). Stage rows constexpr-derived from the
    // w's by the Shu-Osher convexity recursion (S3.1), same pattern as c[].
    static constexpr double g21{0.0};
    static constexpr double g22{w12 * c1};
    static constexpr double g23{w21 * g22 + w22 * c2};
    static constexpr double g24{w31 * g23 + w32 * c3};
    static constexpr double g31{0.0};
    static constexpr double g32{w12 * (c1 * c1 / 2.0)};
    static constexpr double g33{w21 * g32 + w22 * (c2 * c2 / 2.0)};
    static constexpr double g34{w31 * g33 + w32 * (c3 * c3 / 2.0)};
    static constexpr double g41{0.0};
    static constexpr double g42{0.0};
    static constexpr double g43{w22 * g22};
    static constexpr double g44{w31 * g43 + w32 * g23};
    static constexpr double gamma1[5]{c1, c2, c3, c4, 1.0};
    static constexpr double gamma2[5]{g21, g22, g23, g24, 0.5};
    static constexpr double gamma3[5]{g31, g32, g33, g34, 1.0 / 6.0};
    static constexpr double gamma4[5]{g41, g42, g43, g44, 1.0 / 6.0};
};


// Tier 0: per-cell back-solve of the five stage derivatives k_i = f(Y_i) from
// the persisted post-stage states of one coarse step (state-backsolve
// derivation.md S5.1): y0 = u_n, y1..y4 = the Shu-Osher stage states Y2..Y5,
// unp1 = the final blended state, dtC = the sweep's own dt.
//
//   k1 = (y1 - y0)/(w0*dtC)
//   k2 = (y2 - (w10*y0 + w11*y1))/(w12*dtC)
//   k3 = (y3 - (w20*y0 + w21*y2))/(w22*dtC)
//   k4 = (y4 - (w30*y0 + w31*y3))/(w32*dtC)
//   k5 = (unp1 - y0 - dtC*(b1*k1 + b2*k2 + b3*k3 + b4*k4))/(b5*dtC)
//
// Rows 1-4 mirror the retired capture kernels' operands exactly -- the
// combination w_a*y_a + w_b*y_b left-to-right as RKUtils accumulates it, one
// subtraction, one division by the pre-multiplied w_i*dtC -- so they are
// bit-identical to the k's the (Unew - u0)/dt capture used to produce
// (derivation.md S5.1, gate G1). Row 5 is the Butcher-residual variant
// (S5.6 DECISION): unconditional exactness under any re-literalization of the
// tableau, chi=1 endpoint bit-exact against the stored unp1 (gate G2), and it
// reuses the production accumulation weights b[] verbatim.
constexpr std::array<double, 5> backSolve(double const y0, double const y1, double const y2,
                                          double const y3, double const y4, double const unp1,
                                          double const dtC)
{
    double const k1 = (y1 - y0) / (SSPRK54::w0 * dtC);
    double const k2 = (y2 - (SSPRK54::w10 * y0 + SSPRK54::w11 * y1)) / (SSPRK54::w12 * dtC);
    double const k3 = (y3 - (SSPRK54::w20 * y0 + SSPRK54::w21 * y2)) / (SSPRK54::w22 * dtC);
    double const k4 = (y4 - (SSPRK54::w30 * y0 + SSPRK54::w31 * y3)) / (SSPRK54::w32 * dtC);

    double const acc = SSPRK54::b[0] * k1 + SSPRK54::b[1] * k2 + SSPRK54::b[2] * k3
                       + SSPRK54::b[3] * k4;
    double const k5 = (unp1 - y0 - dtC * acc) / (SSPRK54::b[4] * dtC);

    return {k1, k2, k3, k4, k5};
}


// Tier 1: theta-independent split terms, per cell from the five stage
// derivatives k_i = f(Y_i) of one coarse step (full-derivation.md S5.3).
// Returns {splitA, splitB} = {(f')^2 f, f'' f^2}; invDt2 = 1/dtCoarse^2
// (k_i = f(Y_i) convention, hence the /dt^2 scaling).
constexpr std::array<double, 2> splitTerms(std::array<double, 5> const& k, double const invDt2)
{
    double s  = 0.0;
    double u3 = 0.0;
    for (std::size_t i = 0; i < 5; ++i)
    {
        s += SSPRK54::wsplit[i] * k[i];
        u3 += SSPRK54::beta3[i] * k[i];
    }
    double const splitA = s * invDt2;
    return {splitA, 6.0 * invDt2 * u3 - splitA};
}


// Tier 2/3: per-cell stage-consistent reconstruction U^(stage) at fine-substep
// fraction chi in [0,1] of the coarse step (full-derivation.md S5.4):
//
//   ~y(chi)   = y_n + dtCoarse * sum_i b_i(chi) k_i
//   ~y'(chi)  = sum_i b_i'(chi) k_i
//   ~y''(chi) = (1/dtCoarse) sum_i b_i''(chi) k_i
//   U^(stage) = ~y(chi) + dtFine*gamma1[stage]*~y'(chi)
//               + dtFine^2*gamma2[stage]*~y''(chi)
//               + dtFine^3*(gamma3[stage]*splitB + gamma4[stage]*splitA)
//
// dtFine == 0 degenerates to the pure CE value ~y(chi) (every gamma term is
// dtFine-multiplied; stageIndex is then irrelevant).
constexpr double reconstruct(double const yN, std::array<double, 5> const& k, double const splitA,
                             double const splitB, double const chi, double const dtCoarse,
                             double const dtFine, std::size_t const stageIndex)
{
    double yTilde = yN;
    double dY     = 0.0;
    double d2Y    = 0.0;
    for (std::size_t s = 0; s < 5; ++s)
    {
        double const b_i
            = SSPRK54::beta1[s] * chi + SSPRK54::beta2[s] * chi * chi
              + SSPRK54::beta3[s] * chi * chi * chi;
        double const bp_i
            = SSPRK54::beta1[s] + 2.0 * SSPRK54::beta2[s] * chi + 3.0 * SSPRK54::beta3[s] * chi * chi;
        double const bpp_i = 2.0 * SSPRK54::beta2[s] + 6.0 * SSPRK54::beta3[s] * chi;

        yTilde += dtCoarse * b_i * k[s];
        dY += bp_i * k[s];
        d2Y += bpp_i * k[s] / dtCoarse;
    }

    return yTilde + dtFine * SSPRK54::gamma1[stageIndex] * dY
           + dtFine * dtFine * SSPRK54::gamma2[stageIndex] * d2Y
           + dtFine * dtFine * dtFine
                 * (SSPRK54::gamma3[stageIndex] * splitB + SSPRK54::gamma4[stageIndex] * splitA);
}

} // namespace PHARE::core::mc2011

#endif
