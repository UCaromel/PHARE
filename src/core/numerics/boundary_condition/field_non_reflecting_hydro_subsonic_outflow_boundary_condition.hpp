#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_OUTFLOW_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_OUTFLOW_BOUNDARY_CONDITION_HPP

#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_non_reflecting_hydro_base_boundary_condition.hpp"
#include "core/numerics/thermo/thermo.hpp"

#include <cmath>
#include <memory>

namespace PHARE::core
{

/**
 * @brief Hydrodynamic LODI non-reflecting subsonic outflow with target-pressure relax.
 *
 * Counterpart to FieldNonReflectingHydroSubsonicInflowBoundaryCondition. The common
 * machinery (boundary-face primitive reconstruction, EOS pressure recovery, LODI ODE
 * step, 2nd-order Dirichlet-style ghost extrapolation, Etot reconstruction) lives in
 * FieldNonReflectingHydroBaseBoundaryCondition; this class only contributes the wave-amplitude
 * convention.
 *
 * Frame: outward-normal — @c sign_n = +1 at Upper, −1 at Lower, so @c u_n is the
 * outward-normal velocity (positive at any subsonic outflow).
 *
 * Eigenstructure in the outward-normal frame at a subsonic outflow (0 < u_n < c):
 *     λ_5 = u_n + c  > 0 → L_5 outgoing
 *     λ_0 = u_n      > 0 → L_2 (entropy), L_3, L_4 (shears) all outgoing
 *     λ_1 = u_n − c  < 0 → L_1 incoming (acoustic), soft-relaxed to p_target
 *
 * Outgoing amplitudes (L_2..L_5) from one-sided diffs on the previous-substage state:
 *     L_2 = u_n (∂ρ/∂n − ∂p/∂n / c²)
 *     L_3 = u_n  ∂u_t1/∂n
 *     L_4 = u_n  ∂u_t2/∂n
 *     L_5 = (u_n + c)(∂p/∂n + ρ c ∂u_n/∂n)
 *
 * Incoming amplitude (Rudy-Strikwerda soft outflow):
 *     L_1 = K (p_b − p_target),   K = σ (1 − M²) c / L_box
 *
 * Hydrodynamic only — the LODI relations carry no magnetic eigenmodes.
 *
 * @tparam ScalarOrTensorFieldT Scalar field (ρ, Etot1) or vector field (ρv).
 * @tparam GridLayoutT          Grid layout type.
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldNonReflectingHydroSubsonicOutflowBoundaryCondition
    : public FieldNonReflectingHydroBaseBoundaryCondition<
          FieldNonReflectingHydroSubsonicOutflowBoundaryCondition<ScalarOrTensorFieldT,
                                                                  GridLayoutT>,
          ScalarOrTensorFieldT, GridLayoutT>
{
    using Base = FieldNonReflectingHydroBaseBoundaryCondition<
        FieldNonReflectingHydroSubsonicOutflowBoundaryCondition, ScalarOrTensorFieldT, GridLayoutT>;

public:
    FieldNonReflectingHydroSubsonicOutflowBoundaryCondition(double p_target, double sigma,
                                                            double length_scale,
                                                            std::shared_ptr<Thermo> thermo)
        : Base{std::move(thermo)}
        , p_target_{p_target}
        , sigma_{sigma}
        , length_scale_{length_scale}
    {
    }

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::NonReflectingHydroSubsonicOutflow;
    }

    // CRTP hooks — public so the base can reach them without `friend`. Treat the
    // trailing underscore as "implementation detail, not part of the public API".

    /// Outward frame: u_n is the outward-normal velocity (positive at any outflow).
    static constexpr int frame_sign() { return +1; }

    LodiAmplitudes computeAmplitudes_(LodiInputs const& in, std::size_t /*dir_n*/,
                                      double /*sign_n*/) const
    {
        double const M = std::abs(in.u[0]) / in.c;
        double const K = sigma_ * (1.0 - M * M) * in.c / length_scale_;

        // Incoming acoustic — Rudy-Strikwerda soft relax to target pressure.
        double const L1 = K * (in.p_b - p_target_);
        // Outgoing amplitudes — from one-sided diffs.
        double const L2 = in.u[0] * (in.drho_dn - in.dP_dn / (in.c * in.c));
        double const L3 = in.u[0] * in.duT1_dn;
        double const L4 = in.u[0] * in.duT2_dn;
        double const L5 = (in.u[0] + in.c) * (in.dP_dn + in.rho_b * in.c * in.duN_dn);

        return {L1, L2, L3, L4, L5};
    }

private:
    double p_target_;
    double sigma_;
    double length_scale_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_OUTFLOW_BOUNDARY_CONDITION_HPP
