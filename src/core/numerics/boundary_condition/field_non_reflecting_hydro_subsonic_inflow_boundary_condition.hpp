#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_INFLOW_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_INFLOW_BOUNDARY_CONDITION_HPP

#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_non_reflecting_hydro_base_boundary_condition.hpp"
#include "core/numerics/thermo/thermo.hpp"

#include <array>
#include <memory>

namespace PHARE::core
{

/**
 * @brief Hydrodynamic LODI characteristic non-reflecting subsonic inflow.
 *
 * Counterpart to FieldNonReflectingHydroSubsonicOutflowBoundaryCondition. Prescribes a
 * target inflow state (ρ*, V*) — pressure stays free, its value at the boundary is set
 * entirely by the outgoing acoustic wave coming from the interior.
 *
 * Frame: inward-normal — @c sign_n = +1 at Lower, −1 at Upper, so @c u_n is the
 * inward-normal velocity (positive at any subsonic inflow). All the common machinery
 * (boundary-face primitive reconstruction, EOS pressure recovery, LODI ODE step,
 * 2nd-order Dirichlet-style ghost extrapolation, Etot reconstruction) lives in
 * FieldNonReflectingHydroBaseBoundaryCondition; this class only contributes the wave-amplitude
 * convention.
 *
 * Eigenstructure in the inward-normal frame at a subsonic inflow (0 < u_n < c):
 *     λ_5 = u_n + c  > 0 → L_5 incoming (acoustic into the domain)
 *     λ_0 = u_n      > 0 → L_2 (entropy), L_3, L_4 (shears) all incoming
 *     λ_1 = u_n − c  < 0 → L_1 outgoing (acoustic toward the interior)
 *
 * Outgoing L_1 from a one-sided diff of the previous-substage state:
 *     L_1 = (u_n − c) (∂p/∂n − ρ c ∂u_n/∂n)
 *
 * The four incoming amplitudes are soft-relaxed toward the user-supplied targets with
 * three user-prescribed coefficients. Rudy-Strikwerda scaling K = σ(1−M²)c/L is *not*
 * meaningful at an inflow — the user prescribes the rate of approach to the target
 * directly. A value of 0 disables that relaxation (pure non-reflecting on that quantity).
 *
 *     L_2 = relax_density    · (ρ   − ρ*)
 *     L_3 = relax_velocity_t · (u_t1 − u_t1*)
 *     L_4 = relax_velocity_t · (u_t2 − u_t2*)
 *     L_5 = relax_velocity_n · (u_n  − u_n*)
 *
 * Hydrodynamic only — the LODI relations carry no magnetic eigenmodes. B at the inlet is
 * handled separately via a DivergenceFreeTransverseDirichlet BC (target B supplied via
 * the user dict, same recipe as the existing free-pressure-inflow).
 *
 * @tparam ScalarOrTensorFieldT Scalar field (ρ, Etot1) or vector field (ρv).
 * @tparam GridLayoutT          Grid layout type.
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldNonReflectingHydroSubsonicInflowBoundaryCondition
    : public FieldNonReflectingHydroBaseBoundaryCondition<
          FieldNonReflectingHydroSubsonicInflowBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>,
          ScalarOrTensorFieldT, GridLayoutT>
{
    using Base = FieldNonReflectingHydroBaseBoundaryCondition<
        FieldNonReflectingHydroSubsonicInflowBoundaryCondition, ScalarOrTensorFieldT, GridLayoutT>;

public:
    FieldNonReflectingHydroSubsonicInflowBoundaryCondition(
        double rho_target, std::array<double, 3> V_target, double relax_velocity_n,
        double relax_velocity_t, double relax_density, std::shared_ptr<Thermo> thermo)
        : Base{std::move(thermo)}
        , rho_target_{rho_target}
        , V_target_{V_target}
        , relax_velocity_n_{relax_velocity_n}
        , relax_velocity_t_{relax_velocity_t}
        , relax_density_{relax_density}
    {
    }

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::NonReflectingHydroSubsonicInflow;
    }

    // CRTP hooks — public so the base can reach them without `friend`. Treat the
    // trailing underscore as "implementation detail, not part of the public API".

    /// Inward frame: u_n is the inward-normal velocity (positive at any inflow).
    static constexpr int frame_sign() { return -1; }

    LodiAmplitudes computeAmplitudes_(LodiInputs const& in, std::size_t dir_n,
                                      double sign_n) const
    {
        auto const tang          = Base::tangentialDirs(dir_n);
        std::size_t const dir_t1 = tang[0];
        std::size_t const dir_t2 = tang[1];

        // Target inward-normal velocity in the local rotated frame. Same convention as
        // the base: u_n_target = sign_n · V_target_cartesian[dir_n] (positive at any
        // inflow). Tangential targets stay cartesian (matches in.u[1], in.u[2]).
        double const u_n_target  = sign_n * V_target_[dir_n];
        double const u_t1_target = V_target_[dir_t1];
        double const u_t2_target = V_target_[dir_t2];

        // Outgoing wave (λ_1 = u_n − c < 0 in inward frame) — from one-sided diff.
        double const L1 = (in.u[0] - in.c) * (in.dP_dn - in.rho_b * in.c * in.duN_dn);

        // Incoming amplitudes — soft relax toward target.
        double const L2 = relax_density_ * (in.rho_b - rho_target_);
        double const L3 = relax_velocity_t_ * (in.u[1] - u_t1_target);
        double const L4 = relax_velocity_t_ * (in.u[2] - u_t2_target);
        double const L5 = relax_velocity_n_ * (in.u[0] - u_n_target);

        return {L1, L2, L3, L4, L5};
    }

private:
    double rho_target_;
    std::array<double, 3> V_target_;
    double relax_velocity_n_;
    double relax_velocity_t_;
    double relax_density_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_SUBSONIC_INFLOW_BOUNDARY_CONDITION_HPP
