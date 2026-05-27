#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_BASE_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_BASE_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/numerics/thermo/thermo.hpp"
#include "core/utilities/point/point.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>

namespace PHARE::core
{

/**
 * @brief Conservative boundary-face state at the new time, returned by the LODI ODE step.
 */
struct LodiResult
{
    double rho_bdr;
    double rhoVx_bdr;
    double rhoVy_bdr;
    double rhoVz_bdr;
    double p_bdr;
};


/**
 * @brief LODI wave amplitudes at a boundary face (L_1 .. L_5).
 *
 * Defined in the local rotated frame used by the derived BC (inflow → inward-normal,
 * outflow → outward-normal) such that @c u_n > 0 in all cases. The eigenstructure is then:
 *     λ_1 = u_n − c < 0,   λ_{2,3,4} = u_n > 0,   λ_5 = u_n + c > 0
 * @c L_2 / @c L_3 / @c L_4 correspond to the entropy and two tangential shears; @c L_1
 * and @c L_5 are the acoustic waves. Each derived BC decides which amplitudes come from
 * one-sided spatial diffs (outgoing) and which are soft-relaxed toward user targets
 * (incoming).
 */
struct LodiAmplitudes
{
    double L1;
    double L2;
    double L3;
    double L4;
    double L5;
};


/**
 * @brief Inputs handed to a derived BC's @c computeAmplitudes_() — the boundary-face state
 * (cell-centered primitives averaged across the first interior + first ghost cell), the
 * sound speed, the per-direction one-sided derivatives evaluated AT the face, and the
 * three local-frame velocities. All velocities are in the rotated local frame whose
 * normal axis carries @c sign_n with the convention chosen by the derived BC (inflow uses
 * inward-frame, outflow uses outward-frame).
 */
struct LodiInputs
{
    double rho_b;
    double p_b;
    double c;
    std::array<double, 3> u; ///< {u_n, u_t1, u_t2} in the local rotated frame
    double drho_dn;
    double duN_dn;
    double duT1_dn;
    double duT2_dn;
    double dP_dn;
};


/**
 * @brief CRTP base for HD LODI characteristic boundary conditions (subsonic inflow and
 * outflow).
 *
 * Centralises the common machinery: per-tangential-slice caching, boundary-face primitive
 * reconstruction (with EOS-based pressure recovery from Etot1 because P_old is not
 * guaranteed to be ghost-filled), centered ∂/∂n at the boundary face, forward-Euler LODI
 * ODE step, Dirichlet-style 2nd-order ghost extrapolation, Etot reconstruction at the
 * boundary, and dispatch on physical quantity (ρ, ρv, Etot1) inside @c apply().
 *
 * Frame
 * ─────
 * A local rotated frame is used so that @c u_n is **always positive** regardless of side
 * (any of the four faces in 2D). The derived class picks the frame parity:
 *   * @c frame_sign = +1 → **outward** frame  (used by the outflow BC)
 *   * @c frame_sign = -1 → **inward**  frame  (used by the inflow  BC)
 *
 * From @c frame_sign and the side, everything else follows:
 *   * sign_n = frame_sign · (Upper ? +1 : -1)        → @c u_n = sign_n · V_grid[dir_n] > 0
 *   * diff_n_local = frame_sign · (φ_ghost − φ_interior) / dn
 *
 * Eigenstructure in the local frame (subsonic, 0 < u_n < c):
 *     λ_1 = u_n − c < 0 → propagates toward interior
 *     λ_{2,3,4} = u_n > 0 → propagates with the flow in the local-normal direction
 *     λ_5 = u_n + c > 0 → propagates away from interior
 *
 * Whether λ_i is "incoming" or "outgoing" (relative to the physical domain) depends on
 * frame parity:
 *   * outflow (outward frame): L_2..L_5 outgoing, L_1 incoming (soft-relaxed to p_target).
 *   * inflow  (inward  frame): L_5 outgoing, L_1..L_4 incoming (soft-relaxed to targets).
 *
 * Derived classes only need to provide:
 *   - constructor + member data (targets, relaxation coefficients)
 *   - @c FieldBoundaryConditionType getType() const override
 *   - @c static constexpr int frame_sign()       — +1 outward, -1 inward
 *   - @c LodiAmplitudes computeAmplitudes_(LodiInputs const&, std::size_t dir_n,
 *                                          double sign_n) const
 *
 * Tangential velocities @c u_t1, @c u_t2 in @c LodiInputs are kept cartesian (no sign
 * flip) because the tangential dynamics decouple from the normal sign convention.
 *
 * @tparam Derived              CRTP derived class.
 * @tparam ScalarOrTensorFieldT Scalar field (ρ, Etot1) or vector field (ρv).
 * @tparam GridLayoutT          Grid layout type.
 */
template<typename Derived, typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldNonReflectingHydroBaseBoundaryCondition
    : public IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>
{
public:
    using Super                  = IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>;
    using field_type             = typename Super::field_type;
    using physical_quantity_type = typename GridLayoutT::Quantity;
    using scalar_quantity_type   = typename physical_quantity_type::Scalar;
    using vector_quantity_type   = typename physical_quantity_type::Vector;
    using vectorfield_type       = VecField<field_type, physical_quantity_type>;

    static constexpr std::size_t dimension = Super::dimension;
    static constexpr bool is_scalar        = Super::is_scalar;

    /// Frame parity picked by the derived BC: +1 outward (outflow), -1 inward (inflow).
    static constexpr int frame_sign = Derived::frame_sign();

    explicit FieldNonReflectingHydroBaseBoundaryCondition(std::shared_ptr<Thermo> thermo)
        : thermo_{std::move(thermo)}
    {
    }

    void apply(ScalarOrTensorFieldT& field, BoundaryLocation const boundaryLocation,
               Box<std::uint32_t, dimension> const& localGhostBox, GridLayoutT const& gridLayout,
               typename Super::boundary_condition_context_type const& ctx) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);
        std::size_t const dir_n   = static_cast<std::size_t>(direction);
        double const sign_n       = frame_sign * ((side == Side::Upper) ? +1.0 : -1.0);
        double const dn           = gridLayout.meshSize()[dir_n];

        std::uint32_t const interior_n
            = (side == Side::Upper) ? gridLayout.physicalEndIndex(QtyCentering::dual, direction)
                                    : gridLayout.physicalStartIndex(QtyCentering::dual, direction);
        std::uint32_t const ghost_n = (side == Side::Upper) ? interior_n + 1 : interior_n - 1;

        std::map<TangentialKey, LodiResult> cache;

        auto fillScalarGhosts = [&](field_type& comp, auto pickBdr) {
            auto fieldBox = gridLayout.toFieldBox(localGhostBox, comp.physicalQuantity());
            QtyCentering const centering = GridLayoutT::centering(comp.physicalQuantity())[dir_n];
            for (auto const& ghostIdx : fieldBox)
            {
                auto const& lodi = lookupOrCompute_(cache, ghostIdx, dir_n, sign_n, dn, interior_n,
                                                    ghost_n, ctx);
                auto const mirror
                    = gridLayout.boundaryMirrored(direction, side, centering, ghostIdx);
                comp(ghostIdx) = 2.0 * pickBdr(lodi, ghostIdx) - comp(mirror);
            }
        };

        if constexpr (is_scalar)
        {
            if (field.physicalQuantity() == scalar_quantity_type::rho)
            {
                fillScalarGhosts(field,
                                 [](LodiResult const& l, GhostIdx const&) { return l.rho_bdr; });
            }
            else if (field.physicalQuantity() == scalar_quantity_type::Etot1)
            {
                fillScalarGhosts(field, [&](LodiResult const& l, GhostIdx const& g) {
                    return etotAtBoundary_(g, dir_n, interior_n, ghost_n, l, ctx);
                });
            }
            // other scalars (e.g. P) fall through to whatever earlier fill already wrote.
        }
        else
        {
            if (field.physicalQuantity() == vector_quantity_type::rhoV)
            {
                auto comps = field.components();
                fillScalarGhosts(std::get<0>(comps),
                                 [](LodiResult const& l, GhostIdx const&) { return l.rhoVx_bdr; });
                fillScalarGhosts(std::get<1>(comps),
                                 [](LodiResult const& l, GhostIdx const&) { return l.rhoVy_bdr; });
                fillScalarGhosts(std::get<2>(comps),
                                 [](LodiResult const& l, GhostIdx const&) { return l.rhoVz_bdr; });
            }
        }
    }

protected:
    using GhostIdx      = Point<std::uint32_t, dimension>;
    using TangentialKey = std::array<std::uint32_t, (dimension == 0 ? 1 : dimension)>;

    /// Two tangential axes (right-hand orientation) given the normal axis.
    static std::array<std::size_t, 2> tangentialDirs(std::size_t dir_n)
    {
        switch (dir_n)
        {
            case 0: return {1, 2};
            case 1: return {2, 0};
            case 2: return {0, 1};
        }
        throw std::runtime_error("tangentialDirs: unexpected dir_n");
    }

    std::shared_ptr<Thermo> thermo_;

private:
    /// Pack the tangential coordinates of @p idx (excluding the normal axis) into a key
    /// usable as a std::map index — groups ghost cells aligned along the normal.
    static TangentialKey tangKey_(GhostIdx const& idx, std::size_t dir_n)
    {
        TangentialKey k{};
        std::size_t dir_t = 0;
        for (std::size_t i = 0; i < dimension; ++i)
            if (i != dir_n)
            {
                k[dir_t] = idx[i];
                ++dir_t;
            }
        return k;
    }

    /// Build interior- and ghost-cell index points from the tangential coordinates of
    /// @p ghostIdx and the normal indices given by @p interior_n / @p ghost_n.
    static std::pair<GhostIdx, GhostIdx> pairAt_(GhostIdx const& ghostIdx, std::size_t dir_n,
                                                 std::uint32_t interior_n, std::uint32_t ghost_n)
    {
        GhostIdx interior = ghostIdx;
        GhostIdx ghost    = ghostIdx;
        interior[dir_n]   = interior_n;
        ghost[dir_n]      = ghost_n;
        return {interior, ghost};
    }

    LodiResult const& lookupOrCompute_(std::map<TangentialKey, LodiResult>& cache,
                                       GhostIdx const& ghostIdx, std::size_t dir_n, double sign_n,
                                       double dn, std::uint32_t interior_n, std::uint32_t ghost_n,
                                       typename Super::boundary_condition_context_type const& ctx)
    {
        auto const key = tangKey_(ghostIdx, dir_n);
        if (auto it = cache.find(key); it != cache.end())
            return it->second;

        auto const [interiorIdx, ghostIdxBdr] = pairAt_(ghostIdx, dir_n, interior_n, ghost_n);

        auto& rho_old  = ctx.accessor_old.getField(scalar_quantity_type::rho);
        auto& Etot_old = ctx.accessor_old.getField(scalar_quantity_type::Etot1);
        auto rhoV_old  = ctx.accessor_old.getVecField(vector_quantity_type::rhoV);
        auto B_new     = ctx.accessor_new.getVecField(vector_quantity_type::B1);
        auto rhoVc     = rhoV_old.components();
        auto Bc        = B_new.components();

        auto projectB = [&](GhostIdx const& idx) {
            double const bx = GridLayoutT::template project<GridLayoutT::faceXToCellCenter>(
                std::get<0>(Bc), idx);
            double const by = GridLayoutT::template project<GridLayoutT::faceYToCellCenter>(
                std::get<1>(Bc), idx);
            double const bz = GridLayoutT::template project<GridLayoutT::faceZToCellCenter>(
                std::get<2>(Bc), idx);
            return std::array<double, 3>{bx, by, bz};
        };

        // Pressure reconstructed cell-by-cell from Etot1 + ρ + ρv via EOS — P_old is not
        // guaranteed filled in the ghost layer, Etot1_old IS.
        auto pressureAt = [&](GhostIdx const& idx) {
            double const rho_c = rho_old(idx);
            double const Vx_c  = std::get<0>(rhoVc)(idx) / rho_c;
            double const Vy_c  = std::get<1>(rhoVc)(idx) / rho_c;
            double const Vz_c  = std::get<2>(rhoVc)(idx) / rho_c;
            auto const Bv      = projectB(idx);
            double const e_int = internalEnergyFromTotalEnergy(Etot_old(idx), rho_c, Vx_c, Vy_c,
                                                               Vz_c, Bv[0], Bv[1], Bv[2]);
            thermo_->setState_DU(rho_c, e_int / rho_c);
            return thermo_->pressure();
        };

        double const rho_g = rho_old(ghostIdxBdr);
        double const rho_i = rho_old(interiorIdx);
        std::array<double, 3> const V_g{std::get<0>(rhoVc)(ghostIdxBdr) / rho_g,
                                        std::get<1>(rhoVc)(ghostIdxBdr) / rho_g,
                                        std::get<2>(rhoVc)(ghostIdxBdr) / rho_g};
        std::array<double, 3> const V_i{std::get<0>(rhoVc)(interiorIdx) / rho_i,
                                        std::get<1>(rhoVc)(interiorIdx) / rho_i,
                                        std::get<2>(rhoVc)(interiorIdx) / rho_i};
        double const p_g = pressureAt(ghostIdxBdr);
        double const p_i = pressureAt(interiorIdx);

        // boundary face = arithmetic mean of flanking cells (2nd-order)
        double const rho_b = 0.5 * (rho_g + rho_i);
        std::array<double, 3> const V_b{0.5 * (V_g[0] + V_i[0]), 0.5 * (V_g[1] + V_i[1]),
                                        0.5 * (V_g[2] + V_i[2])};
        double const p_b = 0.5 * (p_g + p_i);

        // Local-frame velocities. u_n is the local-normal projection (positive in the
        // local frame regardless of side). Tangentials stay cartesian — they decouple
        // from the normal sign convention.
        auto const tang          = tangentialDirs(dir_n);
        std::size_t const dir_t1 = tang[0];
        std::size_t const dir_t2 = tang[1];
        double const u_n         = sign_n * V_b[dir_n];
        double const u_t1        = V_b[dir_t1];
        double const u_t2        = V_b[dir_t2];

        thermo_->setState_DP(rho_b, p_b);
        double const c = thermo_->soundSpeed();

        // ∂/∂n along the local normal. The ghost cell sits on the +outward side of the
        // boundary face at every side (Upper and Lower both), so (g − i)/dn is the
        // outward derivative; multiply by frame_sign to flip into the inward frame when
        // the derived BC uses one (inflow).
        auto diff_n = [&](double f_g, double f_i) {
            return static_cast<double>(frame_sign) * (f_g - f_i) / dn;
        };

        LodiInputs inputs;
        inputs.rho_b   = rho_b;
        inputs.p_b     = p_b;
        inputs.c       = c;
        inputs.u       = {u_n, u_t1, u_t2};
        inputs.drho_dn = diff_n(rho_g, rho_i);
        inputs.duN_dn  = diff_n(sign_n * V_g[dir_n], sign_n * V_i[dir_n]);
        inputs.duT1_dn = diff_n(V_g[dir_t1], V_i[dir_t1]);
        inputs.duT2_dn = diff_n(V_g[dir_t2], V_i[dir_t2]);
        inputs.dP_dn   = diff_n(p_g, p_i);

        LodiAmplitudes const L
            = static_cast<Derived const*>(this)->computeAmplitudes_(inputs, dir_n, sign_n);

        // Forward-Euler LODI ODE over ctx.dt — same form for inflow and outflow once the
        // local rotated frame is used.
        double const dt       = ctx.dt;
        double const rho_bdr  = rho_b - dt * (L.L2 + 0.5 * (L.L1 + L.L5)) / (c * c);
        double const u_n_bdr  = u_n - dt * (L.L5 - L.L1) / (2.0 * rho_b * c);
        double const u_t1_bdr = u_t1 - dt * L.L3;
        double const u_t2_bdr = u_t2 - dt * L.L4;
        double const p_bdr    = p_b - dt * 0.5 * (L.L1 + L.L5);

        // Reconstruct cartesian V: only the normal component carries sign_n; tangentials
        // are cartesian throughout.
        std::array<double, 3> V_bdr{};
        V_bdr[dir_n]  = sign_n * u_n_bdr;
        V_bdr[dir_t1] = u_t1_bdr;
        V_bdr[dir_t2] = u_t2_bdr;

        LodiResult r{rho_bdr, rho_bdr * V_bdr[0], rho_bdr * V_bdr[1], rho_bdr * V_bdr[2], p_bdr};
        return cache.emplace(key, r).first->second;
    }

    /// Reconstruct Etot at the boundary face from the LODI-derived primitives + B
    /// projected at cell centers (HD: B = 0 so contribution vanishes; kept for
    /// MHD-extension consistency).
    double etotAtBoundary_(GhostIdx const& ghostIdx, std::size_t dir_n, std::uint32_t interior_n,
                           std::uint32_t ghost_n, LodiResult const& lodi,
                           typename Super::boundary_condition_context_type const& ctx)
    {
        auto B_new = ctx.accessor_new.getVecField(vector_quantity_type::B1);
        auto Bc    = B_new.components();

        auto projectB = [&](GhostIdx const& idx) {
            double const bx = GridLayoutT::template project<GridLayoutT::faceXToCellCenter>(
                std::get<0>(Bc), idx);
            double const by = GridLayoutT::template project<GridLayoutT::faceYToCellCenter>(
                std::get<1>(Bc), idx);
            double const bz = GridLayoutT::template project<GridLayoutT::faceZToCellCenter>(
                std::get<2>(Bc), idx);
            return std::array<double, 3>{bx, by, bz};
        };

        auto const [interiorIdx, ghostIdxBdr] = pairAt_(ghostIdx, dir_n, interior_n, ghost_n);
        auto const B_i                        = projectB(interiorIdx);
        auto const B_g                        = projectB(ghostIdxBdr);
        std::array<double, 3> const B_bdr{0.5 * (B_g[0] + B_i[0]), 0.5 * (B_g[1] + B_i[1]),
                                          0.5 * (B_g[2] + B_i[2])};

        double const Vx = lodi.rhoVx_bdr / lodi.rho_bdr;
        double const Vy = lodi.rhoVy_bdr / lodi.rho_bdr;
        double const Vz = lodi.rhoVz_bdr / lodi.rho_bdr;

        thermo_->setState_DP(lodi.rho_bdr, lodi.p_bdr);
        double const e_int = thermo_->internalEnergy() * lodi.rho_bdr;
        return totalEnergyFromInternalEnergy(e_int, lodi.rho_bdr, Vx, Vy, Vz, B_bdr[0], B_bdr[1],
                                             B_bdr[2]);
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NON_REFLECTING_HYDRO_BASE_BOUNDARY_CONDITION_HPP
