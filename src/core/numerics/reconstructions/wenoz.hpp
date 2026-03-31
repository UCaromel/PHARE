#ifndef CORE_NUMERICS_RECONSTRUCTION_WENOZ_HPP
#define CORE_NUMERICS_RECONSTRUCTION_WENOZ_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include <array>
#include <utility>

namespace PHARE::core
{
template<typename GridLayout, typename SlopeLimiter = void>
class WENOZReconstruction
{
public:
    // Stencil width: WENOZ scheme needs 3 ghost cells (5th order accuracy)
    // NOTE: Must match ReconstructionNghosts<ReconstructionType::WENOZ>::value
    static constexpr auto nghosts             = 3;
    static constexpr std::size_t N_substencils = 3;
    static constexpr std::size_t stencil_size  = 5; // [i-2, i-1, i, i+1, i+2]

    using GridLayout_t = GridLayout;

    // Compute smoothness indicators from the centered 5-cell stencil.
    // s = {u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}}
    // Sub-stencils: beta0=[i-2,i-1,i], beta1=[i-1,i,i+1], beta2=[i,i+1,i+2]
    static auto compute_IS(std::array<double, stencil_size> const& s)
        -> std::array<double, N_substencils>
    {
        auto const beta0 = (13. / 12.) * (s[0] - 2. * s[1] + s[2]) * (s[0] - 2. * s[1] + s[2])
                           + (1. / 4.) * (s[0] - 4. * s[1] + 3. * s[2])
                                 * (s[0] - 4. * s[1] + 3. * s[2]);
        auto const beta1 = (13. / 12.) * (s[1] - 2. * s[2] + s[3]) * (s[1] - 2. * s[2] + s[3])
                           + (1. / 4.) * (s[1] - s[3]) * (s[1] - s[3]);
        auto const beta2 = (13. / 12.) * (s[2] - 2. * s[3] + s[4]) * (s[2] - 2. * s[3] + s[4])
                           + (1. / 4.) * (3. * s[2] - 4. * s[3] + s[4])
                                 * (3. * s[2] - 4. * s[3] + s[4]);
        return {beta0, beta1, beta2};
    }

    // Reconstruct using field's own IS (original path, unchanged).
    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index)
    {
        auto u_3
            = F(GridLayout::template previous<direction>(GridLayout::template previous<direction>(
                GridLayout::template previous<direction>(index))));
        auto u_2 = F(GridLayout::template previous<direction>(
            GridLayout::template previous<direction>(index)));
        auto u_1 = F(GridLayout::template previous<direction>(index));
        auto u   = F(index);
        auto u1  = F(GridLayout::template next<direction>(index));
        auto u2
            = F(GridLayout::template next<direction>(GridLayout::template next<direction>(index)));

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2));
    }

    // Reconstruct using pre-computed shared smoothness indicators.
    // ssi_L: IS from the cell to the left of the face (sub-stencils for L reconstruction).
    // ssi_R: IS from the cell to the right of the face (sub-stencils for R reconstruction).
    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index,
                            std::array<double, N_substencils> const& ssi_L,
                            std::array<double, N_substencils> const& ssi_R)
    {
        auto u_3
            = F(GridLayout::template previous<direction>(GridLayout::template previous<direction>(
                GridLayout::template previous<direction>(index))));
        auto u_2 = F(GridLayout::template previous<direction>(
            GridLayout::template previous<direction>(index)));
        auto u_1 = F(GridLayout::template previous<direction>(index));
        auto u   = F(index);
        auto u1  = F(GridLayout::template next<direction>(index));
        auto u2
            = F(GridLayout::template next<direction>(GridLayout::template next<direction>(index)));

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1, ssi_L),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2, ssi_R));
    }

    template<auto direction, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index,
                                   auto projection)
    {
        auto u_3 = GridLayout::project(
            U,
            GridLayout::template previous<direction>(GridLayout::template previous<direction>(
                GridLayout::template previous<direction>(index))),
            projection);
        auto u_2 = GridLayout::project(U,
                                       GridLayout::template previous<direction>(
                                           GridLayout::template previous<direction>(index)),
                                       projection);
        auto u_1
            = GridLayout::project(U, GridLayout::template previous<direction>(index), projection);
        auto u  = GridLayout::project(U, index, projection);
        auto u1 = GridLayout::project(U, GridLayout::template next<direction>(index), projection);
        auto u2 = GridLayout::project(
            U, GridLayout::template next<direction>(GridLayout::template next<direction>(index)),
            projection);

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2));
    }

    // center_reconstruct with pre-computed SSI (separate for L and R).
    template<auto direction, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index,
                                   auto projection,
                                   std::array<double, N_substencils> const& ssi_L,
                                   std::array<double, N_substencils> const& ssi_R)
    {
        auto u_3 = GridLayout::project(
            U,
            GridLayout::template previous<direction>(GridLayout::template previous<direction>(
                GridLayout::template previous<direction>(index))),
            projection);
        auto u_2 = GridLayout::project(U,
                                       GridLayout::template previous<direction>(
                                           GridLayout::template previous<direction>(index)),
                                       projection);
        auto u_1
            = GridLayout::project(U, GridLayout::template previous<direction>(index), projection);
        auto u  = GridLayout::project(U, index, projection);
        auto u1 = GridLayout::project(U, GridLayout::template next<direction>(index), projection);
        auto u2 = GridLayout::project(
            U, GridLayout::template next<direction>(GridLayout::template next<direction>(index)),
            projection);

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1, ssi_L),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2, ssi_R));
    }

private:
    // ---------- original per-variable IS paths (unchanged) ----------

    static auto recons_wenoz_L_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr)
    {
        static constexpr auto dL0 = 1. / 10.;
        static constexpr auto dL1 = 3. / 5.;
        static constexpr auto dL2 = 3. / 10.;

        auto const [wL0, wL1, wL2] = compute_wenoz_weights(ull, ul, u, ur, urr, dL0, dL1, dL2);

        return wL0 * ((1. / 3.) * ull - (7. / 6.) * ul + (11. / 6.) * u)
               + wL1 * (-(1. / 6.) * ul + (5. / 6.) * u + (1. / 3.) * ur)
               + wL2 * ((1. / 3.) * u + (5. / 6.) * ur - (1. / 6.) * urr);
    }

    static auto recons_wenoz_R_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr)
    {
        static constexpr auto dR0 = 3. / 10.;
        static constexpr auto dR1 = 3. / 5.;
        static constexpr auto dR2 = 1. / 10.;

        auto const [wR0, wR1, wR2] = compute_wenoz_weights(ull, ul, u, ur, urr, dR0, dR1, dR2);

        return wR0 * ((1. / 3.) * u + (5. / 6.) * ul - (1. / 6.) * ull)
               + wR1 * (-(1. / 6.) * ur + (5. / 6.) * u + (1. / 3.) * ul)
               + wR2 * ((1. / 3.) * urr - (7. / 6.) * ur + (11. / 6.) * u);
    }

    static auto compute_wenoz_weights(auto const ull, auto const ul, auto const u, auto const ur,
                                      auto const urr, auto const d0, auto const d1, auto const d2)
    {
        static constexpr auto eps = 1.e-40;

        auto const beta0 = (13. / 12.) * (ull - 2. * ul + u) * (ull - 2. * ul + u)
                           + (1. / 4.) * (ull - 4. * ul + 3. * u) * (ull - 4. * ul + 3. * u);
        auto const beta1 = (13. / 12.) * (ul - 2. * u + ur) * (ul - 2. * u + ur)
                           + (1. / 4.) * (ul - ur) * (ul - ur);
        auto const beta2 = (13. / 12.) * (u - 2. * ur + urr) * (u - 2. * ur + urr)
                           + (1. / 4.) * (3. * u - 4. * ur + urr) * (3. * u - 4. * ur + urr);

        auto const tau5 = std::abs(beta0 - beta2);

        auto const alpha0 = d0 * (1. + tau5 / (beta0 + eps));
        auto const alpha1 = d1 * (1. + tau5 / (beta1 + eps));
        auto const alpha2 = d2 * (1. + tau5 / (beta2 + eps));

        auto const sum_alpha = alpha0 + alpha1 + alpha2;

        return std::make_tuple(alpha0 / sum_alpha, alpha1 / sum_alpha, alpha2 / sum_alpha);
    }

    // ---------- SSI paths: weights from pre-computed IS ----------

    static auto compute_wenoz_weights_from_IS(double const beta0, double const beta1,
                                              double const beta2, double const d0, double const d1,
                                              double const d2)
    {
        static constexpr auto eps = 1.e-40;

        auto const tau5   = std::abs(beta0 - beta2);
        auto const alpha0 = d0 * (1. + tau5 / (beta0 + eps));
        auto const alpha1 = d1 * (1. + tau5 / (beta1 + eps));
        auto const alpha2 = d2 * (1. + tau5 / (beta2 + eps));

        auto const sum_alpha = alpha0 + alpha1 + alpha2;

        return std::make_tuple(alpha0 / sum_alpha, alpha1 / sum_alpha, alpha2 / sum_alpha);
    }

    static auto recons_wenoz_L_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr, std::array<double, N_substencils> const& ssi)
    {
        static constexpr auto dL0 = 1. / 16.;
        static constexpr auto dL1 = 5. / 8.;
        static constexpr auto dL2 = 5. / 16.;

        auto const [wL0, wL1, wL2]
            = compute_wenoz_weights_from_IS(ssi[0], ssi[1], ssi[2], dL0, dL1, dL2);

        return wL0 * ((3. / 8.) * ull - (10. / 8.) * ul + (15. / 8.) * u)
               + wL1 * (-(1. / 8.) * ul + (6. / 8.) * u + (3. / 8.) * ur)
               + wL2 * ((3. / 8.) * u + (6. / 8.) * ur - (1. / 8.) * urr);
    }

    static auto recons_wenoz_R_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr, std::array<double, N_substencils> const& ssi)
    {
        static constexpr auto dR0 = 5. / 16.;
        static constexpr auto dR1 = 5. / 8.;
        static constexpr auto dR2 = 1. / 16.;

        auto const [wR0, wR1, wR2]
            = compute_wenoz_weights_from_IS(ssi[0], ssi[1], ssi[2], dR0, dR1, dR2);

        return wR0 * ((3. / 8.) * u + (6. / 8.) * ul - (1. / 8.) * ull)
               + wR1 * (-(1. / 8.) * ur + (6. / 8.) * u + (3. / 8.) * ul)
               + wR2 * ((3. / 8.) * urr - (10. / 8.) * ur + (15. / 8.) * u);
    }
};

} // namespace PHARE::core

#endif
