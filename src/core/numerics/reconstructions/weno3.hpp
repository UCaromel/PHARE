#ifndef CORE_NUMERICS_RECONSTRUCTION_WENO3_HPP
#define CORE_NUMERICS_RECONSTRUCTION_WENO3_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include <array>
#include <utility>

namespace PHARE::core
{
template<typename GridLayout, typename SlopeLimiter = void>
class WENO3Reconstruction
{
public:
    // Stencil width: WENO3 scheme needs 2 ghost cells
    // NOTE: Must match ReconstructionNghosts<ReconstructionType::WENO3>::value
    static constexpr auto nghosts             = 2;
    static constexpr std::size_t N_substencils = 2;
    static constexpr std::size_t stencil_size  = 3; // [i-1, i, i+1]

    using GridLayout_t = GridLayout;

    // Compute smoothness indicators from the centered stencil [i-1, i, i+1].
    // s = {u_{i-1}, u_i, u_{i+1}}
    static auto compute_IS(std::array<double, stencil_size> const& s)
        -> std::array<double, N_substencils>
    {
        return {(s[1] - s[0]) * (s[1] - s[0]), (s[2] - s[1]) * (s[2] - s[1])};
    }

    // Reconstruct using field's own IS (original path, unchanged).
    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index)
    {
        auto u_2 = F(GridLayout::template previous<direction>(
            GridLayout::template previous<direction>(index)));
        auto u_1 = F(GridLayout::template previous<direction>(index));
        auto u   = F(index);
        auto u1  = F(GridLayout::template next<direction>(index));

        return std::make_pair(recons_weno3_L_(u_2, u_1, u), recons_weno3_R_(u_1, u, u1));
    }

    // Reconstruct using pre-computed shared smoothness indicators.
    // ssi_L: IS from cell left of face; ssi_R: IS from cell right of face.
    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index,
                            std::array<double, N_substencils> const& ssi_L,
                            std::array<double, N_substencils> const& ssi_R)
    {
        auto u_2 = F(GridLayout::template previous<direction>(
            GridLayout::template previous<direction>(index)));
        auto u_1 = F(GridLayout::template previous<direction>(index));
        auto u   = F(index);
        auto u1  = F(GridLayout::template next<direction>(index));

        return std::make_pair(recons_weno3_L_(u_2, u_1, u, ssi_L),
                              recons_weno3_R_(u_1, u, u1, ssi_R));
    }

    template<auto direction, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index,
                                   auto projection)
    {
        auto u_2 = GridLayout::project(U,
                                       GridLayout::template previous<direction>(
                                           GridLayout::template previous<direction>(index)),
                                       projection);
        auto u_1
            = GridLayout::project(U, GridLayout::template previous<direction>(index), projection);
        auto u  = GridLayout::project(U, index, projection);
        auto u1 = GridLayout::project(U, GridLayout::template next<direction>(index), projection);

        return std::make_pair(recons_weno3_L_(u_2, u_1, u), recons_weno3_R_(u_1, u, u1));
    }

    // center_reconstruct with pre-computed SSI (separate for L and R).
    template<auto direction, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index,
                                   auto projection,
                                   std::array<double, N_substencils> const& ssi_L,
                                   std::array<double, N_substencils> const& ssi_R)
    {
        auto u_2 = GridLayout::project(U,
                                       GridLayout::template previous<direction>(
                                           GridLayout::template previous<direction>(index)),
                                       projection);
        auto u_1
            = GridLayout::project(U, GridLayout::template previous<direction>(index), projection);
        auto u  = GridLayout::project(U, index, projection);
        auto u1 = GridLayout::project(U, GridLayout::template next<direction>(index), projection);

        return std::make_pair(recons_weno3_L_(u_2, u_1, u, ssi_L),
                              recons_weno3_R_(u_1, u, u1, ssi_R));
    }

private:
    // ---------- original per-variable IS paths (unchanged) ----------

    static auto recons_weno3_L_(auto ul, auto u, auto ur)
    {
        static constexpr auto dL0 = 1. / 3.;
        static constexpr auto dL1 = 2. / 3.;

        auto const [wL0, wL1] = compute_weno3_weights(ul, u, ur, dL0, dL1);

        return wL0 * (-0.5 * ul + 1.5 * u) + wL1 * (0.5 * u + 0.5 * ur);
    }

    static auto recons_weno3_R_(auto ul, auto u, auto ur)
    {
        static constexpr auto dR0 = 2. / 3.;
        static constexpr auto dR1 = 1. / 3.;

        auto const [wR0, wR1] = compute_weno3_weights(ul, u, ur, dR0, dR1);

        return wR0 * (0.5 * u + 0.5 * ul) + wR1 * (-0.5 * ur + 1.5 * u);
    }

    static auto compute_weno3_weights(auto const ul, auto const u, auto const ur, auto const d0,
                                      auto const d1)
    {
        static constexpr auto eps = 1.e-6;

        auto const beta0 = (u - ul) * (u - ul);
        auto const beta1 = (ur - u) * (ur - u);

        auto const alpha0 = d0 / ((beta0 + eps) * (beta0 + eps));
        auto const alpha1 = d1 / ((beta1 + eps) * (beta1 + eps));

        auto const sum_alpha = alpha0 + alpha1;

        return std::make_tuple(alpha0 / sum_alpha, alpha1 / sum_alpha);
    }

    // ---------- SSI paths: weights from pre-computed IS ----------

    static auto compute_weno3_weights_from_IS(double const beta0, double const beta1,
                                              double const d0, double const d1)
    {
        static constexpr auto eps = 1.e-6;

        auto const alpha0 = d0 / ((beta0 + eps) * (beta0 + eps));
        auto const alpha1 = d1 / ((beta1 + eps) * (beta1 + eps));

        auto const sum_alpha = alpha0 + alpha1;

        return std::make_pair(alpha0 / sum_alpha, alpha1 / sum_alpha);
    }

    static auto recons_weno3_L_(auto ul, auto u, auto ur,
                                std::array<double, N_substencils> const& ssi)
    {
        static constexpr auto dL0 = 1. / 3.;
        static constexpr auto dL1 = 2. / 3.;

        auto const [wL0, wL1] = compute_weno3_weights_from_IS(ssi[0], ssi[1], dL0, dL1);

        return wL0 * (-0.5 * ul + 1.5 * u) + wL1 * (0.5 * u + 0.5 * ur);
    }

    static auto recons_weno3_R_(auto ul, auto u, auto ur,
                                std::array<double, N_substencils> const& ssi)
    {
        static constexpr auto dR0 = 2. / 3.;
        static constexpr auto dR1 = 1. / 3.;

        auto const [wR0, wR1] = compute_weno3_weights_from_IS(ssi[0], ssi[1], dR0, dR1);

        return wR0 * (0.5 * u + 0.5 * ul) + wR1 * (-0.5 * ur + 1.5 * u);
    }
};

} // namespace PHARE::core

#endif
