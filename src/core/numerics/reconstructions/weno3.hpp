#ifndef CORE_NUMERICS_RECONSTRUCTION_WENO3_HPP
#define CORE_NUMERICS_RECONSTRUCTION_WENO3_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace PHARE::core
{
template<typename GridLayout, typename SlopeLimiter = void>
class WENO3Reconstruction
{
public:
    static constexpr auto nghosts = 2;

    static constexpr std::size_t stencil_size  = 3;
    static constexpr std::size_t N_substencils = 2;

    using Stencil      = std::array<double, stencil_size>;
    using Betas        = std::array<double, N_substencils>;
    using GridLayout_t = GridLayout;

    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index)
    {
        auto const [u_2, u_1, u, u1]
            = stencil_<direction>([&](auto const& idx) { return F(idx); }, index);

        return std::make_pair(recons_weno3_L_(u_2, u_1, u), recons_weno3_R_(u_1, u, u1));
    }

    // betasL/betasR: indicators of the cells left/right of the face, shared across variables
    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index, Betas const& betasL,
                            Betas const& betasR)
    {
        auto const [u_2, u_1, u, u1]
            = stencil_<direction>([&](auto const& idx) { return F(idx); }, index);

        return std::make_pair(recons_weno3_L_(u_2, u_1, u, betasL),
                              recons_weno3_R_(u_1, u, u1, betasR));
    }

    template<auto direction, auto src, auto dst, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index)
    {
        auto const [u_2, u_1, u, u1] = stencil_<direction>(
            [&](auto const& idx) { return GridLayout::template project<src, dst>(U, idx); }, index);

        return std::make_pair(recons_weno3_L_(u_2, u_1, u), recons_weno3_R_(u_1, u, u1));
    }

    template<auto direction, auto src, auto dst, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index,
                                   Betas const& betasL, Betas const& betasR)
    {
        auto const [u_2, u_1, u, u1] = stencil_<direction>(
            [&](auto const& idx) { return GridLayout::template project<src, dst>(U, idx); }, index);

        return std::make_pair(recons_weno3_L_(u_2, u_1, u, betasL),
                              recons_weno3_R_(u_1, u, u1, betasR));
    }

    // 3-cell stencil centred on the cell at index, for compute_IS
    // at: any callable returning the value at a MeshIndex (a Field, or a projecting lambda)
    template<auto direction, std::size_t dim>
    static auto centered_stencil(auto const& at, MeshIndex<dim> index) -> Stencil
    {
        return {at(GridLayout::template previous<direction>(index)), at(index),
                at(GridLayout::template next<direction>(index))};
    }

    // smoothness indicators of the two 2-cell substencils of the 3-cell stencil centred on s[1]
    static auto compute_IS(Stencil const& s) -> Betas
    {
        auto const [ul, u, ur] = s;
        return {(u - ul) * (u - ul), (ur - u) * (ur - u)};
    }

private:
    // values at cells i-2..i+1 for the face between cells i-1 and i (= index)
    template<auto direction>
    static auto stencil_(auto&& at, auto const& index) -> std::array<double, 4>
    {
        auto const p1 = GridLayout::template previous<direction>(index);
        return {at(GridLayout::template previous<direction>(p1)), at(p1), at(index),
                at(GridLayout::template next<direction>(index))};
    }

    static auto recons_weno3_L_(auto ul, auto u, auto ur)
    {
        return recons_weno3_L_(ul, u, ur, compute_IS({ul, u, ur}));
    }

    static auto recons_weno3_R_(auto ul, auto u, auto ur)
    {
        return recons_weno3_R_(ul, u, ur, compute_IS({ul, u, ur}));
    }

    static auto recons_weno3_L_(auto ul, auto u, auto ur, Betas const& betas)
    {
        static constexpr auto dL0 = 1. / 3.;
        static constexpr auto dL1 = 2. / 3.;

        auto const [wL0, wL1] = compute_weno3_weights(betas, dL0, dL1);

        return wL0 * (-0.5 * ul + 1.5 * u) + wL1 * (0.5 * u + 0.5 * ur);
    }

    static auto recons_weno3_R_(auto ul, auto u, auto ur, Betas const& betas)
    {
        static constexpr auto dR0 = 2. / 3.;
        static constexpr auto dR1 = 1. / 3.;

        auto const [wR0, wR1] = compute_weno3_weights(betas, dR0, dR1);

        return wR0 * (0.5 * u + 0.5 * ul) + wR1 * (-0.5 * ur + 1.5 * u);
    }

    static auto compute_weno3_weights(Betas const& betas, auto const d0, auto const d1)
    {
        static constexpr auto eps = 1.e-6;

        auto const [beta0, beta1] = betas;

        auto const alpha0 = d0 / ((beta0 + eps) * (beta0 + eps));
        auto const alpha1 = d1 / ((beta1 + eps) * (beta1 + eps));

        auto const sum_alpha = alpha0 + alpha1;

        return std::make_tuple(alpha0 / sum_alpha, alpha1 / sum_alpha);
    }
};

} // namespace PHARE::core

#endif
