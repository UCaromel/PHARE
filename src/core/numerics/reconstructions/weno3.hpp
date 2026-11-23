#ifndef CORE_NUMERICS_RECONSTRUCTION_WENO3_HPP
#define CORE_NUMERICS_RECONSTRUCTION_WENO3_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
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

    static constexpr double dL0 = 1. / 3., dL1 = 2. / 3.;
    static constexpr double dR0 = 2. / 3., dR1 = 1. / 3.;

    static auto recons_weno3_L_(auto ul, auto u, auto ur)
    {
        auto const [wL0, wL1] = compute_weno3_weights(compute_IS({ul, u, ur}), dL0, dL1);
        return combine_L_(ul, u, ur, wL0, wL1);
    }

    static auto recons_weno3_R_(auto ul, auto u, auto ur)
    {
        auto const [wR0, wR1] = compute_weno3_weights(compute_IS({ul, u, ur}), dR0, dR1);
        return combine_R_(ul, u, ur, wR0, wR1);
    }

    // shared betas are dimensionless and floored above round-off (see Reconstructor::compute_SSI),
    // so they take scale-invariant Z-type weights instead of the absolute-eps JS weights
    static auto recons_weno3_L_(auto ul, auto u, auto ur, Betas const& betas)
    {
        auto const [wL0, wL1] = shared_weights_(betas, dL0, dL1);
        return combine_L_(ul, u, ur, wL0, wL1);
    }

    static auto recons_weno3_R_(auto ul, auto u, auto ur, Betas const& betas)
    {
        auto const [wR0, wR1] = shared_weights_(betas, dR0, dR1);
        return combine_R_(ul, u, ur, wR0, wR1);
    }

    static auto combine_L_(auto ul, auto u, auto ur, auto wL0, auto wL1)
    {
        return wL0 * (-0.5 * ul + 1.5 * u) + wL1 * (0.5 * u + 0.5 * ur);
    }

    static auto combine_R_(auto ul, auto u, auto ur, auto wR0, auto wR1)
    {
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

    // WENO-Z3 (Don & Borges 2013): tau3 = |beta0 - beta1|
    static auto compute_weno3z_weights(Betas const& betas, auto const d0, auto const d1)
    {
        static constexpr auto eps = 1.e-40;

        auto const [beta0, beta1] = betas;
        auto const tau3           = std::abs(beta0 - beta1);

        auto const alpha0 = d0 * (1. + tau3 / (beta0 + eps));
        auto const alpha1 = d1 * (1. + tau3 / (beta1 + eps));

        auto const sum_alpha = alpha0 + alpha1;

        return std::make_tuple(alpha0 / sum_alpha, alpha1 / sum_alpha);
    }

    // EXPERIMENT: PHARE_SSI_W3=JS selects JS weights on the shared path, until frozen
    static auto shared_weights_(Betas const& betas, auto const d0, auto const d1)
    {
        static bool const js = [] {
            auto const* env = std::getenv("PHARE_SSI_W3");
            return env && std::string{env} == "JS";
        }();
        return js ? compute_weno3_weights(betas, d0, d1) : compute_weno3z_weights(betas, d0, d1);
    }
};

} // namespace PHARE::core

#endif
