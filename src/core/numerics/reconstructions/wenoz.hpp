#ifndef CORE_NUMERICS_RECONSTRUCTION_WENOZ_HPP
#define CORE_NUMERICS_RECONSTRUCTION_WENOZ_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace PHARE::core
{
template<typename GridLayout, typename SlopeLimiter = void>
class WENOZReconstruction
{
public:
    static constexpr auto nghosts = 3;

    static constexpr std::size_t stencil_size  = 5;
    static constexpr std::size_t N_substencils = 3;

    using Stencil      = std::array<double, stencil_size>;
    using Betas        = std::array<double, N_substencils>;
    using GridLayout_t = GridLayout;

    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index)
    {
        auto const [u_3, u_2, u_1, u, u1, u2]
            = stencil_<direction>([&](auto const& idx) { return F(idx); }, index);

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2));
    }

    // betasL/betasR: indicators of the cells left/right of the face, shared across variables
    template<auto direction, typename Field>
    static auto reconstruct(Field const& F, MeshIndex<Field::dimension> index, Betas const& betasL,
                            Betas const& betasR)
    {
        auto const [u_3, u_2, u_1, u, u1, u2]
            = stencil_<direction>([&](auto const& idx) { return F(idx); }, index);

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1, betasL),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2, betasR));
    }

    template<auto direction, auto src, auto dst, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index)
    {
        auto const [u_3, u_2, u_1, u, u1, u2] = stencil_<direction>(
            [&](auto const& idx) { return GridLayout::template project<src, dst>(U, idx); }, index);

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2));
    }

    template<auto direction, auto src, auto dst, typename Field>
    static auto center_reconstruct(Field const& U, MeshIndex<Field::dimension> index,
                                   Betas const& betasL, Betas const& betasR)
    {
        auto const [u_3, u_2, u_1, u, u1, u2] = stencil_<direction>(
            [&](auto const& idx) { return GridLayout::template project<src, dst>(U, idx); }, index);

        return std::make_pair(recons_wenoz_L_(u_3, u_2, u_1, u, u1, betasL),
                              recons_wenoz_R_(u_2, u_1, u, u1, u2, betasR));
    }

    // 5-cell stencil centred on the cell at index, for compute_IS
    // at: any callable returning the value at a MeshIndex (a Field, or a projecting lambda)
    template<auto direction, std::size_t dim>
    static auto centered_stencil(auto const& at, MeshIndex<dim> index) -> Stencil
    {
        auto const prev = GridLayout::template previous<direction>(index);
        auto const next = GridLayout::template next<direction>(index);
        return {at(GridLayout::template previous<direction>(prev)), at(prev), at(index), at(next),
                at(GridLayout::template next<direction>(next))};
    }

    // smoothness indicators of the three 3-cell substencils of the 5-cell stencil centred on s[2]
    static auto compute_IS(Stencil const& s) -> Betas
    {
        auto const [ull, ul, u, ur, urr] = s;

        auto const beta0 = (13. / 12.) * (ull - 2. * ul + u) * (ull - 2. * ul + u)
                           + (1. / 4.) * (ull - 4. * ul + 3. * u) * (ull - 4. * ul + 3. * u);
        auto const beta1 = (13. / 12.) * (ul - 2. * u + ur) * (ul - 2. * u + ur)
                           + (1. / 4.) * (ul - ur) * (ul - ur);
        auto const beta2 = (13. / 12.) * (u - 2. * ur + urr) * (u - 2. * ur + urr)
                           + (1. / 4.) * (3. * u - 4. * ur + urr) * (3. * u - 4. * ur + urr);

        return {beta0, beta1, beta2};
    }

private:
    // values at cells i-3..i+2 for the face between cells i-1 and i (= index)
    template<auto direction>
    static auto stencil_(auto&& at, auto const& index) -> std::array<double, 6>
    {
        auto const p1 = GridLayout::template previous<direction>(index);
        auto const p2 = GridLayout::template previous<direction>(p1);
        auto const n1 = GridLayout::template next<direction>(index);
        return {at(GridLayout::template previous<direction>(p2)), at(p2), at(p1), at(index), at(n1),
                at(GridLayout::template next<direction>(n1))};
    }

    static auto recons_wenoz_L_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr)
    {
        return recons_wenoz_L_(ull, ul, u, ur, urr, compute_IS({ull, ul, u, ur, urr}));
    }

    static auto recons_wenoz_R_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr)
    {
        return recons_wenoz_R_(ull, ul, u, ur, urr, compute_IS({ull, ul, u, ur, urr}));
    }

    static auto recons_wenoz_L_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr, Betas const& betas)
    {
        static constexpr auto dL0 = 1. / 10.;
        static constexpr auto dL1 = 3. / 5.;
        static constexpr auto dL2 = 3. / 10.;

        auto const [wL0, wL1, wL2] = compute_wenoz_weights(betas, dL0, dL1, dL2);

        return wL0 * ((1. / 3.) * ull - (7. / 6.) * ul + (11. / 6.) * u)
               + wL1 * (-(1. / 6.) * ul + (5. / 6.) * u + (1. / 3.) * ur)
               + wL2 * ((1. / 3.) * u + (5. / 6.) * ur - (1. / 6.) * urr);
    }

    static auto recons_wenoz_R_(auto const ull, auto const ul, auto const u, auto const ur,
                                auto const urr, Betas const& betas)
    {
        static constexpr auto dR0 = 3. / 10.;
        static constexpr auto dR1 = 3. / 5.;
        static constexpr auto dR2 = 1. / 10.;

        auto const [wR0, wR1, wR2] = compute_wenoz_weights(betas, dR0, dR1, dR2);

        return wR0 * ((1. / 3.) * u + (5. / 6.) * ul - (1. / 6.) * ull)
               + wR1 * (-(1. / 6.) * ur + (5. / 6.) * u + (1. / 3.) * ul)
               + wR2 * ((1. / 3.) * urr - (7. / 6.) * ur + (11. / 6.) * u);
    }

    static auto compute_wenoz_weights(Betas const& betas, auto const d0, auto const d1,
                                      auto const d2)
    {
        static constexpr auto eps = 1.e-40;

        auto const [beta0, beta1, beta2] = betas;

        auto const tau5 = std::abs(beta0 - beta2);

        auto const alpha0 = d0 * (1. + tau5 / (beta0 + eps));
        auto const alpha1 = d1 * (1. + tau5 / (beta1 + eps));
        auto const alpha2 = d2 * (1. + tau5 / (beta2 + eps));

        auto const sum_alpha = alpha0 + alpha1 + alpha2;

        return std::make_tuple(alpha0 / sum_alpha, alpha1 / sum_alpha, alpha2 / sum_alpha);
    }
};

} // namespace PHARE::core

#endif
