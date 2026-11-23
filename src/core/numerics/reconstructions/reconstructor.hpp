#ifndef PHARE_CORE_NUMERICS_RECONSTRUCTIONS_RECONSTRUCTOR_HPP
#define PHARE_CORE_NUMERICS_RECONSTRUCTIONS_RECONSTRUCTOR_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/reconstructions/ssi_experiment.hpp" // EXPERIMENT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace PHARE::core
{
// WENO-type schemes whose nonlinear weights can be driven by externally supplied smoothness
// indicators (betas), shared by all the variables reconstructed at a face
template<typename Rec>
concept HasSharedSmoothness = requires {
    Rec::N_substencils;
    Rec::stencil_size;
};

template<typename Reconstruction>
struct Reconstructor
{
public:
    using GridLayout = Reconstruction::GridLayout_t;
    using Scalar     = GridLayout::Scalar;

    // betas...: empty for per-variable weights, or (betasL, betasR) shared across variables
    template<auto direction, typename State>
    static auto reconstruct(State const& S, MeshIndex<GridLayout::dimension> index,
                            auto const&... betas)
    {
        auto [rhoL, rhoR] = Reconstruction::template reconstruct<direction>(S.rho, index, betas...);
        auto [VxL, VxR]
            = Reconstruction::template reconstruct<direction>(S.V(Component::X), index, betas...);
        auto [VyL, VyR]
            = Reconstruction::template reconstruct<direction>(S.V(Component::Y), index, betas...);
        auto [VzL, VzR]
            = Reconstruction::template reconstruct<direction>(S.V(Component::Z), index, betas...);
        auto [PL, PR] = Reconstruction::template reconstruct<direction>(S.P, index, betas...);

        auto [BL, BR] = transverse_reconstruct<direction>(S.B, index, betas...);

        PerIndex uL{rhoL, {VxL, VyL, VzL}, BL, PL};
        PerIndex uR{rhoR, {VxR, VyR, VzR}, BR, PR};

        return std::make_pair(uL, uR);
    }

    template<auto direction, auto srcX, auto srcY, auto srcZ, auto dst, typename VecField>
    static auto center_reconstruct(VecField const& U, MeshIndex<VecField::dimension> index,
                                   auto const&... betas)
    {
        auto const& Ux = U(Component::X);
        auto const& Uy = U(Component::Y);
        auto const& Uz = U(Component::Z);

        auto [UxL, UxR] = Reconstruction::template center_reconstruct<direction, srcX, dst>(
            Ux, index, betas...);
        auto [UyL, UyR] = Reconstruction::template center_reconstruct<direction, srcY, dst>(
            Uy, index, betas...);
        auto [UzL, UzR] = Reconstruction::template center_reconstruct<direction, srcZ, dst>(
            Uz, index, betas...);

        return std::make_tuple(PerIndexVector{UxL, UyL, UzL}, PerIndexVector{UxR, UyR, UzR});
    }

    template<auto direction>
    static constexpr auto faceQuantity()
    {
        if constexpr (direction == Direction::X)
            return Scalar::Bx;
        else if constexpr (direction == Direction::Y)
            return Scalar::By;
        else if constexpr (direction == Direction::Z)
            return Scalar::Bz;
    }

    // The normal direction for B is already face centered, so we only reconstruct the transverse
    template<auto direction, typename VecField>
    static auto transverse_reconstruct(VecField const& B, MeshIndex<VecField::dimension> index,
                                       auto const&... betas)
    {
        auto constexpr transverse = []() { // probably should be a util function somewhere
            if constexpr (direction == Direction::X)
                return std::array{Direction::Y, Direction::Z};
            else if constexpr (direction == Direction::Y)
                return std::array{Direction::X, Direction::Z};
            else if constexpr (direction == Direction::Z)
                return std::array{Direction::X, Direction::Y};
        }();


        auto const Bn  = B(static_cast<Component>(direction));
        auto const Bt0 = B(static_cast<Component>(transverse[0]));
        auto const Bt1 = B(static_cast<Component>(transverse[1]));

        auto [Bt0L, Bt0R] = Reconstruction::template center_reconstruct<
            direction, Reconstructor::template faceQuantity<transverse[0]>(), Scalar::rho>(
            Bt0, index, betas...);
        auto [Bt1L, Bt1R] = Reconstruction::template center_reconstruct<
            direction, Reconstructor::template faceQuantity<transverse[1]>(), Scalar::rho>(
            Bt1, index, betas...);

        PerIndexVector<typename VecField::value_type> BL, BR;
        BL(direction)     = Bn(index);
        BR(direction)     = Bn(index);
        BL(transverse[0]) = Bt0L;
        BR(transverse[0]) = Bt0R;
        BL(transverse[1]) = Bt1L;
        BR(transverse[1]) = Bt1R;

        return std::make_pair(BL, BR);
    }

    // Shared smoothness indicators of the cell at index for faces normal to direction. Each
    // reference variable's betas are made dimensionless by a squared scale summed over the
    // stencil; the shared beta of a substencil is the largest of them, so a substencil counts as
    // smooth only if it is smooth for every reference variable.
    template<auto direction, typename State>
        requires HasSharedSmoothness<Reconstruction>
    static auto compute_SSI(State const& S, MeshIndex<GridLayout::dimension> index)
    {
        using Stencil = typename Reconstruction::Stencil;

        static constexpr double eps = 1.e-40;

        // Normalised betas of a uniform state are round-off, ~(machine epsilon)^2 = 5e-32, which
        // the nonlinear weights would turn into a random stencil choice. Below this floor a
        // substencil counts as smooth: equal floored betas give the linear weights.
        static constexpr double roundoff_floor = 1.e-28;

        auto const& opts = ssi_experiment::options(); // EXPERIMENT

        typename Reconstruction::Betas ssi;
        ssi.fill(roundoff_floor);

        auto stencil = [&](auto const& at) {
            return Reconstruction::template centered_stencil<direction>(at, index);
        };
        auto squares = [](Stencil s) {
            for (auto& v : s)
                v *= v;
            return s;
        };
        auto add = [&](Stencil const& s, Stencil const& scale2) {
            auto const betas = Reconstruction::compute_IS(s);
            double norm2     = eps;
            for (auto const v : scale2)
                norm2 += v;
            for (std::size_t m = 0; m < ssi.size(); ++m)
                ssi[m] = std::max(ssi[m], betas[m] / norm2);
        };

        auto const rho = stencil(S.rho);
        auto const P   = stencil(S.P);
        add(rho, squares(rho));
        add(P, squares(P));

        if (opts.B)
        {
            // B is face-centred: its stencil is taken on cell centres, like the B reconstruction
            auto const Bc = [&]<auto component>() {
                auto const& F = S.B(component);
                return stencil([&](auto const& i) {
                    return GridLayout::template project<
                        faceQuantity<static_cast<Direction>(component)>(), Scalar::rho>(F, i);
                });
            };
            std::array<Stencil, 3> const B{Bc.template operator()<Component::X>(),
                                           Bc.template operator()<Component::Y>(),
                                           Bc.template operator()<Component::Z>()};
            // energy scale: |B|^2 + 2p, magnetic and thermal pressure, never vanishes
            Stencil b2{};
            for (std::size_t k = 0; k < b2.size(); ++k)
                b2[k] = B[0][k] * B[0][k] + B[1][k] * B[1][k] + B[2][k] * B[2][k] + 2. * P[k];
            for (auto const& Bi : B)
                add(Bi, opts.energy ? b2 : squares(Bi));
        }

        if (opts.V)
        {
            std::array<Stencil, 3> const V{stencil(S.V(Component::X)), stencil(S.V(Component::Y)),
                                           stencil(S.V(Component::Z))};
            // energy scale: |V|^2 + p/rho, kinetic and thermal, never vanishes
            Stencil v2{};
            for (std::size_t k = 0; k < v2.size(); ++k)
                v2[k] = V[0][k] * V[0][k] + V[1][k] * V[1][k] + V[2][k] * V[2][k] + P[k] / rho[k];
            for (auto const& Vi : V)
                add(Vi, opts.energy ? v2 : squares(Vi));
        }

        return ssi;
    }
};
} // namespace PHARE::core

#endif
