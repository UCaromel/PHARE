#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP

#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/types.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"

#include "initializer/data_provider.hpp"

#include <cmath>
#include <tuple>
#include <cstddef>
#include <cstdint>

namespace PHARE::core
{
template<size_t dim>
constexpr auto getDirections()
{
    if constexpr (dim == 1)
    {
        return std::make_tuple(Direction::X);
    }
    else if constexpr (dim == 2)
    {
        return std::make_tuple(Direction::X, Direction::Y);
    }
    else if constexpr (dim == 3)
    {
        return std::make_tuple(Direction::X, Direction::Y, Direction::Z);
    }
}

template<auto direction, size_t dim, bool HyperResistivity>
auto getGrow(int const nghosts)
{
    Point<std::uint32_t, dim> p{};

    auto dir = static_cast<size_t>(direction);

    for (size_t i = 0; i < dim; ++i)
        if (i != dir)
            p[i] = nghosts;

    return p;
}

struct GodunovInfo : public OhmInfo
{
    double const gamma;

    GodunovInfo static FROM(initializer::PHAREDict const& dict)
    {
        return {{OhmInfo::FROM(dict)}, dict["heat_capacity_ratio"].template to<double>()};
    }
};


template<typename GridLayout, template<typename> typename Reconstruction, typename RiemannSolver,
         typename Equations>
class Godunov : public GodunovInfo
{
    using Super                     = GodunovInfo;
    using Reconstruction_t          = Reconstruction<GridLayout>;
    using Reconstructor_t           = Reconstructor<Reconstruction_t>;
    using RiemannSolver_t           = RiemannSolver;
    constexpr static auto dimension = GridLayout::dimension;

public:
    using Info_t      = Super;
    using Equations_t = Equations;

    template<typename T>
    using Rec = Reconstruction<T>;

    constexpr static auto Hall             = Equations::hall;
    constexpr static auto Resistivity      = Equations::resistivity;
    constexpr static auto HyperResistivity = Equations::hyperResistivity;

    explicit Godunov(GodunovInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
        , equations_{gamma, eta, nu}
        , riemann_{gamma}
    {
    }

    template<typename State, typename Fluxes>
    void operator()(auto& ct_state, State& state, Fluxes& fluxes)
    {
        constexpr auto directions = getDirections<dimension>();

        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_.evalOnBiggerBox(
                fluxes.template expose_centering<direction>(),
                getGrow<direction, dimension, HyperResistivity>(Reconstruction_t::nghosts),
                [&](auto&... indices) {
                    if constexpr (Hall || Resistivity || HyperResistivity)
                    {
                        auto&& [uL, uR]
                            = Reconstructor_t::template reconstruct<direction>(state, {indices...});

                        auto const& [jL, jR] = Reconstructor_t::template center_reconstruct<
                            direction, GridLayout::implT::edgeXToCellCenter,
                            GridLayout::implT::edgeYToCellCenter,
                            GridLayout::implT::edgeZToCellCenter>(state.J, {indices...});

                        auto&& u      = std::forward_as_tuple(uL, uR);
                        auto const& j = std::forward_as_tuple(jL, jR);


                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u),
                                                                          std::get<i>(j));
                        });

                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);

                        ct_state.template save<direction>(riemann_.vt, riemann_.jt, riemann_.rhot,
                                                          riemann_.uct_coefs, {indices...});
                    }
                    else // Ideal
                    {
                        auto&& [uL, uR]
                            = Reconstructor_t::template reconstruct<direction>(state, {indices...});

                        auto&& u = std::forward_as_tuple(uL, uR);

                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u));
                        });

                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR);

                        ct_state.template save<direction>(riemann_.vt, riemann_.uct_coefs,
                                                          {indices...});
                    }
                });

            // Non-ideal (resistive + hyper-resistive) flux contributions, taken from J at its
            // native edge location: the E_diss x B product is formed on the edge and projected
            // to the face. This keeps the Laplacian stencil confined to the edge-J reconstruction
            // width, so it never needs J past the transverse-grow layer.
            if constexpr (Resistivity || HyperResistivity)
            {
                layout_.evalOnBox(
                    fluxes.template expose_centering<direction>(), [&](auto&... indices) {
                        MeshIndex<dimension> idx{indices...};
                        auto F       = fluxes.template get_dir<direction>({indices...});
                        auto& F_B    = F.B;
                        auto& F_Etot = F.Etot();

                        non_ideal_flux_contributions_<direction>(state, ct_state, idx, F_B, F_Etot);
                    });
            }
        });
    }

private:
    // Small local projector: sums a per-edge functor over the weight points of a compile-time
    // edge<->face (or B->edge) stencil. Mirrors GridLayout::project, but takes a functor instead
    // of a Field so it can project products (e.g. E_diss * B) formed at the edge before summing -
    // GridLayout::project is intentionally not widened for this, per established convention.
    template<auto Stencil>
    auto proj_(MeshIndex<dimension> idx, auto&& at_edge) const
    {
        auto constexpr wps = Stencil();
        double r           = 0.;
        for (auto const& wp : wps)
            r += wp.coef * at_edge(idx + wp.indexes);
        return r;
    }

    auto minMeshSize_() const
    {
        auto const meshSize = layout_.meshSize();
        if constexpr (dimension == 1)
            return meshSize[0];
        else if constexpr (dimension == 2)
            return std::min({meshSize[0], meshSize[1]});
        else
            return std::min({meshSize[0], meshSize[1], meshSize[2]});
    }

    // direction's two transverse components ("first"/"second", in cyclic X->Y->Z->X order) each
    // live on their own edge (edge-first, edge-second). EdgeFirstToFace/EdgeSecondToFace project
    // those edges onto direction's face; BSecondToEFirst/BFirstToESecond project the *other*
    // transverse B component onto the opposite edge, so the E_diss*B product can be formed
    // pointwise on a single edge before being projected (projection does not commute with
    // multiplication).
    template<auto direction, auto EdgeFirstToFace, auto EdgeSecondToFace, auto BSecondToEFirst,
             auto BFirstToESecond>
    void non_ideal_face_contribution_(auto const& state, auto const& ct_state,
                                      MeshIndex<dimension> idx, auto const& Jfirst,
                                      auto const& Jsecond, auto const& Bfirst, auto const& Bsecond,
                                      auto const& Bnormal, auto& F_B, auto& F_Etot) const
    {
        auto Bfirst_e = [&](MeshIndex<dimension> e) {
            return GridLayout::template project<BFirstToESecond>(Bfirst, e);
        };
        auto Bsecond_e = [&](MeshIndex<dimension> e) {
            return GridLayout::template project<BSecondToEFirst>(Bsecond, e);
        };

        double coef = 0.;
        if constexpr (HyperResistivity)
        {
            if (hyper_mode == HyperMode::constant)
                coef = nu;
            else if (hyper_mode == HyperMode::spatial)
            {
                auto const BnFace      = Bnormal(idx);
                auto const BfirstFace  = proj_<EdgeSecondToFace>(idx, Bfirst_e);
                auto const BsecondFace = proj_<EdgeFirstToFace>(idx, Bsecond_e);
                auto const b           = std::sqrt(BnFace * BnFace + BfirstFace * BfirstFace
                                                   + BsecondFace * BsecondFace);
                auto const rhot        = ct_state.template getRhot<direction>()(idx);
                auto const meshSize    = minMeshSize_();
                coef                   = nu * meshSize * meshSize * (b / rhot + 1);
            }
            else
                throw std::runtime_error("Error - Godunov - unknown hyper_mode");
        }

        auto diss_first = [&](MeshIndex<dimension> e) {
            double v = 0.;
            if constexpr (Resistivity)
                v += eta * Jfirst(e);
            if constexpr (HyperResistivity)
                v -= coef * layout_.laplacian(Jfirst, e);
            return v;
        };
        auto diss_second = [&](MeshIndex<dimension> e) {
            double v = 0.;
            if constexpr (Resistivity)
                v += eta * Jsecond(e);
            if constexpr (HyperResistivity)
                v -= coef * layout_.laplacian(Jsecond, e);
            return v;
        };

        auto const projFirst  = proj_<EdgeFirstToFace>(idx, diss_first);
        auto const projSecond = proj_<EdgeSecondToFace>(idx, diss_second);

        auto const crossFirst = proj_<EdgeFirstToFace>(
            idx, [&](MeshIndex<dimension> e) { return diss_first(e) * Bsecond_e(e); });
        auto const crossSecond = proj_<EdgeSecondToFace>(
            idx, [&](MeshIndex<dimension> e) { return diss_second(e) * Bfirst_e(e); });

        equations_.template resistive_contributions<direction>(projFirst, projSecond, crossFirst,
                                                               crossSecond, F_B, F_Etot);
    }

    template<auto direction>
    void non_ideal_flux_contributions_(auto const& state, auto const& ct_state,
                                       MeshIndex<dimension> idx, auto& F_B, auto& F_Etot) const
    {
        if constexpr (direction == Direction::X)
            non_ideal_face_contribution_<direction, GridLayout::implT::edgeYToFaceX,
                                         GridLayout::implT::edgeZToFaceX, GridLayout::BzToEy,
                                         GridLayout::ByToEz>(
                state, ct_state, idx, state.J(Component::Y), state.J(Component::Z),
                state.B(Component::Y), state.B(Component::Z), state.B(Component::X), F_B, F_Etot);
        else if constexpr (direction == Direction::Y)
            non_ideal_face_contribution_<direction, GridLayout::implT::edgeZToFaceY,
                                         GridLayout::implT::edgeXToFaceY, GridLayout::BxToEz,
                                         GridLayout::BzToEx>(
                state, ct_state, idx, state.J(Component::Z), state.J(Component::X),
                state.B(Component::Z), state.B(Component::X), state.B(Component::Y), F_B, F_Etot);
        else if constexpr (direction == Direction::Z)
            non_ideal_face_contribution_<direction, GridLayout::implT::edgeXToFaceZ,
                                         GridLayout::implT::edgeYToFaceZ, GridLayout::ByToEx,
                                         GridLayout::BxToEy>(
                state, ct_state, idx, state.J(Component::X), state.J(Component::Y),
                state.B(Component::X), state.B(Component::Y), state.B(Component::Z), F_B, F_Etot);
    }

    GridLayout layout_;
    Equations equations_;
    RiemannSolver_t riemann_;
};

} // namespace PHARE::core

#endif
