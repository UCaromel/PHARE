#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP

#include "initializer/data_provider.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/types.hpp"
#include "core/numerics/ampere/ampere.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

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

// this probably needs the full number of ghost required by the reconstruction
template<auto direction, size_t dim>
auto getGrow(int const nghosts)
{
    Point<std::uint32_t, dim> p{};

    for (size_t i = 0; i < dim; ++i)
    {
        if (i != static_cast<size_t>(direction))
            p[i] = nghosts;
    }

    return p;
}

template<typename GridLayout, template<typename> typename Reconstruction,
         template<typename> typename RiemannSolver, typename Equations>
class Godunov_ref;

template<typename GridLayout, template<typename> typename Reconstruction,
         template<typename> typename RiemannSolver, typename Equations>
class Godunov : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<typename T>
    using Rec = Reconstruction<T>;

    constexpr static auto Hall             = Equations::hall;
    constexpr static auto Resistivity      = Equations::resistivity;
    constexpr static auto HyperResistivity = Equations::hyperResistivity;

    Godunov(PHARE::initializer::PHAREDict const& dict)
        : gamma_{dict["heat_capacity_ratio"].template to<double>()}
        , eta_{dict["resistivity"].template to<double>()}
        , nu_{dict["hyper_resistivity"].template to<double>()}
    {
    }

    template<typename State, typename... Fluxes>
    void operator()(auto& ct, State& state, Fluxes&... fluxes)
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - GodunovFluxes - GridLayout not set, cannot proceed "
                                     "to reconstruction");

        Godunov_ref<GridLayout, Reconstruction, RiemannSolver, Equations>{
            *this->layout_,
            gamma_,
            eta_,
            nu_,
        }(ct, state, fluxes...);
    }

private:
    double const gamma_;
    double const eta_;
    double const nu_;
};


template<typename GridLayout, template<typename> typename Reconstruction,
         template<typename> typename RiemannSolver, typename Equations>
class Godunov_ref
{
    constexpr static auto dimension = GridLayout::dimension;
    using Reconstruction_t          = Reconstruction<GridLayout>;
    using Reconstructor_t           = Reconstructor<Reconstruction_t>;
    using RiemannSolver_t           = RiemannSolver<GridLayout>;

    constexpr static auto Hall             = Equations::hall;
    constexpr static auto Resistivity      = Equations::resistivity;
    constexpr static auto HyperResistivity = Equations::hyperResistivity;

public:
    Godunov_ref(GridLayout const& layout, double const gamma, double const eta, double const nu)
        : layout_{layout}
        , gamma_{gamma}
        , eta_{eta}
        , nu_{nu}
        , equations_{gamma_, eta_, nu_}
        , riemann_{layout, gamma}

    {
    }

    template<typename State, typename Fluxes>
    void operator()(auto& ct, State& state, Fluxes& fluxes)
    {
        constexpr auto directions = getDirections<dimension>();

        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_.evalOnBiggerBox(
                fluxes.template expose_centering<direction>(),
                getGrow<direction, dimension>(Reconstruction_t::nghosts), [&](auto&... indices) {
                    if constexpr (Hall || Resistivity || HyperResistivity)
                    {
                        auto&& [uL, uR]
                            = Reconstructor_t::template reconstruct<direction>(state, {indices...});

                        auto const& [jL, jR]
                            = Reconstructor_t::template center_reconstruct<direction>(
                                state.J, GridLayout::edgeXToCellCenter(),
                                GridLayout::edgeYToCellCenter(), GridLayout::edgeZToCellCenter(),
                                {indices...});

                        auto&& u      = std::forward_as_tuple(uL, uR);
                        auto const& j = std::forward_as_tuple(jL, jR);

                        if constexpr (HyperResistivity)
                        {
                            auto const& [laplJL, laplJR]
                                = Reconstructor_t::template reconstructed_laplacian<direction>(
                                    layout_.inverseMeshSize(), state.J, {indices...});

                            auto const& LaplJ = std::forward_as_tuple(laplJL, laplJR);

                            auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                                return equations_.template compute<direction>(
                                    std::get<i>(u), std::get<i>(j), std::get<i>(LaplJ));
                            });

                            fluxes.template get_dir<direction>({indices...})
                                = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);

                            ct.template save<direction>(uL, uR, riemann_.vt, riemann_.jt,
                                                        riemann_.rhot, riemann_.uct_coefs_,
                                                        {indices...});
                        }
                        else
                        {
                            auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                                return equations_.template compute<direction>(std::get<i>(u),
                                                                              std::get<i>(j));
                            });

                            if constexpr (Hall)
                            {
                                fluxes.template get_dir<direction>({indices...})
                                    = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);

                                ct.template save<direction>(uL, uR, riemann_.vt, riemann_.jt,
                                                            riemann_.rhot, riemann_.uct_coefs_,
                                                            {indices...});
                            }
                            else // Resistivity only
                            {
                                fluxes.template get_dir<direction>({indices...})
                                    = riemann_.template solve<direction>(uL, uR, fL, fR);

                                ct.template save<direction>(uL, uR, riemann_.vt,
                                                            riemann_.uct_coefs_, {indices...});
                            }
                        }
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

                        ct.template save<direction>(uL, uR, riemann_.vt, riemann_.uct_coefs_,
                                                    {indices...});
                    }
                });
        });
    }


private:
    GridLayout layout_;
    double const gamma_;
    double const eta_;
    double const nu_;
    Equations equations_;
    RiemannSolver_t riemann_;
};

} // namespace PHARE::core

#endif
