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

    // add one extra layer in the direction of the flux laplacian computation. Maybe some later
    // optimisation would let us just compute for uct and have the extra layer only reconstructed
    // for j
    if constexpr (HyperResistivity)
        p[dir] += 1;

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


template<typename Model, typename Equations>
class GodunovState
{
    using GridLayout                       = Model::gridlayout_type;
    constexpr static auto dimension        = GridLayout::dimension;
    constexpr static auto Resistivity      = Equations::resistivity;
    constexpr static auto HyperResistivity = Equations::hyperResistivity;

public:
    GodunovState() = default;


    void registerResources(Model& model)
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            model.resourcesManager->registerResources(bt_x);
            if constexpr (dimension >= 2)
            {
                model.resourcesManager->registerResources(bt_y);
                if constexpr (dimension == 3)
                {
                    model.resourcesManager->registerResources(bt_z);
                }
            }
        }
    }

    void allocate(Model& model, auto& patch, double const allocateTime) const
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            model.resourcesManager->allocate(bt_x, patch, allocateTime);
            if constexpr (dimension >= 2)
            {
                model.resourcesManager->allocate(bt_y, patch, allocateTime);
                if constexpr (dimension == 3)
                {
                    model.resourcesManager->allocate(bt_z, patch, allocateTime);
                }
            }
        }
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            if constexpr (dimension == 1)
                return std::forward_as_tuple(bt_x);

            else if constexpr (dimension == 2)
                return std::forward_as_tuple(bt_x, bt_y);

            else if constexpr (dimension == 3)
                return std::forward_as_tuple(bt_x, bt_y, bt_z);
        }
        else
            return std::forward_as_tuple();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            if constexpr (dimension == 1)
                return std::forward_as_tuple(bt_x);

            else if constexpr (dimension == 2)
                return std::forward_as_tuple(bt_x, bt_y);

            else if constexpr (dimension == 3)
                return std::forward_as_tuple(bt_x, bt_y, bt_z);
        }
        else
            return std::forward_as_tuple();
    }


    Model::vecfield_type bt_x{"b_t_x", MHDQuantity::Vector::VecFlux_x};
    Model::vecfield_type bt_y{"b_t_y", MHDQuantity::Vector::VecFlux_y};
    Model::vecfield_type bt_z{"b_t_z", MHDQuantity::Vector::VecFlux_z};
};

template<typename GridLayout, typename Model, template<typename> typename Reconstruction,
         typename RiemannSolver, typename Equations>
class Godunov : public GodunovInfo
{
    using Super                     = GodunovInfo;
    using Reconstruction_t          = Reconstruction<GridLayout>;
    using Reconstructor_t           = Reconstructor<Reconstruction_t>;
    using RiemannSolver_t           = RiemannSolver;
    constexpr static auto dimension = GridLayout::dimension;

public:
    using Info_t  = Super;
    using State_t = GodunovState<Model, Equations>;

    State_t state{};

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
    void operator()(auto& ct, State& state, Fluxes& fluxes)
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
                            direction, GridLayout::edgeXToCellCenter, GridLayout::edgeYToCellCenter,
                            GridLayout::edgeZToCellCenter>(state.J, {indices...});

                        auto&& u      = std::forward_as_tuple(uL, uR);
                        auto const& j = std::forward_as_tuple(jL, jR);


                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u),
                                                                          std::get<i>(j));
                        });

                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);

                        ct.template save<direction>(riemann_.vt, riemann_.jt, riemann_.rhot,
                                                    riemann_.uct_coefs, {indices...});

                        // for energy ExB term
                        if constexpr (Resistivity || HyperResistivity)
                            save_tranverse_magnetic_field_<direction>(uL, uR, {indices...});
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

                        ct.template save<direction>(riemann_.vt, riemann_.uct_coefs, {indices...});

                        // for energy ExB term
                        if constexpr (Resistivity)
                            save_tranverse_magnetic_field_<direction>(uL, uR, {indices...});
                    }
                });

            // adding resistive contributions to energy taking advantage of the already computed jt
            // fluxes for the laplacian computation. This probably doesn't need the grow as the
            // required quantities for ct are already saved.
            if constexpr (Resistivity || HyperResistivity)
            {
                layout_.evalOnBox(
                    fluxes.template expose_centering<direction>(), [&](auto&... indices) {
                        auto& Jt      = ct.template getJt<direction>();
                        auto& Bt      = getBt_<direction>();
                        auto const& F = fluxes.template get_dir<direction>({indices...});
                        auto& F_B     = F.B;
                        auto& F_Etot  = F.Etot();

                        auto const& Btidx = toPerIndexVector(Bt, {indices...});

                        if constexpr (Resistivity)
                        {
                            // transverse B field components (probably a riemann operation).
                            auto const& Jtidx = toPerIndexVector(Jt, {indices...});
                            equations_.template resistive_contributions<direction>(
                                eta, Btidx, Jtidx, F_B, F_Etot);
                        }
                        if constexpr (HyperResistivity)
                        {
                            auto const vecLaplJ
                                = transverse_laplacian_<direction>(Jt, {indices...});

                            if (hyper_mode == HyperMode::constant)
                                return constant_hyperresistive_<direction>(Btidx, vecLaplJ, F_B,
                                                                           F_Etot);
                            else if (hyper_mode == HyperMode::spatial)
                            {
                                auto const& Bn   = toPerIndexVector(state.B, {indices...});
                                auto const& rhot = ct.template getRhot<direction>()(indices...);

                                return spatial_hyperresistive_<direction>(Btidx, Bn, vecLaplJ, rhot,
                                                                          F_B, F_Etot);
                            }
                            else
                                throw std::runtime_error("Error - Ohm - unknown hyper_mode");
                        }
                    });
            }
        });
    }


    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(state); }
    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::forward_as_tuple(state); }

private:
    template<auto direction>
    auto save_tranverse_magnetic_field_(auto const& uL, auto const& uR, MeshIndex<dimension> idx)
    {
        auto Bidx = riemann_.vector_riemann_averaging(uL.B, uR.B);

        auto& Bt = getBt_<direction>();

        Bt(Component::X)(idx) = Bidx.x;
        Bt(Component::Y)(idx) = Bidx.y;
        Bt(Component::Z)(idx) = Bidx.z;
    }

    template<auto direction>
    auto& getBt_() const
    {
        if constexpr (direction == Direction::X)
            return state.bt_x;
        else if constexpr (direction == Direction::Y)
            return state.bt_y;
        else if constexpr (direction == Direction::Z)
            return state.bt_z;
    }

    template<auto direction>
    void constant_hyperresistive_(auto const& Bt, auto const& vecLaplJ, auto& F_B,
                                  auto& F_Etot) const
    {
        equations_.template resistive_contributions<direction>(-nu, Bt, vecLaplJ, F_B, F_Etot);
    }

    template<auto direction>
    void spatial_hyperresistive_(auto const& Bt, auto const& B, auto const& vecLaplJ,
                                 auto const& rhot, auto& F_B, auto& F_Etot) const
    {
        auto minMeshSize = [&]() {
            auto const meshSize = layout_.meshSize();
            if constexpr (dimension == 1)
                return meshSize[0];
            else if constexpr (dimension == 2)
                return std::min({meshSize[0], meshSize[1]});
            else
                return std::min({meshSize[0], meshSize[1], meshSize[2]});
        }();


        auto computeHR = [&](auto Bx, auto By, auto Bz) {
            auto b          = std::sqrt(Bx * Bx + By * By + Bz * Bz);
            auto const coef = -nu * minMeshSize * minMeshSize * (b / rhot + 1);
            equations_.template resistive_contributions<direction>(coef, Bt, vecLaplJ, F_B, F_Etot);
        };

        if constexpr (direction == Direction::X)
        {
            auto const Bx = B.x; // normal component
            auto const By = Bt.y;
            auto const Bz = Bt.z;

            computeHR(Bx, By, Bz);
        }
        else if constexpr (direction == Direction::Y)
        {
            auto const Bx = Bt.x;
            auto const By = B.y; // normal component
            auto const Bz = Bt.z;

            computeHR(Bx, By, Bz);
        }
        else if constexpr (direction == Direction::Z)
        {
            auto const Bx = Bt.x;
            auto const By = Bt.y;
            auto const Bz = B.z; // normal component

            computeHR(Bx, By, Bz);
        }
    }

    template<auto direction>
    auto transverse_laplacian_(auto const& Jt, MeshIndex<dimension> index) const
    {
        if constexpr (direction == Direction::X)
        {
            auto const JyLapl = layout_.laplacian(Jt(Component::Y), index);
            auto const JzLapl = layout_.laplacian(Jt(Component::Z), index);
            return PerIndexVector<double>{std::nan(""), JyLapl, JzLapl};
        }
        else if constexpr (direction == Direction::Y)
        {
            auto const JxLapl = layout_.laplacian(Jt(Component::X), index);
            auto const JzLapl = layout_.laplacian(Jt(Component::Z), index);
            return PerIndexVector<double>{JxLapl, std::nan(""), JzLapl};
        }
        else if constexpr (direction == Direction::Z)
        {
            auto const JxLapl = layout_.laplacian(Jt(Component::X), index);
            auto const JyLapl = layout_.laplacian(Jt(Component::Y), index);
            return PerIndexVector<double>{JxLapl, JyLapl, std::nan("")};
        }
    }


    GridLayout layout_;
    Equations equations_;
    RiemannSolver_t riemann_;
};

} // namespace PHARE::core

#endif
