#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/point/point.hpp"
#include "initializer/data_provider.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/types.hpp"

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

template<auto direction, size_t dim, bool HyperResistivity>
auto getGrow(int const nghosts)
{
    Point<std::uint32_t, dim> p{};

    auto dir = static_cast<size_t>(direction);

    for (size_t i = 0; i < dim; ++i)
    {
        if (i != dir)
            p[i] = nghosts;
    }

    // add one extra layer in the direction of the flux laplacian computation. Maybe some later
    // optimisation would let us just compute for uct and have the extra layer only reconstructed
    // for j
    if constexpr (HyperResistivity)
    {
        p[dir] += 1;
    }

    return p;
}

template<typename GridLayout, typename MHDModel, template<typename> typename Reconstruction,
         typename RiemannSolver, typename Equations>
class Godunov : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

    using Reconstruction_t = Reconstruction<GridLayout>;
    using Reconstructor_t  = Reconstructor<Reconstruction_t>;
    using RiemannSolver_t  = RiemannSolver;

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
        , hyper_mode_{cppdict::get_value(dict, "hyper_mode", std::string{"constant"}) == "constant"
                          ? HyperMode::constant
                          : HyperMode::spatial}
        , equations_{gamma_, eta_, nu_}
        , riemann_{gamma_}
    {
    }

    template<typename State, typename Fluxes>
    void operator()(auto& ct, State& state, Fluxes& fluxes)
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - GodunovFluxes - GridLayout not set");

        constexpr auto directions = getDirections<dimension>();

        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_->evalOnBiggerBox(
                fluxes.template expose_centering<direction>(),
                getGrow<direction, dimension, HyperResistivity>(Reconstruction_t::nghosts),
                [&](auto&... indices) {
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

                        // if constexpr (HyperResistivity)
                        // {
                        //     auto const& [laplJL, laplJR]
                        //         = Reconstructor_t::template reconstructed_laplacian<direction>(
                        //             layout_->inverseMeshSize(), state.J, {indices...});
                        //
                        //     auto const& LaplJ = std::forward_as_tuple(laplJL, laplJR);
                        //
                        //     auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i)
                        //     {
                        //         return equations_.template compute<direction>(
                        //             std::get<i>(u), std::get<i>(j), std::get<i>(LaplJ));
                        //     });
                        //
                        //     fluxes.template get_dir<direction>({indices...})
                        //         = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);
                        //
                        //     ct.template save<direction>(riemann_.vt, riemann_.jt,
                        //                                 riemann_.rhot, riemann_.uct_coefs,
                        //                                 {indices...});
                        // }
                        // else
                        // {
                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u),
                                                                          std::get<i>(j));
                        });

                        // if constexpr (Hall)
                        // {
                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR, jL, jR);

                        ct.template save<direction>(riemann_.vt, riemann_.jt, riemann_.rhot,
                                                    riemann_.uct_coefs, {indices...});

                        // for energy ExB term
                        if constexpr (Resistivity || HyperResistivity)
                        {
                            save_tranverse_magnetic_field_<direction>(uL, uR, {indices...});
                        }
                        // }
                        // else // Resistivity only
                        // {
                        //     fluxes.template get_dir<direction>({indices...})
                        //         = riemann_.template solve<direction>(uL, uR, fL, fR);
                        //
                        //     ct.template save<direction>(riemann_.vt,
                        //                                 riemann_.uct_coefs, {indices...});
                        // }
                        // }
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
                        {
                            save_tranverse_magnetic_field_<direction>(uL, uR, {indices...});
                        }
                    }
                });

            // adding resistive contributions to energy taking advantage of the already computed jt
            // fluxes for the laplacian computation. This probably doesn't need the grow as the
            // required quantities for ct are already saved.
            if constexpr (Resistivity || HyperResistivity)
            {
                layout_->evalOnBox(
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
                                eta_, Btidx, Jtidx, F_B, F_Etot);
                        }
                        if constexpr (HyperResistivity)
                        {
                            auto const vecLaplJ
                                = transverse_laplacian_<direction>(Jt, {indices...});

                            if (hyper_mode_ == HyperMode::constant)
                                return constant_hyperresistive_<direction>(Btidx, vecLaplJ, F_B,
                                                                           F_Etot);
                            else if (hyper_mode_ == HyperMode::spatial)
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

    void registerResources(MHDModel& model)
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            model.resourcesManager->registerResources(bt_x);
            if constexpr (dimension >= 2)
            {
                model.resourcesManager->registerResources(bt_y);
                
                // Register X-flux edge-centered B fields (2D+)
                model.resourcesManager->registerResources(bt_y_at_z_edges_x);
                model.resourcesManager->registerResources(bt_z_at_y_edges_x);
                
                // Register Y-flux edge-centered B fields (2D+)
                model.resourcesManager->registerResources(bt_x_at_z_edges_y);
                model.resourcesManager->registerResources(bt_z_at_x_edges_y);
                
                if constexpr (dimension == 3)
                {
                    model.resourcesManager->registerResources(bt_z);
                    // Register Z-flux edge-centered B fields (3D only)
                    model.resourcesManager->registerResources(bt_x_at_y_edges_z);
                    model.resourcesManager->registerResources(bt_y_at_x_edges_z);
                }
            }
        }
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            model.resourcesManager->allocate(bt_x, patch, allocateTime);
            if constexpr (dimension >= 2)
            {
                model.resourcesManager->allocate(bt_y, patch, allocateTime);
                
                // Allocate X-flux edge-centered B fields (2D+)
                model.resourcesManager->allocate(bt_y_at_z_edges_x, patch, allocateTime);
                model.resourcesManager->allocate(bt_z_at_y_edges_x, patch, allocateTime);
                
                // Allocate Y-flux edge-centered B fields (2D+)
                model.resourcesManager->allocate(bt_x_at_z_edges_y, patch, allocateTime);
                model.resourcesManager->allocate(bt_z_at_x_edges_y, patch, allocateTime);
                
                if constexpr (dimension == 3)
                {
                    model.resourcesManager->allocate(bt_z, patch, allocateTime);
                    // Allocate Z-flux edge-centered B fields (3D only)
                    model.resourcesManager->allocate(bt_x_at_y_edges_z, patch, allocateTime);
                    model.resourcesManager->allocate(bt_y_at_x_edges_z, patch, allocateTime);
                }
            }
        }
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            if constexpr (dimension == 1)
            {
                return std::forward_as_tuple(bt_x);
            }
            else if constexpr (dimension == 2)
            {
                // 2D: X and Y fluxes exist, Z doesn't
                return std::forward_as_tuple(bt_x, bt_y, 
                                           bt_y_at_z_edges_x, bt_z_at_y_edges_x,  // X-flux
                                           bt_x_at_z_edges_y, bt_z_at_x_edges_y); // Y-flux
            }
            else if constexpr (dimension == 3)
            {
                // 3D: All fluxes exist
                return std::forward_as_tuple(bt_x, bt_y, bt_z, 
                                           bt_y_at_z_edges_x, bt_z_at_y_edges_x,  // X-flux
                                           bt_x_at_z_edges_y, bt_z_at_x_edges_y,  // Y-flux
                                           bt_x_at_y_edges_z, bt_y_at_x_edges_z); // Z-flux
            }
        }
        else
            return std::forward_as_tuple();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        if constexpr (Resistivity || HyperResistivity)
        {
            if constexpr (dimension == 1)
            {
                return std::forward_as_tuple(bt_x);
            }
            else if constexpr (dimension == 2)
            {
                // 2D: X and Y fluxes exist, Z doesn't
                return std::forward_as_tuple(bt_x, bt_y,
                                           bt_y_at_z_edges_x, bt_z_at_y_edges_x,  // X-flux
                                           bt_x_at_z_edges_y, bt_z_at_x_edges_y); // Y-flux
            }
            else if constexpr (dimension == 3)
            {
                // 3D: All fluxes exist
                return std::forward_as_tuple(bt_x, bt_y, bt_z,
                                           bt_y_at_z_edges_x, bt_z_at_y_edges_x,  // X-flux
                                           bt_x_at_z_edges_y, bt_z_at_x_edges_y,  // Y-flux
                                           bt_x_at_y_edges_z, bt_y_at_x_edges_z); // Z-flux
            }
        }
        else
            return std::forward_as_tuple();
    }

    template<typename CT, typename State, typename Fluxes>
    void apply_poynting_correction(CT const& ct, State const& state, Fluxes& fluxes)
    {
        // Apply Poynting flux correction to energy: ∂E/∂t -= ∇·(E×B)
        // This must be called AFTER CT has computed both E and edge-B fields
        // Uses edge-centered B from CT (guaranteed temporally consistent with E)
        if (!this->hasLayout())
            throw std::runtime_error("Error - GodunovFluxes::apply_poynting_correction - GridLayout not set");

        constexpr auto directions = getDirections<dimension>();
        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_->evalOnBox(
                fluxes.template expose_centering<direction>(), [&](auto&... indices) {
                    auto& F_Etot = fluxes.template get_dir<direction>({indices...}).Etot();
                    poynting_energy_flux_<direction>(ct, state.E, MeshIndex<dimension>{indices...}, F_Etot);
                });
        });
    }

private:
    template<auto direction>
    auto save_tranverse_magnetic_field_(auto const& uL, auto const& uR, MeshIndex<dimension> idx)
    {
        // Store edge-centered B in CT (unconditional, part of upwind CT scheme)
        ct.template save_edge_B<direction>(uL.B, uR.B, idx);
    }

    template<auto direction>
    auto& getBt_() const
    {
        if constexpr (direction == Direction::X)
            return bt_x;
        else if constexpr (direction == Direction::Y)
            return bt_y;
        else if constexpr (direction == Direction::Z)
            return bt_z;
    }

    template<auto direction>
    void constant_hyperresistive_(auto const& Bt, auto const& vecLaplJ, auto& F_B,
                                  auto& F_Etot) const
    {
        equations_.template resistive_contributions<direction>(-nu_, Bt, vecLaplJ, F_B, F_Etot);
    }

    template<auto direction>
    void spatial_hyperresistive_(auto const& Bt, auto const& B, auto const& vecLaplJ,
                                 auto const& rhot, auto& F_B, auto& F_Etot) const
    {
        auto minMeshSize = [&]() {
            auto const meshSize = layout_->meshSize();
            if constexpr (dimension == 1)
                return meshSize[0];
            else if constexpr (dimension == 2)
                return std::min({meshSize[0], meshSize[1]});
            else
                return std::min({meshSize[0], meshSize[1], meshSize[2]});
        }();


        auto computeHR = [&](auto Bx, auto By, auto Bz) {
            auto b          = std::sqrt(Bx * Bx + By * By + Bz * Bz);
            auto const coef = -nu_ * minMeshSize * minMeshSize * (b / rhot + 1);
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
            auto const JyLapl = layout_->laplacian(Jt(Component::Y), index);
            auto const JzLapl = layout_->laplacian(Jt(Component::Z), index);
            return PerIndexVector<double>{std::nan(""), JyLapl, JzLapl};
        }
        else if constexpr (direction == Direction::Y)
        {
            auto const JxLapl = layout_->laplacian(Jt(Component::X), index);
            auto const JzLapl = layout_->laplacian(Jt(Component::Z), index);
            return PerIndexVector<double>{JxLapl, std::nan(""), JzLapl};
        }
        else if constexpr (direction == Direction::Z)
        {
            auto const JxLapl = layout_->laplacian(Jt(Component::X), index);
            auto const JyLapl = layout_->laplacian(Jt(Component::Y), index);
            return PerIndexVector<double>{JxLapl, JyLapl, std::nan("")};
        }
    }

    template<auto direction, typename CT>
    void poynting_energy_flux_(CT const& ct, auto const& E, MeshIndex<dimension> const& index, auto& F_Etot) const
    {
        // Compute magnetic energy flux via Poynting vector: S·n̂ = (E × B)·n̂
        // E components live on edges (from CT)
        // B components are edge-centered (from CT, temporally consistent with E)
        
        auto const& Ex = E(Component::X);
        auto const& Ey = E(Component::Y);
        auto const& Ez = E(Component::Z);

        if constexpr (direction == Direction::X)
        {
            // X-flux face: Sx = Ey*Bz - Ez*By
            // Ey at y-edges × Bz at y-edges, Ez at z-edges × By at z-edges
            double EyBz = 0.0;
            double EzBy = 0.0;

            if constexpr (dimension >= 2)
            {
                auto const& By_z = ct.template getBt_edge_y_at_z<Direction::X>();
                auto const& Bz_y = ct.template getBt_edge_z_at_y<Direction::X>();
                
                if constexpr (dimension == 2)
                {
                    // 2D: direct multiplication (no z-spatial offset)
                    EzBy = Ez(index) * By_z(index);
                    EyBz = 0.0; // No 3D term in 2D
                }
                else if constexpr (dimension == 3)
                {
                    // 3D: Average Ez (z-edge) with By (z-edge): 0.5*(Ez_k*By_k + Ez_k+1*By_k+1)
                    EzBy = 0.5 * (Ez(index) * By_z(index) 
                                + Ez(index + MeshIndex<dimension>::iz()) * By_z(index + MeshIndex<dimension>::iz()));
                    
                    // Average Ey (y-edge) with Bz (y-edge): 0.5*(Ey_j*Bz_j + Ey_j+1*Bz_j+1)
                    EyBz = 0.5 * (Ey(index) * Bz_y(index) 
                                + Ey(index + MeshIndex<dimension>::iy()) * Bz_y(index + MeshIndex<dimension>::iy()));
                }
            }

            F_Etot += EyBz - EzBy;
        }
        else if constexpr (direction == Direction::Y && dimension >= 2)
        {
            // Y-flux face: Sy = Ez*Bx - Ex*Bz
            // Ez at z-edges × Bx at z-edges, Ex at x-edges × Bz at x-edges
            auto const& Bx_z = ct.template getBt_edge_x_at_z<Direction::Y>();
            auto const& Bz_x = ct.template getBt_edge_z_at_x<Direction::Y>();

            double EzBx = 0.0;
            double ExBz = 0.0;

            if constexpr (dimension == 2)
            {
                // 2D: No z-offset
                EzBx = Ez(index) * Bx_z(index);
                ExBz = Ex(index) * Bz_x(index);
            }
            else if constexpr (dimension == 3)
            {
                // 3D: Average with z-offset 
                // Average Ez (z-edge) with Bx (z-edge): 0.5*(Ez_k*Bx_k + Ez_k+1*Bx_k+1)
                EzBx = 0.5 * (Ez(index) * Bx_z(index) 
                            + Ez(index + MeshIndex<dimension>::iz()) * Bx_z(index + MeshIndex<dimension>::iz()));
                
                // Average Ex (x-edge) with Bz (x-edge): 0.5*(Ex_i*Bz_i + Ex_i+1*Bz_i+1)
                ExBz = 0.5 * (Ex(index) * Bz_x(index) 
                            + Ex(index + MeshIndex<dimension>::ix()) * Bz_x(index + MeshIndex<dimension>::ix()));
            }

            F_Etot += EzBx - ExBz;
        }
        else if constexpr (direction == Direction::Z && dimension == 3)
        {
            // Z-flux only exists in 3D
            // Z-flux face: Sz = Ex*By - Ey*Bx
            // Ex at y-edges × By at y-edges, Ey at x-edges × Bx at x-edges
            auto const& Bx_y = ct.template getBt_edge_x_at_y<Direction::Z>();
            auto const& By_x = ct.template getBt_edge_y_at_x<Direction::Z>();
            
            // Average Ex (x-edge) with By (x-edge): 0.5*(Ex_i*By_i + Ex_i+1*By_i+1)
            double ExBy = 0.5 * (Ex(index) * By_x(index) 
                               + Ex(index + MeshIndex<dimension>::ix()) * By_x(index + MeshIndex<dimension>::ix()));
            
            // Average Ey (y-edge) with Bx (y-edge): 0.5*(Ey_j*Bx_j + Ey_j+1*Bx_j+1)
            double EyBx = 0.5 * (Ey(index) * Bx_y(index) 
                               + Ey(index + MeshIndex<dimension>::iy()) * Bx_y(index + MeshIndex<dimension>::iy()));

            F_Etot += ExBy - EyBx;
        }
    }


    double const gamma_;
    double const eta_;
    double const nu_;
    HyperMode const hyper_mode_;

    Equations equations_;
    RiemannSolver_t riemann_;

    MHDModel::vecfield_type bt_x{"b_t_x", MHDQuantity::Vector::VecFlux_x};
    MHDModel::vecfield_type bt_y{"b_t_y", MHDQuantity::Vector::VecFlux_y};
    MHDModel::vecfield_type bt_z{"b_t_z", MHDQuantity::Vector::VecFlux_z};
};

} // namespace PHARE::core

#endif
