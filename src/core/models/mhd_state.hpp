#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
#include "core/numerics/point_values_handler/point_value_handler.hpp"
#include "core/data/grid/grid.hpp"
#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/data/vecfield/vecfield_initializer.hpp"
#include "core/def.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/models/physical_state.hpp"
#include "initializer/data_provider.hpp"

namespace PHARE
{
namespace core
{

    template<typename VecFieldT>
    class MHDState : public IPhysicalState
    {
    public:
        using vecfield_type = VecFieldT;
        using field_type    = typename VecFieldT::field_type;

        static constexpr auto dimension = VecFieldT::dimension;

        //-------------------------------------------------------------------------
        //                  start the ResourcesUser interface
        //-------------------------------------------------------------------------

        NO_DISCARD bool isUsable() const
        {
            return rho.isUsable() and V.isUsable() and B.isUsable() and P.isUsable()
                   and rhoV.isUsable() and Etot.isUsable() and J.isUsable() and E.isUsable();
        }

        NO_DISCARD bool isSettable() const
        {
            return rho.isSettable() and V.isSettable() and B.isSettable() and P.isSettable()
                   and rhoV.isSettable() and Etot.isSettable() and J.isSettable()
                   and E.isSettable();
        }

        NO_DISCARD auto getCompileTimeResourcesViewList() const
        {
            return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, E);
        }

        NO_DISCARD auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, E);
        }

        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------

        MHDState(PHARE::initializer::PHAREDict const& dict)
            : rho{dict["name"].template to<std::string>() + "_" + "rho", MHDQuantity::Scalar::rho}
            , V{dict["name"].template to<std::string>() + "_" + "V", MHDQuantity::Vector::V}
            , B{dict["name"].template to<std::string>() + "_" + "B", MHDQuantity::Vector::B}
            , P{dict["name"].template to<std::string>() + "_" + "P", MHDQuantity::Scalar::P}


            , rhoV{dict["name"].template to<std::string>() + "_" + "rhoV",
                   MHDQuantity::Vector::rhoV}
            , Etot{dict["name"].template to<std::string>() + "_" + "Etot",
                   MHDQuantity::Scalar::Etot}


            , E{dict["name"].template to<std::string>() + "_" + "E", MHDQuantity::Vector::E}
            , J{dict["name"].template to<std::string>() + "_" + "J", MHDQuantity::Vector::J}


            , rhoinit_{dict["density"]["initializer"]
                           .template to<initializer::InitFunction<dimension>>()}
            , Vinit_{dict["velocity"]["initializer"]}
            , Binit_{dict["magnetic"]["initializer"]}
            , Pinit_{dict["pressure"]["initializer"]
                         .template to<initializer::InitFunction<dimension>>()}
            , gamma_{dict["to_conservative_init"]["heat_capacity_ratio"].template to<double>()}
        {
        }

        MHDState(std::string name)
            : rho{name + "_" + "rho", MHDQuantity::Scalar::rho}
            , V{name + "_" + "V", MHDQuantity::Vector::V}
            , B{name + "_" + "B", MHDQuantity::Vector::B}
            , P{name + "_" + "P", MHDQuantity::Scalar::P}


            , rhoV{name + "_" + "rhoV", MHDQuantity::Vector::rhoV}
            , Etot{name + "_" + "Etot", MHDQuantity::Scalar::Etot}


            , E{name + "_" + "E", MHDQuantity::Vector::E}
            , J{name + "_" + "J", MHDQuantity::Vector::J}

            , gamma_{}
        {
        }

        template<typename GridLayout>
        void initialize(GridLayout const& layout)
        {
            // Step 1: Initialize primitive fields using 4th-order GL quadrature
            // These are already 4th-order accurate area/volume averages
            FieldUserFunctionInitializer::initialize(rho, layout, rhoinit_);
            Vinit_.initialize(V, layout);
            Binit_.initialize(B, layout);
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);

            // Step 2: Compute 4th-order area-averaged conservative quantities
            // Strategy: integral-prim → point-prim → point-cons → integral-cons
            //
            // Problem: Product of averages ≠ average of products: <rho*V> - <rho>*<V> = O(dx²)
            // Without correction, computing rhoV = <rho> * <V> gives only 2nd-order accuracy.
            //
            // Solution: Compute point values, multiply, then convert back to averages:
            //   1. Prim_pv  = Prim_avg - lapl(Prim_avg) / 24    [ToPointValue]
            //   2. Cons_pv  = f(Prim_pv)                         [point-value product]
            //   3. Cons_avg = Cons_pv  + lapl(Cons_pv)  / 24    [ToAverage]
            //
            // This gives 4th-order accurate conservative quantities.
            //
            // Note: Computed on grow_for_init_ (one ghost layer) so downstream flux
            // computation has 4th-order values available for point-value conversion.
            
            using value_type = typename field_type::value_type;
            using array_t    = NdArrayVector<dimension, value_type>;
            using grid_t     = Grid<array_t, MHDQuantity::Scalar>;

            // Temporary grids to store point-value primitives and conservatives
            grid_t rho_pv{"init_rho_pv", layout, MHDQuantity::Scalar::rho};
            grid_t vx_pv{"init_vx_pv", layout, MHDQuantity::Scalar::Vx};
            grid_t vy_pv{"init_vy_pv", layout, MHDQuantity::Scalar::Vy};
            grid_t vz_pv{"init_vz_pv", layout, MHDQuantity::Scalar::Vz};
            grid_t p_pv{"init_p_pv", layout, MHDQuantity::Scalar::P};
            
            // B field: convert face-averages to face point-values, then store for projection
            grid_t bx_face_pv{"init_bx_face_pv", layout, MHDQuantity::Scalar::Bx};
            grid_t by_face_pv{"init_by_face_pv", layout, MHDQuantity::Scalar::By};
            grid_t bz_face_pv{"init_bz_face_pv", layout, MHDQuantity::Scalar::Bz};
            
            grid_t rhoVx_pv{"init_rhoVx_pv", layout, MHDQuantity::Scalar::rhoVx};
            grid_t rhoVy_pv{"init_rhoVy_pv", layout, MHDQuantity::Scalar::rhoVy};
            grid_t rhoVz_pv{"init_rhoVz_pv", layout, MHDQuantity::Scalar::rhoVz};
            grid_t Etot_pv{"init_Etot_pv", layout, MHDQuantity::Scalar::Etot};

            auto to_point = PointValueHandler_ref<GridLayout>{layout};

            // First pass: convert B face-averages to face point-values.
            // Uses grow_for_b_face_pv_ (=3) so that Pass 2's 4th-order projection has
            // valid b*_face_pv values at all positions it reads.
            layout.evalOnBiggerBox(rho, grow_for_b_face_pv_, [&](auto&... args) mutable {
                auto const index = MeshIndex<dimension>{args...};
                
                // Convert face-averaged B to face point-values
                // Projection expects point values, not averages
                bx_face_pv(index) = to_point.template getFaceCentered<Direction::X,
                    PointValueConversionMode::ToPointValue>(B(Component::X), index);
                by_face_pv(index) = to_point.template getFaceCentered<Direction::Y,
                    PointValueConversionMode::ToPointValue>(B(Component::Y), index);
                bz_face_pv(index) = to_point.template getFaceCentered<Direction::Z,
                    PointValueConversionMode::ToPointValue>(B(Component::Z), index);
            });
            
            // Second pass: convert cell-centered primitive averages to point values,
            // project B face point-values to cell center, compute conservatives
            layout.evalOnBiggerBox(rho, grow_for_init_, [&](auto&... args) mutable {
                auto const index = MeshIndex<dimension>{args...};

                // Step 2a: Convert primitive cell-center area-averages to point values
                // Formula: Q_pv = Q_avg - lapl(Q_avg) / 24
                rho_pv(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToPointValue>(rho, index);
                vx_pv(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToPointValue>(V(Component::X), index);
                vy_pv(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToPointValue>(V(Component::Y), index);
                vz_pv(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToPointValue>(V(Component::Z), index);
                p_pv(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToPointValue>(P, index);

                // Step 2b: Project B face point-values to cell-center point-values
                // Projection operates on point values to get point value at cell center
                auto const bx_cc_pv = GridLayout::project(bx_face_pv, index,
                                                          GridLayout::faceXToCellCenter());
                auto const by_cc_pv = GridLayout::project(by_face_pv, index,
                                                          GridLayout::faceYToCellCenter());
                auto const bz_cc_pv = GridLayout::project(bz_face_pv, index,
                                                          GridLayout::faceZToCellCenter());

                // Step 2c: Compute conservative point values from primitive point values
                auto&& [rho_vx_pv, rho_vy_pv, rho_vz_pv]
                    = vToRhoV(rho_pv(index), vx_pv(index), vy_pv(index), vz_pv(index));
                    
                rhoVx_pv(index) = rho_vx_pv;
                rhoVy_pv(index) = rho_vy_pv;
                rhoVz_pv(index) = rho_vz_pv;
                
                Etot_pv(index) = eosPToEtot(gamma_, rho_pv(index), vx_pv(index), vy_pv(index),
                                           vz_pv(index), bx_cc_pv, by_cc_pv, bz_cc_pv, p_pv(index));
            });

            // Step 2d: Convert conservative point values to area-averages
            // Formula: Q_avg = Q_pv + lapl(Q_pv) / 24
            layout.evalOnBiggerBox(rho, grow_for_init_, [&](auto&... args) mutable {
                auto const index = MeshIndex<dimension>{args...};

                rhoV(Component::X)(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToAverage>(rhoVx_pv, index);
                rhoV(Component::Y)(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToAverage>(rhoVy_pv, index);
                rhoV(Component::Z)(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToAverage>(rhoVz_pv, index);
                Etot(index) = to_point.template getCellCentered<
                    PointValueConversionMode::ToAverage>(Etot_pv, index);
            });
        }

        field_type rho;
        VecFieldT V;
        VecFieldT B;
        field_type P;

        VecFieldT rhoV;
        field_type Etot;

        VecFieldT E;

        // we might not need J anymore as it could only be the point value version used in the point
        // value handler.
        VecFieldT J;

    private:
        initializer::InitFunction<dimension> rhoinit_;
        VecFieldInitializer<dimension> Vinit_;
        VecFieldInitializer<dimension> Binit_;
        initializer::InitFunction<dimension> Pinit_;

        double const gamma_;

        static constexpr Point<std::uint32_t, dimension> grow_for_init_ = [] {
            Point<std::uint32_t, dimension> grow{};
            for (std::size_t i = 0; i < dimension; ++i)
                grow[i] = 1;
            return grow;
        }();

        // Pass 1 computes b*_face_pv, which Pass 2 projects to cell-center using a
        // 4th-order PrimalToDual stencil with offsets {-1, 0, +1, +2} from the current
        // dual index.  At the outermost ghost cell (physicalEnd+1), the projection reads
        // primal positions up to physicalEnd+3, so b*_face_pv must be computed at least
        // that far.  grow=3 covers both ends (low end needs physicalStart-2, high end
        // needs physicalEnd+3).
        static constexpr Point<std::uint32_t, dimension> grow_for_b_face_pv_ = [] {
            Point<std::uint32_t, dimension> grow{};
            for (std::size_t i = 0; i < dimension; ++i)
                grow[i] = 3;
            return grow;
        }();
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
