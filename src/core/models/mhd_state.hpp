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
            FieldUserFunctionInitializer::initialize(rho, layout, rhoinit_);
            Vinit_.initialize(V, layout);
            Binit_.initialize(B, layout);
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);

            using value_type = typename field_type::value_type;
            using array_t    = NdArrayVector<dimension, value_type>;
            using grid_t     = Grid<array_t, MHDQuantity::Scalar>;

            grid_t bx_pv{"init_bx_pv", layout, MHDQuantity::Scalar::Bx};
            grid_t by_pv{"init_by_pv", layout, MHDQuantity::Scalar::By};
            grid_t bz_pv{"init_bz_pv", layout, MHDQuantity::Scalar::Bz};

            auto const& bx_pv_f = bx_pv;
            auto const& by_pv_f = by_pv;
            auto const& bz_pv_f = bz_pv;

            auto const grow_two = [] {
                Point<std::uint32_t, dimension> grow{};
                for (std::size_t i = 0; i < dimension; ++i)
                    grow[i] = 2;
                return grow;
            }();

            auto to_point = PointValueHandler_ref<GridLayout>{layout};

            layout.evalOnBiggerBox(rho, grow_two, [&](auto&... args) mutable {
                auto const index = MeshIndex<dimension>{args...};

                bx_pv(index)
                    = to_point
                          .template getFaceCentered<Direction::X,
                                                    PointValueConversionMode::ToPointValue>(
                              B(Component::X), index);
                by_pv(index)
                    = to_point
                          .template getFaceCentered<Direction::Y,
                                                    PointValueConversionMode::ToPointValue>(
                              B(Component::Y), index);
                bz_pv(index)
                    = to_point
                          .template getFaceCentered<Direction::Z,
                                                    PointValueConversionMode::ToPointValue>(
                              B(Component::Z), index);
            });

            auto point_cons = [&](MeshIndex<dimension> const& index) {
                auto const rho_point
                    = to_point.template getCellCentered<PointValueConversionMode::ToPointValue>(
                        rho, index);
                auto const vx_point
                    = to_point.template getCellCentered<PointValueConversionMode::ToPointValue>(
                        V(Component::X), index);
                auto const vy_point
                    = to_point.template getCellCentered<PointValueConversionMode::ToPointValue>(
                        V(Component::Y), index);
                auto const vz_point
                    = to_point.template getCellCentered<PointValueConversionMode::ToPointValue>(
                        V(Component::Z), index);
                auto const p_point
                    = to_point.template getCellCentered<PointValueConversionMode::ToPointValue>(
                        P, index);

                auto const bx_cc = GridLayout::project(bx_pv_f, index, GridLayout::faceXToCellCenter());
                auto const by_cc = GridLayout::project(by_pv_f, index, GridLayout::faceYToCellCenter());
                auto const bz_cc = GridLayout::project(bz_pv_f, index, GridLayout::faceZToCellCenter());

                auto&& [rho_vx, rho_vy, rho_vz]
                    = vToRhoV(rho_point, vx_point, vy_point, vz_point);
                auto const etot_point
                    = eosPToEtot(gamma_, rho_point, vx_point, vy_point, vz_point, bx_cc, by_cc,
                                 bz_cc, p_point);

                return std::array<value_type, 4>{rho_vx, rho_vy, rho_vz, etot_point};
            };

            auto add_directional_lapl = [&]<auto direction>(MeshIndex<dimension> const& index,
                                                            auto const& center, auto& lapl) {
                auto const next_cons = point_cons(GridLayout::template next<direction>(index));
                auto const prev_cons = point_cons(GridLayout::template previous<direction>(index));
                for (std::size_t i = 0; i < lapl.size(); ++i)
                    lapl[i] += next_cons[i] - 2.0 * center[i] + prev_cons[i];
            };

            layout.evalOnBiggerBox(rho, grow_for_init_, [&](auto&... args) mutable {
                auto const index  = MeshIndex<dimension>{args...};
                auto const center = point_cons(index);

                std::array<value_type, 4> lapl{};
                add_directional_lapl.template operator()<Direction::X>(index, center, lapl);
                if constexpr (dimension >= 2)
                    add_directional_lapl.template operator()<Direction::Y>(index, center, lapl);
                if constexpr (dimension == 3)
                    add_directional_lapl.template operator()<Direction::Z>(index, center, lapl);

                rhoV(Component::X)(index) = center[0] + lapl[0] / 24.;
                rhoV(Component::Y)(index) = center[1] + lapl[1] / 24.;
                rhoV(Component::Z)(index) = center[2] + lapl[2] / 24.;
                Etot(index)               = center[3] + lapl[3] / 24.;
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
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
