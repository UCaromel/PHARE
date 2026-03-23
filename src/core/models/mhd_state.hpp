#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
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
        void initialize(GridLayout layout)
        {
            FieldUserFunctionInitializer::initialize(rho, layout, rhoinit_);
            Vinit_.initialize(V, layout);
            Binit_.initialize(B, layout);
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);

            using value_type = typename field_type::value_type;
            using array_t    = NdArrayVector<dimension, value_type>;
            using grid_t     = Grid<array_t, MHDQuantity::Scalar>;

            grid_t rho_pv{"init_rho_pv", layout, MHDQuantity::Scalar::rho};
            grid_t vx_pv{"init_vx_pv", layout, MHDQuantity::Scalar::Vx};
            grid_t vy_pv{"init_vy_pv", layout, MHDQuantity::Scalar::Vy};
            grid_t vz_pv{"init_vz_pv", layout, MHDQuantity::Scalar::Vz};
            grid_t p_pv{"init_p_pv", layout, MHDQuantity::Scalar::P};

            grid_t bx_pv{"init_bx_pv", layout, MHDQuantity::Scalar::Bx};
            grid_t by_pv{"init_by_pv", layout, MHDQuantity::Scalar::By};
            grid_t bz_pv{"init_bz_pv", layout, MHDQuantity::Scalar::Bz};

            grid_t rhoVx_pv{"init_rhoVx_pv", layout, MHDQuantity::Scalar::rhoVx};
            grid_t rhoVy_pv{"init_rhoVy_pv", layout, MHDQuantity::Scalar::rhoVy};
            grid_t rhoVz_pv{"init_rhoVz_pv", layout, MHDQuantity::Scalar::rhoVz};
            grid_t Etot_pv{"init_Etot_pv", layout, MHDQuantity::Scalar::Etot};

            auto const& rho_pv_f   = *(&rho_pv);
            auto const& vx_pv_f    = *(&vx_pv);
            auto const& vy_pv_f    = *(&vy_pv);
            auto const& vz_pv_f    = *(&vz_pv);
            auto const& p_pv_f     = *(&p_pv);
            auto const& bx_pv_f    = *(&bx_pv);
            auto const& by_pv_f    = *(&by_pv);
            auto const& bz_pv_f    = *(&bz_pv);
            auto const& rhoVx_pv_f = *(&rhoVx_pv);
            auto const& rhoVy_pv_f = *(&rhoVy_pv);
            auto const& rhoVz_pv_f = *(&rhoVz_pv);
            auto const& Etot_pv_f  = *(&Etot_pv);

            constexpr double w_to_point = -1. / 24.;
            constexpr double w_to_avg   = 1. / 24.;

            auto const grow_two = [] {
                Point<std::uint32_t, dimension> grow{};
                for (std::size_t i = 0; i < dimension; ++i)
                    grow[i] = 2;
                return grow;
            }();

            layout.evalOnBiggerBox(rho, grow_two, [&](auto&... args) mutable {
                auto const index = MeshIndex<dimension>{args...};

                rho_pv(index) = rho(index) + w_to_point * layout.lapl(rho, index);
                vx_pv(index)  = V(Component::X)(index) + w_to_point * layout.lapl(V(Component::X), index);
                vy_pv(index)  = V(Component::Y)(index) + w_to_point * layout.lapl(V(Component::Y), index);
                vz_pv(index)  = V(Component::Z)(index) + w_to_point * layout.lapl(V(Component::Z), index);
                p_pv(index)   = P(index) + w_to_point * layout.lapl(P, index);

                bx_pv(index) = B(Component::X)(index)
                               + w_to_point
                                     * layout.template tranverseLapl<Direction::X>(B(Component::X),
                                                                                    index);
                by_pv(index) = B(Component::Y)(index)
                               + w_to_point
                                     * layout.template tranverseLapl<Direction::Y>(B(Component::Y),
                                                                                    index);
                bz_pv(index) = B(Component::Z)(index)
                               + w_to_point
                                     * layout.template tranverseLapl<Direction::Z>(B(Component::Z),
                                                                                    index);
            });

            layout.evalOnBiggerBox(rho, grow_two, [&](auto&... args) mutable {
                auto const index = MeshIndex<dimension>{args...};

                auto const bx_cc = GridLayout::project(bx_pv_f, index, GridLayout::faceXToCellCenter());
                auto const by_cc = GridLayout::project(by_pv_f, index, GridLayout::faceYToCellCenter());
                auto const bz_cc = GridLayout::project(bz_pv_f, index, GridLayout::faceZToCellCenter());

                auto&& [rho_vx, rho_vy, rho_vz]
                    = vToRhoV(rho_pv(index), vx_pv(index), vy_pv(index), vz_pv(index));

                rhoVx_pv(index) = rho_vx;
                rhoVy_pv(index) = rho_vy;
                rhoVz_pv(index) = rho_vz;
                Etot_pv(index)  = eosPToEtot(gamma_, rho_pv(index), vx_pv(index), vy_pv(index),
                                            vz_pv(index), bx_cc, by_cc, bz_cc, p_pv(index));
            });

            layout.evalOnBiggerBox(rho, grow_for_init_, [&](auto&... args) mutable {
                auto const index         = MeshIndex<dimension>{args...};
                rhoV(Component::X)(index) = rhoVx_pv(index) + w_to_avg * layout.lapl(rhoVx_pv_f, index);
                rhoV(Component::Y)(index) = rhoVy_pv(index) + w_to_avg * layout.lapl(rhoVy_pv_f, index);
                rhoV(Component::Z)(index) = rhoVz_pv(index) + w_to_avg * layout.lapl(rhoVz_pv_f, index);
                Etot(index)               = Etot_pv(index) + w_to_avg * layout.lapl(Etot_pv_f, index);
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
