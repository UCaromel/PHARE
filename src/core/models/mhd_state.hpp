#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
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
            return rho.isUsable() and V.isUsable() and B1.isUsable() and B0.isUsable()
                   and P.isUsable() and rhoV.isUsable() and Etot1.isUsable() and J.isUsable()
                   and E.isUsable();
        }

        NO_DISCARD bool isSettable() const
        {
            return rho.isSettable() and V.isSettable() and B1.isSettable() and B0.isSettable()
                    and P.isSettable()
                    and rhoV.isSettable() and Etot1.isSettable() and J.isSettable()
                     and E.isSettable();
        }

        NO_DISCARD auto getCompileTimeResourcesViewList() const
        {
            return std::forward_as_tuple(rho, V, B1, B0, P, rhoV, Etot1, J, E);
        }

        NO_DISCARD auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(rho, V, B1, B0, P, rhoV, Etot1, J, E);
        }

        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------

        MHDState(PHARE::initializer::PHAREDict const& dict)
            : rho{dict["name"].template to<std::string>() + "_" + "rho", MHDQuantity::Scalar::rho}
            , V{dict["name"].template to<std::string>() + "_" + "V", MHDQuantity::Vector::V}
            , B1{dict["name"].template to<std::string>() + "_" + "B1", MHDQuantity::Vector::B1}
            , B0{dict["name"].template to<std::string>() + "_" + "B0", MHDQuantity::Vector::B0}
            , P{dict["name"].template to<std::string>() + "_" + "P", MHDQuantity::Scalar::P}


            , rhoV{dict["name"].template to<std::string>() + "_" + "rhoV",
                   MHDQuantity::Vector::rhoV}
            , Etot1{dict["name"].template to<std::string>() + "_" + "Etot1",
                    MHDQuantity::Scalar::Etot1}


            , E{dict["name"].template to<std::string>() + "_" + "E", MHDQuantity::Vector::E}
            , J{dict["name"].template to<std::string>() + "_" + "J", MHDQuantity::Vector::J}


            , rhoinit_{dict["density"]["initializer"]
                           .template to<initializer::InitFunction<dimension>>()}
            , Vinit_{dict["velocity"]["initializer"]}
            , totalBInit_{dict["magnetic"]["initializer"]}
            , B0init_{dict["external_magnetic"]["initializer"]}
            , Pinit_{dict["pressure"]["initializer"]
                          .template to<initializer::InitFunction<dimension>>()}
            , gamma_{dict["to_conservative_init"]["heat_capacity_ratio"].template to<double>()}
        {
        }

        MHDState(std::string name)
            : rho{name + "_" + "rho", MHDQuantity::Scalar::rho}
            , V{name + "_" + "V", MHDQuantity::Vector::V}
            , B1{name + "_" + "B1", MHDQuantity::Vector::B1}
            , B0{name + "_" + "B0", MHDQuantity::Vector::B0}
            , P{name + "_" + "P", MHDQuantity::Scalar::P}


            , rhoV{name + "_" + "rhoV", MHDQuantity::Vector::rhoV}
            , Etot1{name + "_" + "Etot1", MHDQuantity::Scalar::Etot1}


            , E{name + "_" + "E", MHDQuantity::Vector::E}
            , J{name + "_" + "J", MHDQuantity::Vector::J}

            , gamma_{}
        {
        }

        template<typename GridLayout>
        void updateExternalMagneticField(GridLayout const& layout, double /*time*/ = 0.)
        {
            B0init_.initialize(B0, layout);
        }

        template<typename GridLayout>
        void initialize(GridLayout const& layout)
        {
            FieldUserFunctionInitializer::initialize(rho, layout, rhoinit_);
            Vinit_.initialize(V, layout);
            totalBInit_.initialize(B1, layout);
            B0init_.initialize(B0, layout);
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);

            for (auto const& component : {Component::X, Component::Y, Component::Z})
            {
                auto& B1c       = B1(component);
                auto const& B0c = B0(component);
                layout.evalOnGhostBox(B1c, [&](auto&... args) mutable {
                    B1c(args...) -= B0c(args...);
                });
            }

            ToConservativeConverter_ref{layout, gamma_}(
                rho, V, B1, B0, P, rhoV, Etot1); // initial to conservative conversion because we
                                             // store conservative quantities on the grid
        }

        field_type rho;
        VecFieldT V;
        VecFieldT B1;
        VecFieldT B0;
        field_type P;

        VecFieldT rhoV;
        field_type Etot1;

        VecFieldT E;
        VecFieldT J;

    private:
        initializer::InitFunction<dimension> rhoinit_;
        VecFieldInitializer<dimension> Vinit_;
        VecFieldInitializer<dimension> totalBInit_;
        VecFieldInitializer<dimension> B0init_;
        initializer::InitFunction<dimension> Pinit_;

        double const gamma_;
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
