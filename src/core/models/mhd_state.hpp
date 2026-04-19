#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/data/vecfield/vecfield_initializer.hpp"
#include "core/def.hpp"
#include "core/physical_quantities.hpp"
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
            : rho{dict["name"].template to<std::string>() + "_" + "rho", PhysicalQuantity::Scalar::MHD_rho}
            , V{dict["name"].template to<std::string>() + "_" + "V", PhysicalQuantity::Vector::MHD_V}
            , B{dict["name"].template to<std::string>() + "_" + "B", PhysicalQuantity::Vector::B}
            , P{dict["name"].template to<std::string>() + "_" + "P", PhysicalQuantity::Scalar::MHD_P}


            , rhoV{dict["name"].template to<std::string>() + "_" + "rhoV",
                   PhysicalQuantity::Vector::MHD_rhoV}
            , Etot{dict["name"].template to<std::string>() + "_" + "Etot",
                   PhysicalQuantity::Scalar::MHD_Etot}


            , E{dict["name"].template to<std::string>() + "_" + "E", PhysicalQuantity::Vector::E}
            , J{dict["name"].template to<std::string>() + "_" + "J", PhysicalQuantity::Vector::J}


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
            : rho{name + "_" + "rho", PhysicalQuantity::Scalar::MHD_rho}
            , V{name + "_" + "V", PhysicalQuantity::Vector::MHD_V}
            , B{name + "_" + "B", PhysicalQuantity::Vector::B}
            , P{name + "_" + "P", PhysicalQuantity::Scalar::MHD_P}


            , rhoV{name + "_" + "rhoV", PhysicalQuantity::Vector::MHD_rhoV}
            , Etot{name + "_" + "Etot", PhysicalQuantity::Scalar::MHD_Etot}


            , E{name + "_" + "E", PhysicalQuantity::Vector::E}
            , J{name + "_" + "J", PhysicalQuantity::Vector::J}

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

            ToConservativeConverter_ref{layout, gamma_}(
                rho, V, B, P, rhoV, Etot); // initial to conservative conversion because we
                                           // store conservative quantities on the grid
        }

        field_type rho;
        VecFieldT V;
        VecFieldT B;
        field_type P;

        VecFieldT rhoV;
        field_type Etot;

        VecFieldT E;
        VecFieldT J;

    private:
        initializer::InitFunction<dimension> rhoinit_;
        VecFieldInitializer<dimension> Vinit_;
        VecFieldInitializer<dimension> Binit_;
        initializer::InitFunction<dimension> Pinit_;

        double const gamma_;
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
