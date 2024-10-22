#ifndef PHARE_MHD_STATE_HPP
#define PHARE_MHD_STATE_HPP

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
        using field_type = typename VecFieldT::field_type;

    public:
        static constexpr auto dimension = VecFieldT::dimension;

        //-------------------------------------------------------------------------
        //                  start the ResourcesUser interface
        //-------------------------------------------------------------------------

        NO_DISCARD bool isUsable() const
        {
            return rho.isUsable() and V.isUsable() and B_FV.isUsable() and P.isUsable()
                   and rhoV.isUsable() and Etot.isUsable() and B_CT.isUsable() and J.isUsable()
                   and E.isUsable() and B_RSx.isUsable() and B_RSy.isUsable() and B_RSz.isUsable();
        }

        NO_DISCARD bool isSettable() const
        {
            return rho.isSettable() and V.isSettable() and B_FV.isSettable() and P.isSettable()
                   and rhoV.isSettable() and Etot.isSettable() and B_CT.isSettable()
                   and J.isSettable() and E.isSettable() and B_RSx.isSettable()
                   and B_RSy.isSettable() and B_RSz.isSettable();
        }

        NO_DISCARD auto getCompileTimeResourcesViewList() const
        {
            return std::forward_as_tuple(rho, V, B_FV, P, rhoV, Etot, B_CT, J, E, B_RSx, B_RSy,
                                         B_RSz);
        }

        NO_DISCARD auto getCompileTimeResourcesViewList()
        {
            return std::forward_as_tuple(rho, V, B_FV, P, rhoV, Etot, B_CT, J, E, B_RSx, B_RSy,
                                         B_RSz);
        }

        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------

        MHDState(PHARE::initializer::PHAREDict const& dict)
            : rho{"rho", MHDQuantity::Scalar::rho}
            , V{"V", MHDQuantity::Vector::V}
            , B_FV{"B_FV", MHDQuantity::Vector::B_FV}
            , P{"P", MHDQuantity::Scalar::P}
            ,

            rhoV{"rhoV", MHDQuantity::Vector::rhoV}
            , Etot{"Etot", MHDQuantity::Scalar::Etot}
            ,

            B_CT{"B_CT", MHDQuantity::Vector::B_CT}
            , E{"E", MHDQuantity::Vector::E}
            , J{"J", MHDQuantity::Vector::J}
            ,

            B_RSx{"B_RS", MHDQuantity::Vector::B_RSx}
            , B_RSy{"B_RS", MHDQuantity::Vector::B_RSy}
            , B_RSz{"B_RS", MHDQuantity::Vector::B_RSz}
            ,



            rhoinit_{
                dict["density"]["initializer"].template to<initializer::InitFunction<dimension>>()}
            , Vinit_{dict["velocity"]["initializer"]}
            , Binit_{dict["magnetic"]["initializer"]}
            , Pinit_{dict["pressure"]["initializer"]
                         .template to<initializer::InitFunction<dimension>>()}
        {
        }

        template<typename GridLayout>
        void initialize(GridLayout const& layout)
        {
            FieldUserFunctionInitializer::initialize(rho, layout, rhoinit_);
            Vinit_.initialize(V, layout);
            Binit_.initialize(B_CT, layout);
            FieldUserFunctionInitializer::initialize(P, layout, Pinit_);
        }

        field_type rho;
        VecFieldT V;
        VecFieldT B_FV;
        field_type P;

        VecFieldT rhoV;
        field_type Etot;

        VecFieldT B_CT;
        VecFieldT E;
        VecFieldT J;

        VecFieldT B_RSx;
        VecFieldT B_RSy;
        VecFieldT B_RSz;


    private:
        initializer::InitFunction<dimension> rhoinit_;
        VecFieldInitializer<dimension> Vinit_;
        VecFieldInitializer<dimension> Binit_;
        initializer::InitFunction<dimension> Pinit_;
    };
} // namespace core
} // namespace PHARE

#endif // PHARE_MHD_STATE_HPP
