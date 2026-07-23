#ifndef PHARE_MHD_STATE_INCREMENT_HPP
#define PHARE_MHD_STATE_INCREMENT_HPP

#include "core/def.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/models/physical_state.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <string>
#include <tuple>

namespace PHARE::core
{
// Storage for one per-stage state-space derivative (a "k" in Butcher/Shu-Osher
// convention) or a theta-independent split term of the MC2011 temporal
// reconstruction. Mirrors MHDState's rho/rhoV/Etot/B quads, minus V/P/E/J:
// those are either not filled by a schedule (V/P, local) or not part of the
// conserved state-space RHS this reconstruction acts on (E/J).
template<typename VecFieldT>
class MHDStateIncrement : public IPhysicalState
{
public:
    using vecfield_type = VecFieldT;
    using field_type    = typename VecFieldT::field_type;

    explicit MHDStateIncrement(std::string name)
        : rho{name + "_" + "rho", MHDQuantity::Scalar::rho}
        , rhoV{name + "_" + "rhoV", MHDQuantity::Vector::rhoV}
        , Etot{name + "_" + "Etot", MHDQuantity::Scalar::Etot}
        , B{name + "_" + "B", MHDQuantity::Vector::B}
    {
    }

    NO_DISCARD bool isUsable() const
    {
        return rho.isUsable() and rhoV.isUsable() and Etot.isUsable() and B.isUsable();
    }

    NO_DISCARD bool isSettable() const
    {
        return rho.isSettable() and rhoV.isSettable() and Etot.isSettable() and B.isSettable();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(rho, rhoV, Etot, B);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(rho, rhoV, Etot, B);
    }

    field_type rho;
    VecFieldT rhoV;
    field_type Etot;
    VecFieldT B;
};

// Name-bundle for cross-messenger registration, mirrors core::AllFluxesNames's
// conversion-constructor pattern.
struct MHDStateIncrementNames
{
    std::string rho;
    std::string rhoV_x, rhoV_y, rhoV_z;
    std::string Etot;
    std::string B_x, B_y, B_z;

    MHDStateIncrementNames() = default;

    template<typename MHDStateIncrementT>
    explicit MHDStateIncrementNames(MHDStateIncrementT const& k)
        : rho{k.rho.name()}
        , rhoV_x{k.rhoV.getComponentName(Component::X)}
        , rhoV_y{k.rhoV.getComponentName(Component::Y)}
        , rhoV_z{k.rhoV.getComponentName(Component::Z)}
        , Etot{k.Etot.name()}
        , B_x{k.B.getComponentName(Component::X)}
        , B_y{k.B.getComponentName(Component::Y)}
        , B_z{k.B.getComponentName(Component::Z)}
    {
    }
};

} // namespace PHARE::core

#endif // PHARE_MHD_STATE_INCREMENT_HPP
