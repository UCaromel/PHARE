#ifndef PHARE_CORE_MODELS_MHD_CONSERVED_STATE_HPP
#define PHARE_CORE_MODELS_MHD_CONSERVED_STATE_HPP

#include "core/def.hpp"
#include "core/models/physical_state.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include <string>
#include <tuple>

namespace PHARE::core
{
// Resource names of an MHDConservedState, derived from its base name.
struct MHDStateNames
{
    std::string base;

    NO_DISCARD std::string rho() const { return base + "_rho"; }
    NO_DISCARD std::string rhoV() const { return base + "_rhoV"; }
    NO_DISCARD std::string Etot() const { return base + "_Etot"; }
    NO_DISCARD std::string B() const { return base + "_B"; }

    // the names of any state following the same base + "_rho", ... convention (MHDState too)
    static MHDStateNames of(auto const& state)
    {
        auto const& rho = state.rho.name();
        return {rho.substr(0, rho.size() - std::string{"_rho"}.size())};
    }
};

// The conserved MHD variables only (rho, rhoV, Etot, B), without primitive or derived quantities.
template<typename VecFieldT>
class MHDConservedState : public IPhysicalState
{
public:
    using vecfield_type = VecFieldT;
    using field_type    = typename VecFieldT::field_type;

    explicit MHDConservedState(std::string const& base)
        : names{base}
        , rho{names.rho(), MHDQuantity::Scalar::rho}
        , rhoV{names.rhoV(), MHDQuantity::Vector::rhoV}
        , Etot{names.Etot(), MHDQuantity::Scalar::Etot}
        , B{names.B(), MHDQuantity::Vector::B}
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

    MHDStateNames const names;

    field_type rho;
    VecFieldT rhoV;
    field_type Etot;
    VecFieldT B;
};
} // namespace PHARE::core

#endif // PHARE_CORE_MODELS_MHD_CONSERVED_STATE_HPP
