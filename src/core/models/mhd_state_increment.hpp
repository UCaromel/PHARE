#ifndef PHARE_MHD_STATE_INCREMENT_HPP
#define PHARE_MHD_STATE_INCREMENT_HPP

#include "core/def.hpp"
#include "core/models/physical_state.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <tuple>

namespace PHARE::core
{
struct MHDStateIncrementNames
{
    std::string rho;
    std::string rhoV_x, rhoV_y, rhoV_z;
    std::string Etot;
    std::string B_x, B_y, B_z;

    MHDStateIncrementNames() = default;

    explicit MHDStateIncrementNames(std::string const& base)
        : rho{base + "_rho"}
        , rhoV_x{base + "_rhoV_x"}
        , rhoV_y{base + "_rhoV_y"}
        , rhoV_z{base + "_rhoV_z"}
        , Etot{base + "_Etot"}
        , B_x{base + "_B_x"}
        , B_y{base + "_B_y"}
        , B_z{base + "_B_z"}
    {
        static_cast<void>(validatedBaseName());
    }

    template<typename MHDStateIncrementT>
        requires requires(MHDStateIncrementT const& state) {
            state.rho.name();
            state.rhoV.getComponentName(Component::X);
            state.Etot.name();
            state.B.getComponentName(Component::X);
        }
    explicit MHDStateIncrementNames(MHDStateIncrementT const& state)
        : rho{state.rho.name()}
        , rhoV_x{state.rhoV.getComponentName(Component::X)}
        , rhoV_y{state.rhoV.getComponentName(Component::Y)}
        , rhoV_z{state.rhoV.getComponentName(Component::Z)}
        , Etot{state.Etot.name()}
        , B_x{state.B.getComponentName(Component::X)}
        , B_y{state.B.getComponentName(Component::Y)}
        , B_z{state.B.getComponentName(Component::Z)}
    {
        static_cast<void>(validatedBaseName());
    }

    [[nodiscard]] std::string validatedBaseName() const
    {
        std::array<std::pair<std::string const*, char const*>, 8> const suffixes{{
            {&rho, "_rho"},       {&rhoV_x, "_rhoV_x"}, {&rhoV_y, "_rhoV_y"},
            {&rhoV_z, "_rhoV_z"}, {&Etot, "_Etot"},     {&B_x, "_B_x"},
            {&B_y, "_B_y"},       {&B_z, "_B_z"}}};

        auto const& firstName = *suffixes.front().first;
        auto const firstSuffix = std::string{suffixes.front().second};
        if (firstName.size() <= firstSuffix.size()
            || !firstName.ends_with(firstSuffix))
            throw std::invalid_argument{"invalid MHD state-increment name bundle"};

        auto const base = firstName.substr(0, firstName.size() - firstSuffix.size());
        for (auto const& [name, suffix] : suffixes)
            if (*name != base + suffix)
                throw std::invalid_argument{"MHD state-increment names must form one base family"};
        return base;
    }
};

// Conserved MHD state bundle used by solver-owned temporal history. It deliberately excludes
// primitive and derived quantities: temporal coarse-fine transfer and reflux consume only U.
template<typename VecFieldT>
class MHDStateIncrement : public IPhysicalState
{
public:
    using vecfield_type = VecFieldT;
    using field_type    = typename VecFieldT::field_type;

    explicit MHDStateIncrement(std::string name)
        : MHDStateIncrement{MHDStateIncrementNames{name}.validatedBaseName(), PrivateTag{}}
    {
    }

    explicit MHDStateIncrement(MHDStateIncrementNames const& names)
        : MHDStateIncrement{names.validatedBaseName(), PrivateTag{}}
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

private:
    struct PrivateTag
    {
    };

    MHDStateIncrement(std::string const& base, PrivateTag)
        : rho{base + "_rho", MHDQuantity::Scalar::rho}
        , rhoV{base + "_rhoV", MHDQuantity::Vector::rhoV}
        , Etot{base + "_Etot", MHDQuantity::Scalar::Etot}
        , B{base + "_B", MHDQuantity::Vector::B}
    {
    }
};
} // namespace PHARE::core

#endif // PHARE_MHD_STATE_INCREMENT_HPP
