#ifndef PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_NON_IDEAL_EMF_HPP
#define PHARE_CORE_NUMERICS_CONSTRAINED_TRANSPORT_NON_IDEAL_EMF_HPP

#include "core/def.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/meta/meta_utilities.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

namespace PHARE::core
{

template<typename VecField>
class NonIdealEMFState
{
public:
    NonIdealEMFState() = default;
    explicit NonIdealEMFState(bool const isNonIdeal)
    {
        if (isNonIdeal)
            Ediss_.emplace_back("E_diss", MHDQuantity::Vector::E);
    }

    NO_DISCARD std::vector<VecField>& getRunTimeResourcesViewList() { return Ediss_; }
    NO_DISCARD std::vector<VecField> const& getRunTimeResourcesViewList() const { return Ediss_; }

    NO_DISCARD auto& Ediss() { return Ediss_[0]; }
    NO_DISCARD auto const& Ediss() const { return Ediss_[0]; }

private:
    std::vector<VecField> Ediss_;
};


template<typename GridLayout>
class NonIdealEMF : public OhmInfo
{
    using Super                     = OhmInfo;
    constexpr static auto dimension = GridLayout::dimension;

public:
    using Info_t = Super;

    NonIdealEMF(OhmInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
    {
    }

    void operator()(auto& emf_state, auto const& mhd_state) const
    {
        auto& Ediss     = emf_state.Ediss();
        auto const& J   = mhd_state.J;
        auto const& B   = mhd_state.B;
        auto const& rho = mhd_state.rho;

        Constexprifier{isResistive(), isHyperResistive(),
                       hyper_mode}([&]<bool isResistive, bool isHyperResistive, HyperMode hyper>() {
            for_N<3>([&](auto i) {
                constexpr auto component = static_cast<Component>(i());
                auto& E                  = Ediss(component);
                auto const& Jc           = J(component);
                layout_.evalOnBox(E, [&](auto&... args) {
                    MeshIndex<dimension> idx{args...};
                    double e = 0.;
                    if constexpr (isResistive)
                        e += eta * Jc(idx);
                    if constexpr (isHyperResistive)
                        e -= hyper_coef_<component, hyper>(B, rho, idx)
                             * layout_.laplacian(Jc, idx);
                    E(idx) = e;
                });
            });
        });
    }

private:
    template<auto component, HyperMode hyper>
    double hyper_coef_(auto const& B, auto const& rho, MeshIndex<dimension> idx) const
    {
        if constexpr (hyper == HyperMode::constant)
            return nu;
        else
        {
            auto const& meshSize = layout_.meshSize();
            auto const dx        = *std::min_element(meshSize.begin(), meshSize.end());

            auto coef = [&]<auto BxProj, auto ByProj, auto BzProj, auto rhoProj>() {
                auto const bx = GridLayout::template project<BxProj>(B(Component::X), idx);
                auto const by = GridLayout::template project<ByProj>(B(Component::Y), idx);
                auto const bz = GridLayout::template project<BzProj>(B(Component::Z), idx);
                auto const n  = GridLayout::template project<rhoProj>(rho, idx);
                auto const b  = std::sqrt(bx * bx + by * by + bz * bz);
                return nu * dx * dx * (b / n + 1);
            };

            if constexpr (component == Component::X)
                return coef
                    .template operator()<GridLayout::BxToEx, GridLayout::ByToEx, GridLayout::BzToEx,
                                         GridLayout::implT::cellCenterToEdgeX>();
            else if constexpr (component == Component::Y)
                return coef
                    .template operator()<GridLayout::BxToEy, GridLayout::ByToEy, GridLayout::BzToEy,
                                         GridLayout::implT::cellCenterToEdgeY>();
            else
                return coef
                    .template operator()<GridLayout::BxToEz, GridLayout::ByToEz, GridLayout::BzToEz,
                                         GridLayout::implT::cellCenterToEdgeZ>();
        }
    }

    GridLayout layout_;
};

} // namespace PHARE::core

#endif
