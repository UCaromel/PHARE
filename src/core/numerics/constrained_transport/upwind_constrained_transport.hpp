#ifndef PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP
#define PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP

#include <cmath>

#include "amr/physical_models/mhd_model.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/def.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/utilities/index/index.hpp"
#include "initializer/data_provider.hpp"

namespace PHARE::core
{
template<typename GridLayout, typename MHDModel, bool Hall, bool Resistivity, bool HyperResistivity>
class UpwindConstrainedTransport : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    UpwindConstrainedTransport(PHARE::initializer::PHAREDict const& dict)
        : eta_{dict["resistivity"].template to<double>()}
        , nu_{dict["hyper_resistivity"].template to<double>()}
    {
    }

    template<auto direction>
    void save(auto const& uL, auto const& uR, auto const vt, auto const& coefs,
              MeshIndex<dimension> const& idx)
    {
        auto assign_fields
            = [&]<typename Field>(auto& vT, Field& aL, Field& aR, Field& dL, Field& dR) {
                  vT(Component::X)(idx) = vt.x;
                  vT(Component::Y)(idx) = vt.y;
                  vT(Component::Z)(idx) = vt.z;
                  if constexpr (Hall || Resistivity || HyperResistivity) {}
                  aL(idx) = coefs[0];
                  aR(idx) = coefs[1];
                  dL(idx) = coefs[2];
                  dR(idx) = coefs[3];
              };

        if constexpr (direction == Direction::X)
            assign_fields(vt_x, aL_x, aR_x, dL_x, dR_x);
        else if constexpr (direction == Direction::Y)
            assign_fields(vt_y, aL_y, aR_y, dL_y, dR_y);
        else if constexpr (direction == Direction::Z)
            assign_fields(vt_z, aL_z, aR_z, dL_z, dR_z);
    }

    template<typename VecField>
    void operator()(VecField& E, VecField const& B)
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - UpwindConstrainedTransport - GridLayout not set, "
                                     "cannot proceed to calculate E");

        if constexpr (dimension == 2)
        {
            auto& Ex = E(Component::X);
            auto& Ey = E(Component::Y);
            auto& Ez = E(Component::Z);

            // for now test in ideal MHD only
            Ex.zero();
            Ey.zero();

            layout_->evalOnBox(Ez, [&](auto&... args) mutable { EzEq_(Ez, B, {args...}); });
        }
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(vt_x);
        model.resourcesManager->registerResources(aL_x);
        model.resourcesManager->registerResources(aR_x);
        model.resourcesManager->registerResources(dL_x);
        model.resourcesManager->registerResources(dR_x);
        // if constexpr (Hall || Resistivity || HyperResistivity)
        // {
        //     model.registerResources(jL_x);
        //     model.registerResources(jR_x);
        // }
        if constexpr (dimension >= 2)
        {
            model.resourcesManager->registerResources(vt_y);
            model.resourcesManager->registerResources(aL_y);
            model.resourcesManager->registerResources(aR_y);
            model.resourcesManager->registerResources(dL_y);
            model.resourcesManager->registerResources(dR_y);
            // if constexpr (Hall || Resistivity || HyperResistivity)
            // {
            //     model.resourcesManager->registerResources(jL_y);
            //     model.resourcesManager->registerResources(jR_y);
            // }
            if constexpr (dimension == 3)
            {
                model.resourcesManager->registerResources(vt_z);
                model.resourcesManager->registerResources(aL_z);
                model.resourcesManager->registerResources(aR_z);
                model.resourcesManager->registerResources(dL_z);
                model.resourcesManager->registerResources(dR_z);
                // if constexpr (Hall || Resistivity || HyperResistivity)
                // {
                //     model.registerResources(jL_z);
                //     model.registerResources(jR_z);
                // }
            }
        }
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(vt_x, patch, allocateTime);
        model.resourcesManager->allocate(aL_x, patch, allocateTime);
        model.resourcesManager->allocate(aR_x, patch, allocateTime);
        model.resourcesManager->allocate(dL_x, patch, allocateTime);
        model.resourcesManager->allocate(dR_x, patch, allocateTime);
        // if constexpr (Hall || Resistivity || HyperResistivity)
        // {
        //     model.resourcesManager->allocate(jL_x, patch, allocateTime);
        //     model.resourcesManager->allocate(jR_x, patch, allocateTime);
        // }
        if constexpr (dimension >= 2)
        {
            model.resourcesManager->allocate(vt_y, patch, allocateTime);
            model.resourcesManager->allocate(aL_y, patch, allocateTime);
            model.resourcesManager->allocate(aR_y, patch, allocateTime);
            model.resourcesManager->allocate(dL_y, patch, allocateTime);
            model.resourcesManager->allocate(dR_y, patch, allocateTime);
            // if constexpr (Hall || Resistivity || HyperResistivity)
            // {
            //     model.resourcesManager->allocate(jL_y, patch, allocateTime);
            //     model.resourcesManager->allocate(jR_y, patch, allocateTime);
            // }
            if constexpr (dimension == 3)
            {
                model.resourcesManager->allocate(vt_z, patch, allocateTime);
                model.resourcesManager->allocate(aL_z, patch, allocateTime);
                model.resourcesManager->allocate(aR_z, patch, allocateTime);
                model.resourcesManager->allocate(dL_z, patch, allocateTime);
                model.resourcesManager->allocate(dR_z, patch, allocateTime);
                // if constexpr (Hall || Resistivity || HyperResistivity)
                // {
                //     model.resourcesManager->allocate(jL_z, patch, allocateTime);
                //     model.resourcesManager->allocate(jR_z, patch, allocateTime);
                // }
            }
        }
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        if constexpr (dimension == 1)
            return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x,
                                         dR_x /*, jL_x,
jR_x*/);
        else if constexpr (dimension == 2)
            return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x /*, jL_x,
                        jR_x*/, vt_y, aL_y, aR_y, dL_y,
                                         dR_y                         /*,
                        jL_y, jR_y*/);
        else if constexpr (dimension == 3)
            return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x /*, jL_x,
                    jR_x*/, vt_y, aL_y, aR_y, dL_y, dR_y              /*,
                        jL_y, jR_y*/, vt_z, aL_z, aR_z, dL_z, dR_z /*, jL_z, jR_z*/);
        else
            throw std::runtime_error(
                "Error - UpwindConstrainedTransport - dimension not supported");
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        if constexpr (dimension == 1)
            return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x,
                                         dR_x /*, jL_x,
jR_x*/);
        else if constexpr (dimension == 2)
            return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x /*, jL_x,
                        jR_x*/, vt_y, aL_y, aR_y, dL_y,
                                         dR_y                         /*,
                        jL_y, jR_y*/);
        else if constexpr (dimension == 3)
            return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x /*, jL_x,
                    jR_x*/, vt_y, aL_y, aR_y, dL_y, dR_y              /*,
                        jL_y, jR_y*/, vt_z, aL_z, aR_z, dL_z, dR_z /*, jL_z, jR_z*/);
        else
            throw std::runtime_error(
                "Error - UpwindConstrainedTransport - dimension not supported");
    }

private:
    // ??
    void ExEq_() {}

    void EyEq_() {}

    template<typename VecField>
    void EzEq_(auto& Ez, VecField const& B, MeshIndex<dimension> idx)
    {
        auto aW = 0.5 * (aL_x(idx) + aL_x(layout_->template previous<Direction::Y>(idx)));
        auto aE = 0.5 * (aR_x(idx) + aR_x(layout_->template previous<Direction::Y>(idx)));
        auto aS = 0.5 * (aL_y(idx) + aL_y(layout_->template previous<Direction::X>(idx)));
        auto aN = 0.5 * (aR_y(idx) + aR_y(layout_->template previous<Direction::X>(idx)));
        auto dW = 0.5 * (dL_x(idx) + dL_x(layout_->template previous<Direction::Y>(idx)));
        auto dE = 0.5 * (dR_x(idx) + dR_x(layout_->template previous<Direction::Y>(idx)));
        auto dS = 0.5 * (dL_y(idx) + dL_y(layout_->template previous<Direction::X>(idx)));
        auto dN = 0.5 * (dR_y(idx) + dR_y(layout_->template previous<Direction::X>(idx)));

        auto vyS = vt_x(Component::Y)(layout_->template previous<Direction::Y>(idx));
        auto vxW = vt_y(Component::X)(layout_->template previous<Direction::X>(idx));
        auto vyN = vt_x(Component::Y)(idx);
        auto vxE = vt_y(Component::X)(idx);

        auto ByW = B(Component::Y)(layout_->template previous<Direction::X>(idx));
        auto ByE = B(Component::Y)(idx);
        auto BxS = B(Component::X)(layout_->template previous<Direction::Y>(idx));
        auto BxN = B(Component::X)(idx);

        Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                  + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);

        // if constexpr (Hall)
        // {
        //     auto jyS = aW * jL_x(Component::Y)(layout_->template previous<Direction::Y>(idx))
        //                + aE * jR_x(Component::Y)(layout_->template previous<Direction::Y>(idx));
        //     auto jxW = aS * jL_y(Component::X)(layout_->template previous<Direction::X>(idx))
        //                + aN * jR_y(Component::X)(layout_->template previous<Direction::X>(idx));
        //
        //     auto jyN = aW * jL_x(Component::Y)(idx) + aE * jR_x(Component::Y)(idx);
        //     auto jxE = aS * jL_y(Component::X)(idx) + aN * jR_y(Component::X)(idx);
        //
        //     auto rhoS = aW * rhoL_x(layout_->template previous<Direction::Y>(idx))
        //                 + aE * rhoR_x(layout_->template previous<Direction::Y>(idx));
        //     auto rhoW = aS * rhoL_y(layout_->template previous<Direction::X>(idx))
        //                 + aN * rhoR_y(layout_->template previous<Direction::X>(idx));
        //     auto rhoN = aW * rhoL_x(idx) + aE * rhoR_x(idx);
        //     auto rhoE = aS * rhoL_y(idx) + aN * rhoR_y(idx);
        //
        //     Ez(idx) += (aW * jxW * ByW / rhoW + aE * jxE * ByE / rhoE)
        //                - (aS * jyS * BxS / rhoS + aN * jyN * BxN / rhoN);
        // }

        // ??
        // if constexpr (Resistivity)
        // {
        //     Ez(idx) += eta_ * Jz(idx);
        // }
        //
        // if constexpr (HyperResistivity)
        // {
        //     Ez(idx) += nu_ * layout_->laplacian(Jz, idx);
        // }
    }

    double const eta_;
    double const nu_;

    MHDModel::vecfield_type vt_x{"vL_x", MHDQuantity::Vector::VecFlux_x};
    MHDModel::vecfield_type vt_y{"vL_y", MHDQuantity::Vector::VecFlux_y};
    MHDModel::vecfield_type vt_z{"vL_z", MHDQuantity::Vector::VecFlux_z};

    MHDModel::field_type aL_x{"aL_x", MHDQuantity::Scalar::ScalarFlux_x},
        aR_x{"aR_x", MHDQuantity::Scalar::ScalarFlux_x},
        dL_x{"dL_x", MHDQuantity::Scalar::ScalarFlux_x},
        dR_x{"dR_x", MHDQuantity::Scalar::ScalarFlux_x};

    MHDModel::field_type aL_y{"aL_y", MHDQuantity::Scalar::ScalarFlux_y},
        aR_y{"aR_y", MHDQuantity::Scalar::ScalarFlux_y},
        dL_y{"dL_y", MHDQuantity::Scalar::ScalarFlux_y},
        dR_y{"dR_y", MHDQuantity::Scalar::ScalarFlux_y};

    MHDModel::field_type aL_z{"aL_z", MHDQuantity::Scalar::ScalarFlux_z},
        aR_z{"aR_z", MHDQuantity::Scalar::ScalarFlux_z},
        dL_z{"dL_z", MHDQuantity::Scalar::ScalarFlux_z},
        dR_z{"dR_z", MHDQuantity::Scalar::ScalarFlux_z};
};
} // namespace PHARE::core

#endif
