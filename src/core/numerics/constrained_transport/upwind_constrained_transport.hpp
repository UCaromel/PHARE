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
    void save(auto const& uL, auto const& uR, auto const Bx, auto const By, auto const Bz,
              auto const& coefs, MeshIndex<dimension> const& idx)
    {
        auto get_field
            = [this]<typename Field>(Field& field_x, Field& field_y, Field& field_z) -> Field& {
            if constexpr (direction == Direction::X)
                return field_x;
            else if constexpr (direction == Direction::Y)
                return field_y;
            else if constexpr (direction == Direction::Z)
                return field_z;
        };

        get_field(rhoL_x, rhoL_y, rhoL_z)(idx) = uL.rho;
        get_field(rhoR_x, rhoR_y, rhoR_z)(idx) = uR.rho;

        get_field(vL_x(Component::X), vL_y(Component::X), vL_z(Component::X))(idx) = uL.V.x;
        get_field(vR_x(Component::X), vR_y(Component::X), vR_z(Component::X))(idx) = uR.V.x;
        get_field(vL_x(Component::Y), vL_y(Component::Y), vL_z(Component::Y))(idx) = uL.V.y;
        get_field(vR_x(Component::Y), vR_y(Component::Y), vR_z(Component::Y))(idx) = uR.V.y;
        get_field(vL_x(Component::Z), vL_y(Component::Z), vL_z(Component::Z))(idx) = uL.V.z;
        get_field(vR_x(Component::Z), vR_y(Component::Z), vR_z(Component::Z))(idx) = uR.V.z;

        if constexpr (Hall || Resistivity || HyperResistivity)
        {
            get_field(jL_x(Component::X), jL_y(Component::X), jL_z(Component::X))(idx) = uL.J.x;
            get_field(jR_x(Component::X), jR_y(Component::X), jR_z(Component::X))(idx) = uR.J.x;
            get_field(jL_x(Component::Y), jL_y(Component::Y), jL_z(Component::Y))(idx) = uL.J.y;
            get_field(jR_x(Component::Y), jR_y(Component::Y), jR_z(Component::Y))(idx) = uR.J.y;
            get_field(jL_x(Component::Z), jL_y(Component::Z), jL_z(Component::Z))(idx) = uL.J.z;
            get_field(jR_x(Component::Z), jR_y(Component::Z), jR_z(Component::Z))(idx) = uR.J.z;
        }

        get_field(aL_x, aL_y, aL_z)(idx) = coefs[0];
        get_field(aR_x, aR_y, aR_z)(idx) = coefs[1];
        get_field(dL_x, dL_y, dL_z)(idx) = coefs[2];
        get_field(dR_x, dR_y, dR_z)(idx) = coefs[3];
    }

    template<typename VecField>
    void operator()(VecField& E, VecField const& B)
    {
        if constexpr (!this->hasLayout())
            throw std::runtime_error("Error - UpwindConstrainedTransport - GridLayout not set, "
                                     "cannot proceed to calculate E");

        else if constexpr (dimension >= 2)
        {
            auto& Ex = E(Component::X);
            auto& Ey = E(Component::Y);
            auto& Ez = E(Component::Z);

            // for now test in ideal MHD only
            Ex.zero();
            Ey.zero();

            layout_->evalOnBox(Ez, [&](auto&... args) mutable { EzEq_(Ez, B, args...); });
        }
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResource(rhoL_x);
        model.resourcesManager->registerResource(rhoR_x);
        model.resourcesManager->registerResource(vL_x);
        model.resourcesManager->registerResource(vR_x);
        model.resourcesManager->registerResource(aL_x);
        model.resourcesManager->registerResource(aR_x);
        model.resourcesManager->registerResource(dL_x);
        model.resourcesManager->registerResource(dR_x);
        // if constexpr (Hall || Resistivity || HyperResistivity)
        // {
        //     model.registerResource(jL_x);
        //     model.registerResource(jR_x);
        // }
        if constexpr (dimension >= 2)
        {
            model.resourcesManager->registerResource(rhoL_y);
            model.resourcesManager->registerResource(rhoR_y);
            model.resourcesManager->registerResource(vL_y);
            model.resourcesManager->registerResource(vR_y);
            model.resourcesManager->registerResource(aL_y);
            model.resourcesManager->registerResource(aR_y);
            model.resourcesManager->registerResource(dL_y);
            model.resourcesManager->registerResource(dR_y);
            // if constexpr (Hall || Resistivity || HyperResistivity)
            // {
            //     model.resourcesManager->registerResource(jL_y);
            //     model.resourcesManager->registerResource(jR_y);
            // }
            if constexpr (dimension == 3)
            {
                model.resourcesManager->registerResource(rhoL_z);
                model.resourcesManager->registerResource(rhoR_z);
                model.resourcesManager->registerResource(vL_z);
                model.resourcesManager->registerResource(vR_z);
                model.resourcesManager->registerResource(aL_z);
                model.resourcesManager->registerResource(aR_z);
                model.resourcesManager->registerResource(dL_z);
                model.resourcesManager->registerResource(dR_z);
                // if constexpr (Hall || Resistivity || HyperResistivity)
                // {
                //     model.registerResource(jL_z);
                //     model.registerResource(jR_z);
                // }
            }
        }
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime)
    {
        model.resourcesManager->allocate(rhoL_x, patch, allocateTime);
        model.resourcesManager->allocate(rhoR_x, patch, allocateTime);
        model.resourcesManager->allocate(vL_x, patch, allocateTime);
        model.resourcesManager->allocate(vR_x, patch, allocateTime);
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
            model.resourcesManager->allocate(rhoL_y, patch, allocateTime);
            model.resourcesManager->allocate(rhoR_y, patch, allocateTime);
            model.resourcesManager->allocate(vL_y, patch, allocateTime);
            model.resourcesManager->allocate(vR_y, patch, allocateTime);
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
                model.resourcesManager->allocate(rhoL_z, patch, allocateTime);
                model.resourcesManager->allocate(rhoR_z, patch, allocateTime);
                model.resourcesManager->allocate(vL_z, patch, allocateTime);
                model.resourcesManager->allocate(vR_z, patch, allocateTime);
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
            return std::forward_as_tuple(rhoL_x, rhoR_x, vL_x, vR_x, aL_x, aR_x, dL_x,
                                         dR_x /*, jL_x,
jR_x*/);
        else if constexpr (dimension == 2)
            return std::forward_as_tuple(rhoL_x, rhoR_x, vL_x, vR_x, aL_x, aR_x, dL_x,
                                         dR_x /*, jL_x,
jR_x*/, rhoL_y, rhoR_y, vL_y, vR_y, aL_y, aR_y, dL_y,
                                         dR_y /*,
jL_y, jR_y*/);
        else if constexpr (dimension == 3)
            return std::forward_as_tuple(rhoL_x, rhoR_x, vL_x, vR_x, aL_x, aR_x, dL_x,
                                         dR_x              /*, jL_x,
         jR_x*/, rhoL_y, rhoR_y, vL_y, vR_y, aL_y, aR_y, dL_y, dR_y /*,
jL_y, jR_y*/, rhoL_z, rhoR_z, vL_z, vR_z, aL_z, aR_z, dL_z, dR_z /*, jL_z, jR_z*/);
        else
            throw std::runtime_error(
                "Error - UpwindConstrainedTransport - dimension not supported");
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        if constexpr (dimension == 1)
            return std::forward_as_tuple(rhoL_x, rhoR_x, vL_x, vR_x, aL_x, aR_x, dL_x,
                                         dR_x /*, jL_x,
jR_x*/);
        else if constexpr (dimension == 2)
            return std::forward_as_tuple(rhoL_x, rhoR_x, vL_x, vR_x, aL_x, aR_x, dL_x,
                                         dR_x /*, jL_x,
jR_x*/, rhoL_y, rhoR_y, vL_y, vR_y, aL_y, aR_y, dL_y,
                                         dR_y /*,
jL_y, jR_y*/);
        else if constexpr (dimension == 3)
            return std::forward_as_tuple(rhoL_x, rhoR_x, vL_x, vR_x, aL_x, aR_x, dL_x,
                                         dR_x              /*, jL_x,
         jR_x*/, rhoL_y, rhoR_y, vL_y, vR_y, aL_y, aR_y, dL_y, dR_y /*,
jL_y, jR_y*/, rhoL_z, rhoR_z, vL_z, vR_z, aL_z, aR_z, dL_z, dR_z /*, jL_z, jR_z*/);
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

        auto vyS = aW * vL_x(Component::Y)(layout_->template previous<Direction::Y>(idx))
                   + aE * vR_x(Component::Y)(layout_->template previous<Direction::Y>(idx));
        auto vxW = aS * vL_y(Component::X)(layout_->template previous<Direction::X>(idx))
                   + aN * vR_y(Component::X)(layout_->template previous<Direction::X>(idx));

        auto vyN = aW * vL_x(Component::Y)(idx) + aE * vR_x(Component::Y)(idx);
        auto vxE = aS * vL_y(Component::X)(idx) + aN * vR_y(Component::X)(idx);

        auto ByW = B(Component::Y)(layout_->template previous<Direction::X>(idx));
        auto ByE = B(Component::Y)(idx);
        auto BxS = B(Component::X)(layout_->template previous<Direction::Y>(idx));
        auto BxN = B(Component::X)(idx);

        Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                  + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);

        if constexpr (Hall)
        {
            auto jyS = aW * jL_x(Component::Y)(layout_->template previous<Direction::Y>(idx))
                       + aE * jR_x(Component::Y)(layout_->template previous<Direction::Y>(idx));
            auto jxW = aS * jL_y(Component::X)(layout_->template previous<Direction::X>(idx))
                       + aN * jR_y(Component::X)(layout_->template previous<Direction::X>(idx));

            auto jyN = aW * jL_x(Component::Y)(idx) + aE * jR_x(Component::Y)(idx);
            auto jxE = aS * jL_y(Component::X)(idx) + aN * jR_y(Component::X)(idx);

            auto rhoS = aW * rhoL_x(layout_->template previous<Direction::Y>(idx))
                        + aE * rhoR_x(layout_->template previous<Direction::Y>(idx));
            auto rhoW = aS * rhoL_y(layout_->template previous<Direction::X>(idx))
                        + aN * rhoR_y(layout_->template previous<Direction::X>(idx));
            auto rhoN = aW * rhoL_x(idx) + aE * rhoR_x(idx);
            auto rhoE = aS * rhoL_y(idx) + aN * rhoR_y(idx);

            Ez(idx) += (aW * jxW * ByW / rhoW + aE * jxE * ByE / rhoE)
                       - (aS * jyS * BxS / rhoS + aN * jyN * BxN / rhoN);
        }

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

    MHDModel::field_type rhoL_x{"rhoL_x", MHDQuantity::Scalar::ScalarFlux_x},
        rhoR_x{"rhoR_x", MHDQuantity::Scalar::ScalarFlux_x};
    MHDModel::field_type rhoL_y{"rhoL_y", MHDQuantity::Scalar::ScalarFlux_y},
        rhoR_y{"rhoR_y", MHDQuantity::Scalar::ScalarFlux_y};
    MHDModel::field_type rhoL_z{"rhoL_z", MHDQuantity::Scalar::ScalarFlux_z},
        rhoR_z{"rhoR_z", MHDQuantity::Scalar::ScalarFlux_z};


    MHDModel::vecfield_type vL_x{"vL_x", MHDQuantity::Vector::VecFlux_x},
        vR_x{"vR_x", MHDQuantity::Vector::VecFlux_x};
    MHDModel::vecfield_type vL_y{"vL_y", MHDQuantity::Vector::VecFlux_y},
        vR_y{"vR_y", MHDQuantity::Vector::VecFlux_y};
    MHDModel::vecfield_type vL_z{"vL_z", MHDQuantity::Vector::VecFlux_z},
        vR_z{"vR_z", MHDQuantity::Vector::VecFlux_z};

    MHDModel::vecfield_type jL_x{"jL_x", MHDQuantity::Vector::VecFlux_x},
        jR_x{"jR_x", MHDQuantity::Vector::VecFlux_x};
    MHDModel::vecfield_type jL_y{"jL_y", MHDQuantity::Vector::VecFlux_y},
        jR_y{"jR_y", MHDQuantity::Vector::VecFlux_y};
    MHDModel::vecfield_type jL_z{"jL_z", MHDQuantity::Vector::VecFlux_z},
        jR_z{"jR_z", MHDQuantity::Vector::VecFlux_z};

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
