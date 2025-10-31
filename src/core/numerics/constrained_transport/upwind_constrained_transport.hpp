#ifndef PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP
#define PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP

#include <cmath>
#include <filesystem>

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
template<typename GridLayout, typename MHDModel, template<typename> typename Reconstruction,
         bool Hall, bool Resistivity, bool HyperResistivity>
class UpwindConstrainedTransport : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

    using Reconstruction_t = Reconstruction<GridLayout>;

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
        auto assign_fields = [&](auto& vT, auto& aL, auto& aR, auto& dL, auto& dR) {
            vT(Component::X)(idx) = vt.x;
            vT(Component::Y)(idx) = vt.y;
            vT(Component::Z)(idx) = vt.z;

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

    template<auto direction>
    void save(auto const& uL, auto const& uR, auto const& vt, auto const& jt, auto const rhot,
              auto const& coefs, MeshIndex<dimension> const& idx)
    {
        auto assign_fields
            = [&](auto& vT, auto& jT, auto& rhoT, auto& aL, auto& aR, auto& dL, auto& dR) {
                  vT(Component::X)(idx) = vt.x;
                  vT(Component::Y)(idx) = vt.y;
                  vT(Component::Z)(idx) = vt.z;

                  jT(Component::X)(idx) = jt.x;
                  jT(Component::Y)(idx) = jt.y;
                  jT(Component::Z)(idx) = jt.z;

                  rhoT(idx) = rhot;

                  aL(idx) = coefs[0];
                  aR(idx) = coefs[1];
                  dL(idx) = coefs[2];
                  dR(idx) = coefs[3];
              };

        if constexpr (direction == Direction::X)
            assign_fields(vt_x, jt_x, rhot_x, aL_x, aR_x, dL_x, dR_x);
        else if constexpr (direction == Direction::Y)
            assign_fields(vt_y, jt_y, rhot_y, aL_y, aR_y, dL_y, dR_y);
        else if constexpr (direction == Direction::Z)
            assign_fields(vt_z, jt_z, rhot_z, aL_z, aR_z, dL_z, dR_z);
    }

    template<typename VecField>
    void operator()(VecField& E, VecField const& B) const
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - UpwindConstrainedTransport - GridLayout not set, "
                                     "cannot proceed to calculate E");

        if constexpr (dimension == 2)
        {
            auto& Ex = E(Component::X);
            auto& Ey = E(Component::Y);
            auto& Ez = E(Component::Z);

            layout_->evalOnBox(Ex, [&](auto&... args) mutable { ExEq_(Ex, B, {args...}); });
            layout_->evalOnBox(Ey, [&](auto&... args) mutable { EyEq_(Ey, B, {args...}); });
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
        if constexpr (Hall)
        {
            model.resourcesManager->registerResources(jt_x);
            model.resourcesManager->registerResources(rhot_x);
        }
        if constexpr (dimension >= 2)
        {
            model.resourcesManager->registerResources(vt_y);
            model.resourcesManager->registerResources(aL_y);
            model.resourcesManager->registerResources(aR_y);
            model.resourcesManager->registerResources(dL_y);
            model.resourcesManager->registerResources(dR_y);
            if constexpr (Hall)
            {
                model.resourcesManager->registerResources(jt_y);
                model.resourcesManager->registerResources(rhot_y);
            }
            if constexpr (dimension == 3)
            {
                model.resourcesManager->registerResources(vt_z);
                model.resourcesManager->registerResources(aL_z);
                model.resourcesManager->registerResources(aR_z);
                model.resourcesManager->registerResources(dL_z);
                model.resourcesManager->registerResources(dR_z);
                if constexpr (Hall)
                {
                    model.resourcesManager->registerResources(jt_z);
                    model.resourcesManager->registerResources(rhot_z);
                }
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
        if constexpr (Hall)
        {
            model.resourcesManager->allocate(jt_x, patch, allocateTime);
            model.resourcesManager->allocate(rhot_x, patch, allocateTime);
        }
        if constexpr (dimension >= 2)
        {
            model.resourcesManager->allocate(vt_y, patch, allocateTime);
            model.resourcesManager->allocate(aL_y, patch, allocateTime);
            model.resourcesManager->allocate(aR_y, patch, allocateTime);
            model.resourcesManager->allocate(dL_y, patch, allocateTime);
            model.resourcesManager->allocate(dR_y, patch, allocateTime);
            if constexpr (Hall)
            {
                model.resourcesManager->allocate(jt_y, patch, allocateTime);
                model.resourcesManager->allocate(rhot_y, patch, allocateTime);
            }
            if constexpr (dimension == 3)
            {
                model.resourcesManager->allocate(vt_z, patch, allocateTime);
                model.resourcesManager->allocate(aL_z, patch, allocateTime);
                model.resourcesManager->allocate(aR_z, patch, allocateTime);
                model.resourcesManager->allocate(dL_z, patch, allocateTime);
                model.resourcesManager->allocate(dR_z, patch, allocateTime);
                if constexpr (Hall)
                {
                    model.resourcesManager->allocate(jt_z, patch, allocateTime);
                    model.resourcesManager->allocate(rhot_z, patch, allocateTime);
                }
            }
        }
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        if constexpr (dimension == 1)
        {
            if constexpr (Hall)
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, jt_x, rhot_x);
            else
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x);
        }
        else if constexpr (dimension == 2)
        {
            if constexpr (Hall)
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, jt_x, rhot_x, vt_y, aL_y,
                                             aR_y, dL_y, dR_y, jt_y, rhot_y);
            else
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, vt_y, aL_y, aR_y, dL_y,
                                             dR_y);
        }
        else if constexpr (dimension == 3)
        {
            if constexpr (Hall)
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, jt_x, rhot_x, vt_y, aL_y,
                                             aR_y, dL_y, dR_y, jt_y, rhot_y, vt_z, aL_z, aR_z, dL_z,
                                             dR_z, jt_z, rhot_z);
            else
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, vt_y, aL_y, aR_y, dL_y,
                                             dR_y, vt_z, aL_z, aR_z, dL_z, dR_z);
        }
        else
            throw std::runtime_error(
                "Error - UpwindConstrainedTransport - dimension not supported");
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        if constexpr (dimension == 1)
        {
            if constexpr (Hall)
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, jt_x, rhot_x);
            else
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x);
        }
        else if constexpr (dimension == 2)
        {
            if constexpr (Hall)
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, jt_x, rhot_x, vt_y, aL_y,
                                             aR_y, dL_y, dR_y, jt_y, rhot_y);
            else
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, vt_y, aL_y, aR_y, dL_y,
                                             dR_y);
        }
        else if constexpr (dimension == 3)
        {
            if constexpr (Hall)
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, jt_x, rhot_x, vt_y, aL_y,
                                             aR_y, dL_y, dR_y, jt_y, rhot_y, vt_z, aL_z, aR_z, dL_z,
                                             dR_z, jt_z, rhot_z);
            else
                return std::forward_as_tuple(vt_x, aL_x, aR_x, dL_x, dR_x, vt_y, aL_y, aR_y, dL_y,
                                             dR_y, vt_z, aL_z, aR_z, dL_z, dR_z);
        }
        else
            throw std::runtime_error(
                "Error - UpwindConstrainedTransport - dimension not supported");
    }

private:
    void ExEq_(auto& Ex, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto [BzL, BzR]
                = Reconstruction_t::template reconstruct<Direction::Y>(B(Component::Z), idx);

            auto FL
                = BzL * vt_y(Component::Y)(idx) - B(Component::Y)(idx) * vt_y(Component::Z)(idx);
            auto FR
                = BzR * vt_y(Component::Y)(idx) - B(Component::Y)(idx) * vt_y(Component::Z)(idx);

            Ex(idx) = -(aL_y(idx) * FL + aR_y(idx) * FR) + (dR_y(idx) * BzR - dL_y(idx) * BzL);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / rhot_y(idx);
                auto JxB_x_L = jt_y(Component::Y)(idx) * BzL
                               - jt_y(Component::Z)(idx) * B(Component::Y)(idx);
                auto JxB_x_R = jt_y(Component::Y)(idx) * BzR
                               - jt_y(Component::Z)(idx) * B(Component::Y)(idx);

                auto HallL = -JxB_x_L * invRho;
                auto HallR = -JxB_x_R * invRho;

                auto F_Bz_y = aL_y(idx) * HallL + aR_y(idx) * HallR;

                Ex(idx) += -F_Bz_y;
            }
        }
    }

    void EyEq_(auto& Ey, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto [BzL, BzR]
                = Reconstruction_t::template reconstruct<Direction::X>(B(Component::Z), idx);

            auto FL
                = BzL * vt_x(Component::X)(idx) - B(Component::X)(idx) * vt_x(Component::Z)(idx);
            auto FR
                = BzR * vt_x(Component::X)(idx) - B(Component::X)(idx) * vt_x(Component::Z)(idx);

            Ey(idx) = (aL_x(idx) * FL + aR_x(idx) * FR) - (dR_x(idx) * BzR - dL_x(idx) * BzL);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / rhot_x(idx);
                auto JxB_y_L = jt_x(Component::Z)(idx) * B(Component::X)(idx)
                               - jt_x(Component::X)(idx) * BzL;
                auto JxB_y_R = jt_x(Component::Z)(idx) * B(Component::X)(idx)
                               - jt_x(Component::X)(idx) * BzR;

                auto HallL = JxB_y_L * invRho;
                auto HallR = JxB_y_R * invRho;

                auto F_Bz_x = aL_x(idx) * HallL + aR_x(idx) * HallR;

                Ey(idx) += F_Bz_x;
            }
        }
    }

    void EzEq_(auto& Ez, auto const& B, MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto aW = 0.5 * (aL_x(idx) + aL_x(layout_->template previous<Direction::Y>(idx)));
            auto aE = 0.5 * (aR_x(idx) + aR_x(layout_->template previous<Direction::Y>(idx)));
            auto aS = 0.5 * (aL_y(idx) + aL_y(layout_->template previous<Direction::X>(idx)));
            auto aN = 0.5 * (aR_y(idx) + aR_y(layout_->template previous<Direction::X>(idx)));
            auto dW = 0.5 * (dL_x(idx) + dL_x(layout_->template previous<Direction::Y>(idx)));
            auto dE = 0.5 * (dR_x(idx) + dR_x(layout_->template previous<Direction::Y>(idx)));
            auto dS = 0.5 * (dL_y(idx) + dL_y(layout_->template previous<Direction::X>(idx)));
            auto dN = 0.5 * (dR_y(idx) + dR_y(layout_->template previous<Direction::X>(idx)));

            auto [vyS, vyN]
                = Reconstruction_t::template reconstruct<Direction::Y>(vt_x(Component::Y), idx);
            auto [vxW, vxE]
                = Reconstruction_t::template reconstruct<Direction::X>(vt_y(Component::X), idx);

            auto [BxS, BxN]
                = Reconstruction_t::template reconstruct<Direction::Y>(B(Component::X), idx);
            auto [ByW, ByE]
                = Reconstruction_t::template reconstruct<Direction::X>(B(Component::Y), idx);

            Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                      + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);

            if constexpr (Hall)
            {
                auto [jyS, jyN]
                    = Reconstruction_t::template reconstruct<Direction::Y>(jt_x(Component::Y), idx);
                auto [jxW, jxE]
                    = Reconstruction_t::template reconstruct<Direction::X>(jt_y(Component::X), idx);

                auto [rhoS, rhoN]
                    = Reconstruction_t::template reconstruct<Direction::Y>(rhot_x, idx);
                auto [rhoW, rhoE]
                    = Reconstruction_t::template reconstruct<Direction::X>(rhot_y, idx);

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
    }

    double const eta_;
    double const nu_;

    MHDModel::vecfield_type vt_x{"vL_x", MHDQuantity::Vector::VecFlux_x};
    MHDModel::vecfield_type vt_y{"vL_y", MHDQuantity::Vector::VecFlux_y};
    MHDModel::vecfield_type vt_z{"vL_z", MHDQuantity::Vector::VecFlux_z};

    MHDModel::vecfield_type jt_x{"jL_x", MHDQuantity::Vector::VecFlux_x};
    MHDModel::vecfield_type jt_y{"jL_y", MHDQuantity::Vector::VecFlux_y};
    MHDModel::vecfield_type jt_z{"jL_z", MHDQuantity::Vector::VecFlux_z};

    MHDModel::field_type rhot_x{"rho_t_x", MHDQuantity::Scalar::ScalarFlux_x};
    MHDModel::field_type rhot_y{"rho_t_y", MHDQuantity::Scalar::ScalarFlux_y};
    MHDModel::field_type rhot_z{"rho_t_z", MHDQuantity::Scalar::ScalarFlux_z};

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
