#ifndef PHARE_SOLVER_MHD_HPP
#define PHARE_SOLVER_MHD_HPP

#include <array>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "amr/messengers/messenger.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/solvers/solver.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"

namespace PHARE::solver
{
template<typename MHDModel, typename AMR_Types, typename Messenger = amr::MHDMessenger<MHDModel>,
         typename ModelViews_t = MHDModelView<MHDModel>>
class SolverMHD : public ISolver<AMR_Types>
{
private:
    static constexpr auto dimension = MHDModel::dimension;

    using patch_t     = typename AMR_Types::patch_t;
    using level_t     = typename AMR_Types::level_t;
    using hierarchy_t = typename AMR_Types::hierarchy_t;

    using FieldT      = typename MHDModel::field_type;
    using VecFieldT   = typename MHDModel::vecfield_type;
    using GridLayout  = typename MHDModel::gridlayout_type;
    using MHDQuantity = core::MHDQuantity;

    using IPhysicalModel_t = IPhysicalModel<AMR_Types>;
    using IMessenger       = amr::IMessenger<IPhysicalModel_t>;
    using Direction        = core::Direction;

    using GodunovFluxes_t = typename MHDModelView<MHDModel>::GodunovFluxes_t;
    using Ampere_t        = typename MHDModelView<MHDModel>::Ampere_t;


    // Flux calculations
    FieldT rho_x{"rho_x", MHDQuantity::Scalar::ScalarFlux_x};
    VecFieldT rhoV_x{"rhoV_x", MHDQuantity::Vector::VecFlux_x};
    VecFieldT B_x{"B_x", MHDQuantity::Vector::VecFlux_x};
    FieldT Etot_x{"rho_x", MHDQuantity::Scalar::ScalarFlux_x};

    FieldT rho_y{"rho_y", MHDQuantity::Scalar::ScalarFlux_y};
    VecFieldT rhoV_y{"rhoV_y", MHDQuantity::Vector::VecFlux_y};
    VecFieldT B_y{"B_y", MHDQuantity::Vector::VecFlux_y};
    FieldT Etot_y{"rho_y", MHDQuantity::Scalar::ScalarFlux_y};

    FieldT rho_z{"rho_z", MHDQuantity::Scalar::ScalarFlux_z};
    VecFieldT rhoV_z{"rhoV_z", MHDQuantity::Vector::VecFlux_z};
    VecFieldT B_z{"B_z", MHDQuantity::Vector::VecFlux_z};
    FieldT Etot_z{"rho_z", MHDQuantity::Scalar::ScalarFlux_z};

    // Time integration
    FieldT rho1{"rho1", MHDQuantity::Scalar::rho};
    VecFieldT rhoV1{"rhoV1", MHDQuantity::Vector::rhoV};
    VecFieldT B_CT1{"B1", MHDQuantity::Vector::B_CT};
    FieldT Etot1{"rho1", MHDQuantity::Scalar::Etot};

    FieldT rho2{"rho2", MHDQuantity::Scalar::rho};
    VecFieldT rhoV2{"rhoV2", MHDQuantity::Vector::rhoV};
    VecFieldT B_CT2{"B2", MHDQuantity::Vector::B_CT};
    FieldT Etot2{"rho2", MHDQuantity::Scalar::Etot};

    GodunovFluxes_t godunov_;
    Ampere_t ampere_;

public:
    SolverMHD(PHARE::initializer::PHAREDict const& dict)
        : ISolver<AMR_Types>{"MHDSolver"}
        , godunov_{dict["godunov"]}
    {
    }

    virtual ~SolverMHD() = default;

    std::string modelName() const override { return MHDModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& /*info*/) const override {}

    void registerResources(IPhysicalModel<AMR_Types>& /*model*/) override {}

    // TODO make this a resourcesUser
    void allocate(IPhysicalModel<AMR_Types>& /*model*/, patch_t& /*patch*/,
                  double const /*allocateTime*/) const override
    {
    }

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
                      IMessenger& fromCoarserMessenger, const double currentTime,
                      const double newTime) override;


    std::shared_ptr<ISolverModelView> make_view(level_t& level, IPhysicalModel_t& model) override
    {
        /*return std::make_shared<ModelViews_t>(level, dynamic_cast<MHDModel&>(model));*/
        throw std::runtime_error("no MHD model yet");
        return nullptr;
    }

private:
    void godunov_fluxes_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                         double const currentTime, double const newTime);

    void time_integrator_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                          double const currentTime, double const newTime);

    template<typename Layout, typename VecField, typename Q, typename... Fluxes>
    void euler_(Layout& layouts, Q quantities, VecField E, Q quantities_new, double const dt,
                Fluxes... fluxes);

    template<typename Field, typename VecField>
    struct Q
    {
        Q(Field& rho_, VecField& rhoV_, VecField& B_, VecField& Etot_)
            : rho(rho_)
            , rhoVx(rhoV_(core::Component::X))
            , rhoVy(rhoV_(core::Component::Y))
            , rhoVz(rhoV_(core::Component::Z))
            , B(B_)
            , Etot(Etot_)
        {
        }

        Field& rho;
        Field& rhoVx;
        Field& rhoVy;
        Field& rhoVz;
        VecField& B;
        Field& Etot;
    };

    struct TimeSetter
    {
        /*template <typename QuantityAccessor>*/
        /*void operator()(QuantityAccessor accessor) {*/
        /*    for (auto& state : views)*/
        /*        views.model().resourcesManager->setTime(accessor(state), *state.patch, newTime);*/
        /*}*/
        /**/
        /*ModelViews_t& views;*/
        /*double        newTime;*/
    };
};

// -----------------------------------------------------------------------------

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::advanceLevel(
    hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
    IMessenger& fromCoarserMessenger, const double currentTime, const double newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::advanceLevel");

    auto& modelView   = dynamic_cast<ModelViews_t&>(view);
    auto& fromCoarser = dynamic_cast<Messenger&>(fromCoarserMessenger);
    auto level        = hierarchy.getPatchLevel(levelNumber);

    godunov_fluxes_(*level, modelView, fromCoarser, currentTime, newTime);

    time_integrator_(*level, modelView, fromCoarser, currentTime, newTime);
}


template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::godunov_fluxes_(
    level_t& level, ModelViews_t& views, Messenger& fromCoarser, double const currentTime,
    double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::godunov_fluxes_");

    ampere_(views.layouts, views.B_CT, views.J);

    if constexpr (dimension == 1)
    {
        godunov_(views.layouts, views.rho, views.V, views.B_CT, views.P, views.J, views.rho_x,
                 views.rhoV_x, views.B_x, views.Etot_x);
    }
    if constexpr (dimension == 2)
    {
        godunov_(views.layouts, views.rho, views.V, views.B_CT, views.P, views.J, views.rho_x,
                 views.rhoV_x, views.B_x, views.Etot_x, views.rho_y, views.rhoV_y, views.B_y,
                 views.Etot_y);
    }
    if constexpr (dimension == 3)
    {
        godunov_(views.layouts, views.rho, views.V, views.B_CT, views.P, views.J, views.rho_x,
                 views.rhoV_x, views.B_x, views.Etot_x, views.rho_y, views.rhoV_y, views.B_y,
                 views.Etot_y, views.rho_z, views.rhoV_z, views.B_z, views.Etot_z);
    }
}


template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::time_integrator_(
    level_t& level, ModelViews_t& views, Messenger& fromCoarser, double const currentTime,
    double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::time_integrator_");

    auto dt = newTime - currentTime;

    Q Un(views.rho, views.rhoV, views.B_CT, views.Etot);
    Q U1(views.rho1, views.rhoV1, views.B_CT1, views.Etot1);

    Q F_x(views.rho_x, views.rhoV_x, views.B_x, views.Etot_x);

    if constexpr (dimension == 1)
    {
        euler_(views.layouts, Un, views.E, Un, dt, F_x);
    }
    if constexpr (dimension >= 2)
    {
        Q F_y(views.rho_y, views.rhoV_y, views.B_y, views.Etot_y);

        if constexpr (dimension == 2)
        {
            euler_(views.layouts, Un, views.E, Un, dt, F_x, F_y);
        }
        if constexpr (dimension == 3)
        {
            Q F_z(views.rho_z, views.rhoV_z, views.B_z, views.Etot_z);

            euler_(views.layouts, Un, views.E, Un, dt, F_x, F_y, F_z);
        }
    }
}

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
template<typename Layout, typename VecField, typename Q, typename... Fluxes>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::euler_(Layout& layouts, Q quantities,
                                                                     VecField E, Q quantities_new,
                                                                     double const dt,
                                                                     Fluxes... fluxes)
{
    if constexpr (dimension == 1)
    {
        auto&& fluxes_x = std::get<0>(fluxes...);

        finite_volume_euler_(layouts, quantities.rho, quantities_new.rho, dt, fluxes_x.rho);
        finite_volume_euler_(layouts, quantities.rhoVx, quantities_new.rhoVx, dt, fluxes_x.rhoVx);
        finite_volume_euler_(layouts, quantities.rhoVy, quantities_new.rhoVy, dt, fluxes_x.rhoVy);
        finite_volume_euler_(layouts, quantities.rhoVz, quantities_new.rhoVz, dt, fluxes_x.rhoVz);
        finite_volume_euler_(layouts, quantities.Etot, quantities_new.Etot, dt, fluxes_x.Etot);

        constrained_transport_(layouts, E, fluxes_x.B);
        faraday_(layouts, quantities.B, E, quantities_new.B, dt);
    }
    if constexpr (dimension == 2)
    {
        auto&& fluxes_x = std::get<0>(fluxes...);
        auto&& fluxes_y = std::get<1>(fluxes...);

        finite_volume_euler_(layouts, quantities.rho, quantities_new.rho, dt, fluxes_x.rho,
                             fluxes_y.rho);
        finite_volume_euler_(layouts, quantities.rhoVx, quantities_new.rhoVx, dt, fluxes_x.rhoVx,
                             fluxes_y.rhoVx);
        finite_volume_euler_(layouts, quantities.rhoVy, quantities_new.rhoVy, dt, fluxes_x.rhoVy,
                             fluxes_y.rhoVy);
        finite_volume_euler_(layouts, quantities.rhoVz, quantities_new.rhoVz, dt, fluxes_x.rhoVz,
                             fluxes_y.rhoVz);
        finite_volume_euler_(layouts, quantities.Etot, quantities_new.Etot, dt, fluxes_x.Etot,
                             fluxes_y.Etot);

        constrained_transport_(layouts, E, fluxes_x.B, fluxes_y.B);
        faraday_(layouts, quantities.B, E, quantities_new.B, dt);
    }
    if constexpr (dimension == 3)
    {
        auto&& fluxes_x = std::get<0>(fluxes...);
        auto&& fluxes_y = std::get<1>(fluxes...);
        auto&& fluxes_z = std::get<2>(fluxes...);

        finite_volume_euler_(layouts, quantities.rho, quantities_new.rho, dt, fluxes_x.rho,
                             fluxes_y.rho, fluxes_z.rho);
        finite_volume_euler_(layouts, quantities.rhoVx, quantities_new.rhoVx, dt, fluxes_x.rhoVx,
                             fluxes_y.rhoVx, fluxes_z.rhoVx);
        finite_volume_euler_(layouts, quantities.rhoVy, quantities_new.rhoVy, dt, fluxes_x.rhoVy,
                             fluxes_y.rhoVy, fluxes_z.rhoVy);
        finite_volume_euler_(layouts, quantities.rhoVz, quantities_new.rhoVz, dt, fluxes_x.rhoVz,
                             fluxes_y.rhoVz, fluxes_z.rhoVz);
        finite_volume_euler_(layouts, quantities.Etot, quantities_new.Etot, dt, fluxes_x.Etot,
                             fluxes_y.Etot, fluxes_z.Etot);

        constrained_transport_(layouts, E, fluxes_x.B, fluxes_y.B, fluxes_z.B);
        faraday_(layouts, quantities.B, E, quantities_new.B, dt);
    }
}

} // namespace PHARE::solver

#endif
