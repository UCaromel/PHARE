#ifndef PHARE_SOLVER_MHD_HPP
#define PHARE_SOLVER_MHD_HPP

#include <array>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "core/mhd/mhd_quantities.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/solvers/solver.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

namespace PHARE::solver
{
template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy,
         typename Messenger    = amr::MHDMessenger<MHDModel>,
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

    FieldT rho_fx{"rho_fx", MHDQuantity::Scalar::ScalarFlux_x};
    VecFieldT rhoV_fx{"rhoV_fx", MHDQuantity::Vector::VecFlux_x};
    VecFieldT B_fx{"B_fx", MHDQuantity::Vector::VecFlux_x};
    FieldT Etot_fx{"Etot_fx", MHDQuantity::Scalar::ScalarFlux_x};

    FieldT rho_fy{"rho_fy", MHDQuantity::Scalar::ScalarFlux_y};
    VecFieldT rhoV_fy{"rhoV_fy", MHDQuantity::Vector::VecFlux_y};
    VecFieldT B_fy{"B_fy", MHDQuantity::Vector::VecFlux_y};
    FieldT Etot_fy{"Etot_fy", MHDQuantity::Scalar::ScalarFlux_y};

    FieldT rho_fz{"rho_fz", MHDQuantity::Scalar::ScalarFlux_z};
    VecFieldT rhoV_fz{"rhoV_fz", MHDQuantity::Vector::VecFlux_z};
    VecFieldT B_fz{"B_fz", MHDQuantity::Vector::VecFlux_z};
    FieldT Etot_fz{"Etot_fz", MHDQuantity::Scalar::ScalarFlux_z};

    TimeIntegratorStrategy evolve_;

public:
    SolverMHD(PHARE::initializer::PHAREDict const& dict)
        : ISolver<AMR_Types>{"MHDSolver"}
        , evolve_{dict}
    {
    }

    virtual ~SolverMHD() = default;

    std::string modelName() const override { return MHDModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;

    void registerResources(IPhysicalModel<AMR_Types>& model) override;

    // TODO make this a resourcesUser
    void allocate(IPhysicalModel<AMR_Types>& model, patch_t& patch,
                  double const allocateTime) const override;

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
                      IMessenger& fromCoarserMessenger, double const currentTime,
                      double const newTime) override;

    void onRegrid() override {}

    std::shared_ptr<ISolverModelView> make_view(level_t& level, IPhysicalModel_t& model) override
    {
        /*return std::make_shared<ModelViews_t>(level, dynamic_cast<MHDModel&>(model));*/
        throw std::runtime_error("SolverMHD::make_view not implemented");
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(rho_fx, rhoV_fx, B_fx, Etot_fx, rho_fy, rhoV_fy, B_fy, Etot_fy,
                                     rho_fz, rhoV_fz, B_fz, Etot_fz, evolve_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(rho_fx, rhoV_fx, B_fx, Etot_fx, rho_fy, rhoV_fy, B_fy, Etot_fy,
                                     rho_fz, rhoV_fz, B_fz, Etot_fz, evolve_);
    }

private:
    struct TimeSetter
    {
        template<typename QuantityAccessor>
        void operator()(QuantityAccessor accessor)
        {
            for (auto& state : views)
                views.model().resourcesManager->setTime(accessor(state), *state.patch, newTime);
        }

        ModelViews_t& views;
        double newTime;
    };
};

// -----------------------------------------------------------------------------

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::registerResources(IPhysicalModel_t& model)
{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->registerResources(rho_fx);
    mhdmodel.resourcesManager->registerResources(rhoV_fx);
    mhdmodel.resourcesManager->registerResources(B_fx);
    mhdmodel.resourcesManager->registerResources(Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(rho_fy);
        mhdmodel.resourcesManager->registerResources(rhoV_fy);
        mhdmodel.resourcesManager->registerResources(B_fy);
        mhdmodel.resourcesManager->registerResources(Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(rho_fz);
            mhdmodel.resourcesManager->registerResources(rhoV_fz);
            mhdmodel.resourcesManager->registerResources(B_fz);
            mhdmodel.resourcesManager->registerResources(Etot_fz);
        }
    }

    evolve_.registerResources(mhdmodel);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::allocate(
    IPhysicalModel_t& model, patch_t& patch, double const allocateTime) const

{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->allocate(rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(Etot_fz, patch, allocateTime);
        }
    }

    evolve_.allocate(mhdmodel, patch, allocateTime);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info)
    const

{
    auto& mhdInfo = dynamic_cast<amr::MHDMessengerInfo&>(*info);

    mhdInfo.ghostMagneticFluxesX.emplace_back(core::VecFieldNames(B_fx));

    if constexpr (dimension >= 2)
    {
        mhdInfo.ghostMagneticFluxesY.emplace_back(core::VecFieldNames(B_fy));

        if constexpr (dimension == 3)
        {
            mhdInfo.ghostMagneticFluxesZ.emplace_back(core::VecFieldNames(B_fz));
        }
    }

    evolve_.fillMessengerInfo(mhdInfo);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::advanceLevel(
    hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
    IMessenger& fromCoarserMessenger, double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::advanceLevel");

    auto& modelView   = dynamic_cast<ModelViews_t&>(view);
    auto& fromCoarser = dynamic_cast<Messenger&>(fromCoarserMessenger);
    auto level        = hierarchy.getPatchLevel(levelNumber);

    auto&& fluxes = std::forward_as_tuple(rho_fx, rhoV_fx, B_fx, Etot_fx, rho_fy, rhoV_fy, B_fy,
                                          Etot_fy, rho_fz, rhoV_fz, B_fz, Etot_fz);

    evolve_(modelView.layouts, modelView.model().state, fluxes, fromCoarser, *level, currentTime,
            newTime);
}

} // namespace PHARE::solver

#endif
