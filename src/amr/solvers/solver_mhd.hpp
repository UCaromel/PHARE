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
#include "core/data/vecfield/vecfield_component.hpp"

namespace PHARE::solver
{
template<typename MHDModel, typename AMR_Types, typename Messenger = amr::MHDMessenger<MHDModel>,
         typename ModelViews_t = MHDModelView<MHDModel>>
class SolverMHD : public ISolver<AMR_Types>
{
public:
    static constexpr auto dimension = MHDModel::dimension;

    using patch_t     = typename AMR_Types::patch_t;
    using level_t     = typename AMR_Types::level_t;
    using hierarchy_t = typename AMR_Types::hierarchy_t;

    using field_type = typename MHDModel::field_type;

    using IPhysicalModel_t = IPhysicalModel<AMR_Types>;
    using IMessenger       = amr::IMessenger<IPhysicalModel_t>;

    SolverMHD()
        : ISolver<AMR_Types>{"MHDSolver"}
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
    void reconstruction_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                         double const currentTime, double const newTime);

    void riemann_solver_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                         double const currentTime, double const newTime);

    void FV_cycle_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                   double const currentTime, double const newTime);

    void constrained_transport_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                                double const currentTime, double const newTime);

    void time_integrator_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                          double const currentTime, double const newTime);

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

    reconstruction_(*level, modelView, fromCoarser, currentTime, newTime);

    riemann_solver_(*level, modelView, fromCoarser, currentTime, newTime);

    FV_cycle_(*level, modelView, fromCoarser, currentTime, newTime);

    constrained_transport_(*level, modelView, fromCoarser, currentTime, newTime);

    time_integrator_(*level, modelView, fromCoarser, currentTime, newTime);
}

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::reconstruction_(
    level_t& level, ModelViews_t& views, Messenger& fromCoarser, double const currentTime,
    double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::reconstruction_");

    // Ampere
    // centering
    auto Fields = std::forward_as_tuple(
        views.super().rho, views.super().V(core::Component::X), views.super().V(core::Component::Y),
        views.super().V(core::Component::Z), views.super().B_CT(core::Component::X),
        views.super().B_CT(core::Component::Y), views.super().B_CT(core::Component::Z),
        views.super().P);

    /*for (auto& field : Fields)*/
    /*{*/
    /*    if constexpr (dimension == 1) {} // vec<field_type> uL_x*/
    /*    if constexpr (dimension == 2) {} // uL_x, uL_y*/
    /*    if constexpr (dimension == 3) {} // uL_x, uL_y, uL_z*/
    /*}*/
}

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::riemann_solver_(
    level_t& level, ModelViews_t& views, Messenger& fromCoarser, double const currentTime,
    double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::riemann_solver_");
}

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::FV_cycle_(level_t& level,
                                                                        ModelViews_t& views,
                                                                        Messenger& fromCoarser,
                                                                        double const currentTime,
                                                                        double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::FV_cycle_");

    auto dt = newTime - currentTime;

    // Flux difference in each direction
    // 1 cycle
}

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::constrained_transport_(
    level_t& level, ModelViews_t& views, Messenger& fromCoarser, double const currentTime,
    double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::constrained_transport_");

    auto dt = newTime - currentTime;

    // averaging B_RS(x, y, z)
    // faraday
}

template<typename MHDModel, typename AMR_Types, typename Messenger, typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, Messenger, ModelViews_t>::time_integrator_(
    level_t& level, ModelViews_t& views, Messenger& fromCoarser, double const currentTime,
    double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::time_integrator_");

    auto dt = newTime - currentTime;
}

} // namespace PHARE::solver

#endif
