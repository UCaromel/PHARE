#ifndef PHARE_SOLVER_MHD_HPP
#define PHARE_SOLVER_MHD_HPP

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "core/data/vecfield/vecfield.hpp"
#include "core/errors.hpp"
#include "core/numerics/finite_volume_euler/finite_volume_euler.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/utilities/index/index.hpp"
#include "initializer/data_provider.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include <SAMRAI/hier/BoxContainer.h>
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/solvers/mhd_reflux_geometry.hpp"
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

    core::AllFluxes<FieldT, VecFieldT> fluxes_;

    TimeIntegratorStrategy evolve_;

    // Refluxing
    core::AllFluxes<FieldT, VecFieldT> fluxSum_;
    VecFieldT fluxSumE_{this->name() + "_fluxSumE", MHDQuantity::Vector::E};

    std::unordered_map<std::size_t, double> oldTime_;

public:
    SolverMHD(PHARE::initializer::PHAREDict const& dict)
        : ISolver<AMR_Types>{"MHDSolver"}
        , fluxes_{{"rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                  {"rhoV_fx", MHDQuantity::Vector::VecFlux_x},
                  {"B_fx", MHDQuantity::Vector::VecFlux_x},
                  {"Etot_fx", MHDQuantity::Scalar::ScalarFlux_x},

                  {"rho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                  {"rhoV_fy", MHDQuantity::Vector::VecFlux_y},
                  {"B_fy", MHDQuantity::Vector::VecFlux_y},
                  {"Etot_fy", MHDQuantity::Scalar::ScalarFlux_y},

                  {"rho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                  {"rhoV_fz", MHDQuantity::Vector::VecFlux_z},
                  {"B_fz", MHDQuantity::Vector::VecFlux_z},
                  {"Etot_fz", MHDQuantity::Scalar::ScalarFlux_z}}
        , evolve_{dict}
        , fluxSum_{{"sumRho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                   {"sumRhoV_fx", MHDQuantity::Vector::VecFlux_x},
                   {"sumB_fx", MHDQuantity::Vector::VecFlux_x},
                   {"sumEtot_fx", MHDQuantity::Scalar::ScalarFlux_x},

                   {"sumRho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                   {"sumRhoV_fy", MHDQuantity::Vector::VecFlux_y},
                   {"sumB_fy", MHDQuantity::Vector::VecFlux_y},
                   {"sumEtot_fy", MHDQuantity::Scalar::ScalarFlux_y},

                   {"sumRho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                   {"sumRhoV_fz", MHDQuantity::Vector::VecFlux_z},
                   {"sumB_fz", MHDQuantity::Vector::VecFlux_z},
                   {"sumEtot_fz", MHDQuantity::Scalar::ScalarFlux_z}}
    {
    }

    virtual ~SolverMHD() = default;

    std::string modelName() const override { return MHDModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;

    void registerResources(IPhysicalModel<AMR_Types>& model) override;

    // TODO make this a resourcesUser
    void allocate(IPhysicalModel<AMR_Types>& model, patch_t& patch,
                  double const allocateTime) const override;

    void prepareStep(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level,
                     double const currentTime) override;

    void accumulateFluxSum(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level,
                           double const coef,
                           SAMRAI::hier::CoarseFineBoundary const& cfBoundary) override;

    void resetFluxSum(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level) override;

    void reflux(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level, IMessenger& messenger,
                double const time,
                SAMRAI::hier::CoarseFineBoundary const& /*fineCfBdry*/,
                SAMRAI::hier::PatchLevel const& fineLevel) override;

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
                      IMessenger& fromCoarserMessenger, double const currentTime,
                      double const newTime) override;

    void onRegrid() override {}

    std::shared_ptr<ISolverModelView> make_view(level_t& level, IPhysicalModel_t& model) override
    {
        return std::make_shared<ModelViews_t>(level, dynamic_cast<MHDModel&>(model));
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(fluxes_, fluxSum_, fluxSumE_, evolve_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(fluxes_, fluxSum_, fluxSumE_, evolve_);
    }

private:
    void mhdNaNCheck_(MHDModel& state, level_t const& level, double time);

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

    mhdmodel.resourcesManager->registerResources(fluxes_.rho_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.B_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(fluxes_.rho_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.B_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(fluxes_.rho_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.B_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fz);
        }
    }

    mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.B_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.B_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.B_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fz);
        }
    }
    mhdmodel.resourcesManager->registerResources(fluxSumE_);

    evolve_.registerResources(mhdmodel);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::allocate(
    IPhysicalModel_t& model, patch_t& patch, double const allocateTime) const

{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->allocate(fluxes_.rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(fluxes_.rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(fluxes_.rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.Etot_fz, patch, allocateTime);
        }
    }

    mhdmodel.resourcesManager->allocate(fluxSum_.rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(fluxSum_.rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(fluxSum_.rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fz, patch, allocateTime);
        }
    }
    mhdmodel.resourcesManager->allocate(fluxSumE_, patch, allocateTime);

    evolve_.allocate(mhdmodel, patch, allocateTime);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info)
    const

{
    auto& mhdInfo = dynamic_cast<amr::MHDMessengerInfo&>(*info);

    mhdInfo.ghostMagneticFluxesX.emplace_back(fluxes_.B_fx.name());

    if constexpr (dimension >= 2)
    {
        mhdInfo.ghostMagneticFluxesY.emplace_back(fluxes_.B_fy.name());

        if constexpr (dimension == 3)
        {
            mhdInfo.ghostMagneticFluxesZ.emplace_back(fluxes_.B_fz.name());
        }
    }

    evolve_.fillMessengerInfo(mhdInfo);

    auto&& [timeFluxes, timeElectric] = evolve_.exposeFluxes();

    mhdInfo.reflux          = core::AllFluxesNames{timeFluxes};
    mhdInfo.refluxElectric  = timeElectric.name();
    mhdInfo.fluxSum         = core::AllFluxesNames{fluxSum_};
    mhdInfo.fluxSumElectric = fluxSumE_.name();

    // for the faraday in reflux
    mhdInfo.ghostElectric.emplace_back(timeElectric.name());
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::prepareStep(
    IPhysicalModel_t&, SAMRAI::hier::PatchLevel& level, double const currentTime)
{
    oldTime_[level.getLevelNumber()] = currentTime;
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::accumulateFluxSum(IPhysicalModel_t& model,
                                                SAMRAI::hier::PatchLevel& level, double const coef,
                                                SAMRAI::hier::CoarseFineBoundary const& cfBoundary)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::accumulateFluxSum");

    auto& mhdModel = dynamic_cast<MHDModel&>(model);

    for (auto& patch : level)
    {
        auto&& tf          = evolve_.exposeFluxes();
        auto& timeFluxes   = std::get<0>(tf);
        auto& timeElectric = std::get<1>(tf);

        auto const& layout      = amr::layoutFromPatch<GridLayout>(*patch);
        auto const& patchCellBox = patch->getBox();
        auto _ = mhdModel.resourcesManager->setOnPatch(*patch, fluxSum_, fluxSumE_, timeFluxes,
                                                       timeElectric);

        auto const addScalar = [&](auto& left, auto const& right,
                                   core::Point<int, dimension> const& amrIdx) {
            auto const idx = layout.AMRToLocal(amrIdx);
            left(idx) += right(idx) * coef;
        };
        auto const addVector = [&](auto& left, auto const& right,
                                   core::Point<int, dimension> const& amrIdx) {
            auto const idx = layout.AMRToLocal(amrIdx);
            left(core::Component::X)(idx) += right(core::Component::X)(idx) * coef;
            left(core::Component::Y)(idx) += right(core::Component::Y)(idx) * coef;
            left(core::Component::Z)(idx) += right(core::Component::Z)(idx) * coef;
        };

        auto const inPatchTransverse = [&](auto const& amrIdx, int normalDir) {
            for (int d = 0; d < static_cast<int>(dimension); ++d)
            {
                if (d == normalDir) continue;
                if (amrIdx[d] < patchCellBox.lower(d) || amrIdx[d] > patchCellBox.upper(d))
                    return false;
            }
            return true;
        };

        // Pass 1: conserved flux accumulation (codim-1 boundaries)
        for (auto const& bb : cfBoundary.getBoundaries(patch->getGlobalId(), 1))
        {
            auto const location  = bb.getLocationIndex();
            bool const isLower   = (location % 2 == 0);
            int const normalDir  = location / 2;

            for (auto const& amrIdx : amr::phare_box_from<dimension>(bb.getBox()))
            {
                if (!inPatchTransverse(amrIdx, normalDir)) continue;
                auto readIdx = amrIdx;
                if (isLower) readIdx[normalDir] += 1;

                if (normalDir == core::dirX)
                {
                    addScalar(fluxSum_.rho_fx, timeFluxes.rho_fx, readIdx);
                    addVector(fluxSum_.rhoV_fx, timeFluxes.rhoV_fx, readIdx);
                    addVector(fluxSum_.B_fx, timeFluxes.B_fx, readIdx);
                    addScalar(fluxSum_.Etot_fx, timeFluxes.Etot_fx, readIdx);
                }
                else if (normalDir == core::dirY)
                {
                    addScalar(fluxSum_.rho_fy, timeFluxes.rho_fy, readIdx);
                    addVector(fluxSum_.rhoV_fy, timeFluxes.rhoV_fy, readIdx);
                    addVector(fluxSum_.B_fy, timeFluxes.B_fy, readIdx);
                    addScalar(fluxSum_.Etot_fy, timeFluxes.Etot_fy, readIdx);
                }
                else if constexpr (dimension == 3)
                {
                    addScalar(fluxSum_.rho_fz, timeFluxes.rho_fz, readIdx);
                    addVector(fluxSum_.rhoV_fz, timeFluxes.rhoV_fz, readIdx);
                    addVector(fluxSum_.B_fz, timeFluxes.B_fz, readIdx);
                    addScalar(fluxSum_.Etot_fz, timeFluxes.Etot_fz, readIdx);
                }
            }
        }

        // Pass 2: E field accumulation. Geometry (codim-1 vs codim-2, transverse clipping,
        // Ez primal-endpoint patching) lives in the dim-generic enumerator, which returns
        // read-shifted, box-deduped containers per E component. Box-dedup (simplify)
        // replaces the per-index seenEzNodes set.
        auto const eBoxes = reflux_geometry::cfElectricBoxes<dimension>(
            cfBoundary, patch->getGlobalId(), patchCellBox);

        auto const accumulateE = [&](SAMRAI::hier::BoxContainer const& boxes,
                                     core::Component comp) {
            for (auto const& box : boxes)
                for (auto const& amrIdx : amr::phare_box_from<dimension>(box))
                    addScalar(fluxSumE_(comp), timeElectric(comp), amrIdx);
        };

        accumulateE(eBoxes.ex, core::Component::X);
        accumulateE(eBoxes.ey, core::Component::Y);
        accumulateE(eBoxes.ez, core::Component::Z);
    }
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::resetFluxSum(
    IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level)
{
    auto& mhdModel = dynamic_cast<MHDModel&>(model);

    for (auto& patch : level)
    {
        auto const& layout = amr::layoutFromPatch<GridLayout>(*patch);
        auto _             = mhdModel.resourcesManager->setOnPatch(*patch, fluxSum_, fluxSumE_);

        evalFluxesOnGhostBox(
            layout, [&](auto& left, auto const&... args) mutable { left(args...) = 0.0; },
            fluxSum_);

        layout.evalOnGhostBox(fluxSumE_(core::Component::X), [&](auto const&... args) mutable {
            fluxSumE_(core::Component::X)(args...) = 0.0;
        });

        layout.evalOnGhostBox(fluxSumE_(core::Component::Y), [&](auto const&... args) mutable {
            fluxSumE_(core::Component::Y)(args...) = 0.0;
        });

        layout.evalOnGhostBox(fluxSumE_(core::Component::Z), [&](auto const&... args) mutable {
            fluxSumE_(core::Component::Z)(args...) = 0.0;
        });
    }
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::reflux(
    IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level, IMessenger& messenger,
    double const time, SAMRAI::hier::CoarseFineBoundary const& /*fineCfBdry*/,
    SAMRAI::hier::PatchLevel const& fineLevel)
{
    auto& bc           = dynamic_cast<Messenger&>(messenger);
    auto& mhdModel     = dynamic_cast<MHDModel&>(model);
    auto&& tf          = evolve_.exposeFluxes();
    auto& timeFluxes   = std::get<0>(tf);
    auto& timeElectric = std::get<1>(tf);
    auto& state        = mhdModel.state;
    double const dt    = time - oldTime_[level.getLevelNumber()];

    constexpr auto dirX = core::dirX;
    constexpr auto dirY = core::dirY;
    constexpr auto dirZ = core::dirZ;

    // Build coarsened fine domain from global fine boxes (MPI-collective, done once per call)
    auto const& globalFineBoxes = fineLevel.getBoxLevel()->getGlobalizedVersion().getGlobalBoxes();
    auto const ratio = fineLevel.getRatioToCoarserLevel();

    std::vector<SAMRAI::hier::Box> coarsenedFine;
    for (auto const& box : globalFineBoxes)
        coarsenedFine.push_back(SAMRAI::hier::Box::coarsen(box, ratio));

    for (auto& coarsePatch : level)
    {
        auto const& patchAMRBox = coarsePatch->getBox();
        auto const& layout      = amr::layoutFromPatch<GridLayout>(*coarsePatch);
        auto _ = mhdModel.resourcesManager->setOnPatch(
            *coarsePatch, state.rho, state.rhoV, state.Etot, state.B, fluxSum_, fluxSumE_,
            timeFluxes, timeElectric);

        // Pass 1: hydro flux correction. Coarse cells adjacent to the CF boundary for
        // (dir, side), box-deduped across all coarsened-fine boxes (replaces seenFlux). The
        // boundary flux read coordinate is recovered per cell from amrIdx[dir]
        // (= isLower ? amrIdx[dir]+1 : amrIdx[dir]), same reconstruction as the B pass.
        for (int dir = 0; dir < static_cast<int>(dimension); ++dir)
            for (int side = 0; side < 2; ++side)
            {
                bool const isLower      = (side == 0);
                int const sign          = isLower ? +1 : -1;
                double const hydroScale = sign * dt / layout.meshSize()[dir];

                auto const cells = reflux_geometry::cfAdjacentCoarseCells(
                    dir, side, patchAMRBox, coarsenedFine, /*expand=*/0);

                for (auto const& ccBox : cells)
                    for (auto const& amrIdx : amr::phare_box_from<dimension>(ccBox))
                    {
                        auto fReadIdx   = amrIdx;
                        fReadIdx[dir]   = isLower ? amrIdx[dir] + 1 : amrIdx[dir];
                        auto const idxF = layout.AMRToLocal(fReadIdx);
                        auto const idx  = layout.AMRToLocal(amrIdx);

                        if (dir == dirX)
                        {
                            state.rho(idx) += hydroScale * (timeFluxes.rho_fx(idxF) - fluxSum_.rho_fx(idxF));
                            state.rhoV(core::Component::X)(idx) += hydroScale * (timeFluxes.rhoV_fx(core::Component::X)(idxF) - fluxSum_.rhoV_fx(core::Component::X)(idxF));
                            state.rhoV(core::Component::Y)(idx) += hydroScale * (timeFluxes.rhoV_fx(core::Component::Y)(idxF) - fluxSum_.rhoV_fx(core::Component::Y)(idxF));
                            state.rhoV(core::Component::Z)(idx) += hydroScale * (timeFluxes.rhoV_fx(core::Component::Z)(idxF) - fluxSum_.rhoV_fx(core::Component::Z)(idxF));
                            state.Etot(idx) += hydroScale * (timeFluxes.Etot_fx(idxF) - fluxSum_.Etot_fx(idxF));
                        }
                        else if (dir == dirY)
                        {
                            state.rho(idx) += hydroScale * (timeFluxes.rho_fy(idxF) - fluxSum_.rho_fy(idxF));
                            state.rhoV(core::Component::X)(idx) += hydroScale * (timeFluxes.rhoV_fy(core::Component::X)(idxF) - fluxSum_.rhoV_fy(core::Component::X)(idxF));
                            state.rhoV(core::Component::Y)(idx) += hydroScale * (timeFluxes.rhoV_fy(core::Component::Y)(idxF) - fluxSum_.rhoV_fy(core::Component::Y)(idxF));
                            state.rhoV(core::Component::Z)(idx) += hydroScale * (timeFluxes.rhoV_fy(core::Component::Z)(idxF) - fluxSum_.rhoV_fy(core::Component::Z)(idxF));
                            state.Etot(idx) += hydroScale * (timeFluxes.Etot_fy(idxF) - fluxSum_.Etot_fy(idxF));
                        }
                        else if constexpr (dimension == 3)
                        {
                            state.rho(idx) += hydroScale * (timeFluxes.rho_fz(idxF) - fluxSum_.rho_fz(idxF));
                            state.rhoV(core::Component::X)(idx) += hydroScale * (timeFluxes.rhoV_fz(core::Component::X)(idxF) - fluxSum_.rhoV_fz(core::Component::X)(idxF));
                            state.rhoV(core::Component::Y)(idx) += hydroScale * (timeFluxes.rhoV_fz(core::Component::Y)(idxF) - fluxSum_.rhoV_fz(core::Component::Y)(idxF));
                            state.rhoV(core::Component::Z)(idx) += hydroScale * (timeFluxes.rhoV_fz(core::Component::Z)(idxF) - fluxSum_.rhoV_fz(core::Component::Z)(idxF));
                            state.Etot(idx) += hydroScale * (timeFluxes.Etot_fz(idxF) - fluxSum_.Etot_fz(idxF));
                        }
                    }
            }

        // Pass 2: B correction via Faraday. Coarse Yee B-faces on the CF boundary are
        // enumerated and box-deduped per (component, dir, side) across all coarsened-fine
        // boxes (replaces seenBx/By/Bz). The normal-direction E read coordinate is
        // recovered per face from amrIdx[dir]. Hydro (state.rho/rhoV/Etot) and B (state.B)
        // touch disjoint fields, so this runs after all hydro corrections.
        for (int dir = 0; dir < static_cast<int>(dimension); ++dir)
            for (int side = 0; side < 2; ++side)
            {
                bool const isLower  = (side == 0);
                int const sign      = isLower ? +1 : -1;
                double const bScale = -sign * dt / layout.meshSize()[dir];

                for (auto const& t : reflux_geometry::faradayTerms(dir))
                {
                    auto const faces = reflux_geometry::cfBFaceBoxes(
                        layout, t.bQty, dir, side, patchAMRBox, coarsenedFine);

                    for (auto const& box : faces)
                        for (auto const& amrIdx : amr::phare_box_from<dimension>(box))
                        {
                            if (reflux_geometry::bFaceInsideFine(layout, t.bQty, dir, amrIdx,
                                                                 coarsenedFine))
                                continue;

                            auto eReadIdx  = amrIdx;
                            eReadIdx[dir]  = isLower ? amrIdx[dir] + 1 : amrIdx[dir];
                            auto const idxE = layout.AMRToLocal(eReadIdx);
                            auto const idx  = layout.AMRToLocal(amrIdx);

                            auto const tE = timeElectric(t.eComp)(idxE);
                            auto const fE = fluxSumE_(t.eComp)(idxE);
                            state.B(t.bComp)(idx) += t.eSign * bScale * (tE - fE);
                        }
                }
            }
    }

    bc.fillMomentsGhosts(state, level, time);
    bc.fillMagneticGhosts(state.B, level, time);
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

    try
    {
        evolve_(modelView.model(), modelView.model().state, fluxes_, fromCoarser, *level,
                currentTime, newTime);

        mhdNaNCheck_(modelView.model(), *level, currentTime);
    }
    catch (core::DictionaryException& ex)
    {
        PHARE_LOG_ERROR(ex());
    }

    if (core::mpi::any(core::Errors::instance().any()))
        throw core::DictionaryException{}("ID", "SolverMHD::advanceLevel");
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::mhdNaNCheck_(
    MHDModel& model, level_t const& level, double time)
{
    auto& rm  = model.resourcesManager;
    auto& rho = model.state.rho;

    auto check_nans = [&](auto const& field, auto const& origin,
                          core::MeshIndex<MHDModel::dimension> const& index) {
        if (std::isnan(field(index)))
        {
            std::stringstream ss;
            ss << "NaN detected in MHD field at index " << index << " on patch of origin " << origin
               << " on level " << level.getLevelNumber() << " at time " << time;
            core::DictionaryException ex{"cause", ss.str()};
            throw ex;
        }
    };

    for (auto const& patch : rm->enumerate(level, rho))
    {
        auto layout = amr::layoutFromPatch<GridLayout>(*patch);
        layout.evalOnGhostBox(
            rho, [&](auto const&... args) { check_nans(rho, layout.origin(), {args...}); });
    }
}

} // namespace PHARE::solver

#endif
