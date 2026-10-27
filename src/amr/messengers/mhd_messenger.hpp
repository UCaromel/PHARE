#ifndef PHARE_MHD_MESSENGER_HPP
#define PHARE_MHD_MESSENGER_HPP

#include "phare_mpi.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include "amr/data/field/coarsening/electric_field_coarsener.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/mhd_flux_coarsener.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/composite_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_composite_refiner.hpp"
#include "amr/messengers/refinement_config.hpp"
#include "amr/data/field/time_interpolate/field_linear_time_interpolate.hpp"
#include "amr/messengers/refiner.hpp"
#include "amr/messengers/refiner_pool.hpp"
#include "amr/messengers/synchronizer_pool.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/messenger_info.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/data/field/refine/field_refine_patch_strategy.hpp"
#include "amr/messengers/mhd_messenger_fwd.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/data/field/refine/magnetic_refine_patch_strategy.hpp"
#include "core/numerics/rk_stage_context.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"


#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/RefineOperator.h"
#include "SAMRAI/hier/CoarsenOperator.h"

#include <memory>
#include <optional>
#include <string>
#include <type_traits>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel, typename TemporalTransfer>
    class MHDMessenger : public IMessenger<typename MHDModel::Interface>
    {
        using amr_types     = MHDModel::amr_types;
        using level_t       = amr_types::level_t;
        using patch_t       = amr_types::patch_t;
        using hierarchy_t   = amr_types::hierarchy_t;
        using level_ptr     = amr_types::level_ptr;
        using hierarchy_ptr = amr_types::hierarchy_ptr;

        using IPhysicalModel    = MHDModel::Interface;
        using FieldT            = MHDModel::field_type;
        using VecFieldT         = MHDModel::vecfield_type;
        using MHDStateT         = MHDModel::state_type;
        using GridLayoutT       = MHDModel::gridlayout_type;
        using GridT             = MHDModel::grid_type;
        using ResourcesManagerT = MHDModel::resources_manager_type;
        using BoundaryManagerT  = MHDModel::boundary_manager_type;
        using FieldDataT        = FieldData<GridLayoutT, GridT, core::MHDQuantity::Scalar>;
        using VectorFieldDataT  = TensorFieldData<1, GridLayoutT, GridT, core::MHDQuantity>;

        static constexpr auto dimension = MHDModel::dimension;

        // linear interpolation between the solver's old and current states, otherwise static
        // refinement from the transfer's assembled state
        static constexpr bool linearInTime
            = std::is_same_v<TemporalTransfer, Linear2TemporalTransfer<MHDModel>>;

    public:
        using temporal_transfer_type = TemporalTransfer;

        static constexpr std::size_t rootLevelNumber = 0;
        static inline std::string const stratName    = "MHDModel-MHDModel";

        MHDMessenger(std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager,
                     std::shared_ptr<BoundaryManagerT> boundaryManager, int const firstLevel,
                     RefinementConfig const& refinementConfig = {})
            : temporalTransfer_{resourcesManager}
            , resourcesManager_{std::move(resourcesManager)}
            , boundaryManager_{std::move(boundaryManager)}
            , firstLevel_{firstLevel}
        {
            makeRefineOperators_(refinementConfig);

            resourcesManager_->registerResources(temporalTransfer_);
        }

        virtual ~MHDMessenger() = default;

        void allocate(patch_t& patch, double const allocateTime) const override
        {
            resourcesManager_->allocate(temporalTransfer_, patch, allocateTime);
        }


        void registerQuantities(std::unique_ptr<IMessengerInfo> /*fromCoarserInfo*/,
                                std::unique_ptr<IMessengerInfo> fromFinerInfo) override
        {
            std::unique_ptr<MHDMessengerInfo> mhdInfo{
                dynamic_cast<MHDMessengerInfo*>(fromFinerInfo.release())};
            temporalTransfer_.registerQuantities(*mhdInfo);

            if (temporalTransfer_.hasCoarseFine())
                registerCoarseFineComms_(mhdInfo);
            else
                registerNoCoarseFineGhostComms_(*mhdInfo);
        }

        void registerCoarseFineComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo)
        {
            auto b_id = resourcesManager_->getID(mhdInfo->modelMagnetic);

            if (!b_id)
            {
                throw std::runtime_error(
                    "MHDMessengerStrategy: missing magnetic field variable IDs");
            }

            magneticRefinePatchStrategy_.registerIDs(*b_id);

            BalgoInit.registerRefine(*b_id, *b_id, *b_id, BfieldRegridOp_,
                                     overwriteInteriorTFfillPattern);

            BregridAlgo.registerRefine(*b_id, *b_id, *b_id, BfieldRegridOp_,
                                       overwriteInteriorTFfillPattern);

            auto e_id = resourcesManager_->getID(mhdInfo->modelElectric);

            if (!e_id)
            {
                throw std::runtime_error(
                    "MHDMessengerStrategy: missing electric field variable IDs");
            }

            // EalgoPatchGhost.registerRefine(*e_id, *e_id, *e_id, EfieldRefineOp_,
            //                                nonOverwriteInteriorTFfillPattern);

            // refluxing
            // we first want to coarsen the flux sum onto the coarser level
            auto rho_fx_reflux_id  = resourcesManager_->getID(mhdInfo->reflux.rho_fx);
            auto rhoV_fx_reflux_id = resourcesManager_->getID(mhdInfo->reflux.rhoV_fx);
            auto Etot_fx_reflux_id = resourcesManager_->getID(mhdInfo->reflux.Etot_fx);

            if (!rho_fx_reflux_id or !rhoV_fx_reflux_id or !Etot_fx_reflux_id)
            {
                throw std::runtime_error(
                    "MHDMessenger: missing reflux variable IDs for fluxes in x direction");
            }

            auto rho_fx_fluxsum_id  = resourcesManager_->getID(mhdInfo->fluxSum.rho_fx);
            auto rhoV_fx_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSum.rhoV_fx);
            auto Etot_fx_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSum.Etot_fx);


            if (!rho_fx_fluxsum_id or !rhoV_fx_fluxsum_id or !Etot_fx_fluxsum_id)
            {
                throw std::runtime_error(
                    "MHDMessenger: missing flux sum variable IDs for fluxes in x direction");
            }


            // all of the fluxes fx are defined on the same faces no matter the component, so we
            // just need a different fill pattern per direction
            HydroXrefluxAlgo.registerCoarsen(*rho_fx_reflux_id, *rho_fx_fluxsum_id,
                                             mhdFluxCoarseningOp_);
            HydroXrefluxAlgo.registerCoarsen(*rhoV_fx_reflux_id, *rhoV_fx_fluxsum_id,
                                             mhdVecFluxCoarseningOp_);
            HydroXrefluxAlgo.registerCoarsen(*Etot_fx_reflux_id, *Etot_fx_fluxsum_id,
                                             mhdFluxCoarseningOp_);

            // we then need to refill the ghosts so that they agree with the newly refluxed
            // cells
            HydroXpatchGhostRefluxedAlgo.registerRefine(*rho_fx_reflux_id, *rho_fx_reflux_id,
                                                        *rho_fx_reflux_id, mhdFluxRefineOp_,
                                                        nonOverwriteInteriorTFfillPattern);
            HydroXpatchGhostRefluxedAlgo.registerRefine(*rhoV_fx_reflux_id, *rhoV_fx_reflux_id,
                                                        *rhoV_fx_reflux_id, mhdVecFluxRefineOp_,
                                                        nonOverwriteInteriorTFfillPattern);
            HydroXpatchGhostRefluxedAlgo.registerRefine(*Etot_fx_reflux_id, *Etot_fx_reflux_id,
                                                        *Etot_fx_reflux_id, mhdFluxRefineOp_,
                                                        nonOverwriteInteriorTFfillPattern);

            if constexpr (dimension >= 2)
            {
                auto rho_fy_reflux_id  = resourcesManager_->getID(mhdInfo->reflux.rho_fy);
                auto rhoV_fy_reflux_id = resourcesManager_->getID(mhdInfo->reflux.rhoV_fy);
                auto Etot_fy_reflux_id = resourcesManager_->getID(mhdInfo->reflux.Etot_fy);

                if (!rho_fy_reflux_id or !rhoV_fy_reflux_id or !Etot_fy_reflux_id)
                {
                    throw std::runtime_error(
                        "MHDMessenger: missing reflux variable IDs for fluxes in y direction");
                }

                auto rho_fy_fluxsum_id  = resourcesManager_->getID(mhdInfo->fluxSum.rho_fy);
                auto rhoV_fy_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSum.rhoV_fy);
                auto Etot_fy_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSum.Etot_fy);

                if (!rho_fy_fluxsum_id or !rhoV_fy_fluxsum_id or !Etot_fy_fluxsum_id)
                {
                    throw std::runtime_error(
                        "MHDMessenger: missing flux sum variable IDs for fluxes in y direction");
                }

                HydroYrefluxAlgo.registerCoarsen(*rho_fy_reflux_id, *rho_fy_fluxsum_id,
                                                 mhdFluxCoarseningOp_);
                HydroYrefluxAlgo.registerCoarsen(*rhoV_fy_reflux_id, *rhoV_fy_fluxsum_id,
                                                 mhdVecFluxCoarseningOp_);
                HydroYrefluxAlgo.registerCoarsen(*Etot_fy_reflux_id, *Etot_fy_fluxsum_id,
                                                 mhdFluxCoarseningOp_);

                HydroYpatchGhostRefluxedAlgo.registerRefine(*rho_fy_reflux_id, *rho_fy_reflux_id,
                                                            *rho_fy_reflux_id, mhdFluxRefineOp_,
                                                            nonOverwriteInteriorTFfillPattern);
                HydroYpatchGhostRefluxedAlgo.registerRefine(*rhoV_fy_reflux_id, *rhoV_fy_reflux_id,
                                                            *rhoV_fy_reflux_id, mhdVecFluxRefineOp_,
                                                            nonOverwriteInteriorTFfillPattern);
                HydroYpatchGhostRefluxedAlgo.registerRefine(*Etot_fy_reflux_id, *Etot_fy_reflux_id,
                                                            *Etot_fy_reflux_id, mhdFluxRefineOp_,
                                                            nonOverwriteInteriorTFfillPattern);

                if constexpr (dimension == 3)
                {
                    auto rho_fz_reflux_id  = resourcesManager_->getID(mhdInfo->reflux.rho_fz);
                    auto rhoV_fz_reflux_id = resourcesManager_->getID(mhdInfo->reflux.rhoV_fz);
                    auto Etot_fz_reflux_id = resourcesManager_->getID(mhdInfo->reflux.Etot_fz);


                    if (!rho_fz_reflux_id or !rhoV_fz_reflux_id or !Etot_fz_reflux_id)
                    {
                        throw std::runtime_error(
                            "MHDMessenger: missing reflux variable IDs for fluxes in z direction");
                    }

                    auto rho_fz_fluxsum_id  = resourcesManager_->getID(mhdInfo->fluxSum.rho_fz);
                    auto rhoV_fz_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSum.rhoV_fz);
                    auto Etot_fz_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSum.Etot_fz);

                    if (!rho_fz_fluxsum_id or !rhoV_fz_fluxsum_id or !Etot_fz_fluxsum_id)
                    {
                        throw std::runtime_error("MHDMessenger: missing flux sum variable IDs for "
                                                 "fluxes in z direction");
                    }

                    HydroZrefluxAlgo.registerCoarsen(*rho_fz_reflux_id, *rho_fz_fluxsum_id,
                                                     mhdFluxCoarseningOp_);
                    HydroZrefluxAlgo.registerCoarsen(*rhoV_fz_reflux_id, *rhoV_fz_fluxsum_id,
                                                     mhdVecFluxCoarseningOp_);
                    HydroZrefluxAlgo.registerCoarsen(*Etot_fz_reflux_id, *Etot_fz_fluxsum_id,
                                                     mhdFluxCoarseningOp_);


                    HydroZpatchGhostRefluxedAlgo.registerRefine(
                        *rho_fz_reflux_id, *rho_fz_reflux_id, *rho_fz_reflux_id, mhdFluxRefineOp_,
                        nonOverwriteInteriorTFfillPattern);
                    HydroZpatchGhostRefluxedAlgo.registerRefine(
                        *rhoV_fz_reflux_id, *rhoV_fz_reflux_id, *rhoV_fz_reflux_id,
                        mhdVecFluxRefineOp_, nonOverwriteInteriorTFfillPattern);
                    HydroZpatchGhostRefluxedAlgo.registerRefine(
                        *Etot_fz_reflux_id, *Etot_fz_reflux_id, *Etot_fz_reflux_id,
                        mhdFluxRefineOp_, nonOverwriteInteriorTFfillPattern);
                }
            }

            auto e_reflux_id = resourcesManager_->getID(mhdInfo->refluxElectric);

            auto e_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSumElectric);

            if (!e_reflux_id or !e_fluxsum_id)
            {
                throw std::runtime_error(
                    "MHDMessenger: missing electric refluxing field variable IDs");
            }

            ErefluxAlgo.registerCoarsen(*e_reflux_id, *e_fluxsum_id, electricFieldCoarseningOp_);

            EpatchGhostRefluxedAlgo.registerRefine(*e_reflux_id, *e_reflux_id, *e_reflux_id,
                                                   EfieldRefineOp_,
                                                   nonOverwriteInteriorTFfillPattern);

            registerGhostComms_(mhdInfo);
            registerInitComms_(mhdInfo);
        }



        void registerLevel(hierarchy_ptr const& hierarchy, int const levelNumber) override
        {
            hierarchy_       = hierarchy;
            auto const level = hierarchy->getPatchLevel(levelNumber);

            if (levelNumber != rootLevelNumber && !temporalTransfer_.hasCoarseFine())
                throw std::runtime_error(
                    "MHDMessenger: this temporal transfer has no coarse-fine path (fourth-order "
                    "MHD on refined levels needs the SSPRK4_5 time integrator)");

            // magPatchGhostsRefineSchedules[levelNumber]
            //     = BalgoPatchGhost.createSchedule(level, &magneticRefinePatchStrategy_);

            // elecPatchGhostsRefineSchedules[levelNumber] = EalgoPatchGhost.createSchedule(level);

            if (temporalTransfer_.hasCoarseFine())
            {
                EpatchGhostRefluxedSchedules[levelNumber]
                    = EpatchGhostRefluxedAlgo.createSchedule(level);
                HydroXpatchGhostRefluxedSchedules[levelNumber]
                    = HydroXpatchGhostRefluxedAlgo.createSchedule(level);
                HydroYpatchGhostRefluxedSchedules[levelNumber]
                    = HydroYpatchGhostRefluxedAlgo.createSchedule(level);
                HydroZpatchGhostRefluxedSchedules[levelNumber]
                    = HydroZpatchGhostRefluxedAlgo.createSchedule(level);
            }

            elecGhostsRefiners_.registerLevel(hierarchy, level);

            rhoGhostsRefiners_.registerLevel(hierarchy, level);
            // velGhostsRefiners_.registerLevel(hierarchy, level);
            // pressureGhostsRefiners_.registerLevel(hierarchy, level);

            momentumGhostsRefiners_.registerLevel(hierarchy, level);
            totalEnergyGhostsRefiners_.registerLevel(hierarchy, level);

            magFluxesXGhostRefiners_.registerLevel(hierarchy, level);
            magFluxesYGhostRefiners_.registerLevel(hierarchy, level);
            magFluxesZGhostRefiners_.registerLevel(hierarchy, level);

            magGhostsRefiners_.registerLevel(hierarchy, level);
            if (levelNumber == rootLevelNumber)
                magMaxModelRefiners_.registerLevel(hierarchy, level);

            if (temporalTransfer_.hasCoarseFine() && levelNumber != rootLevelNumber)
            {
                // refluxing
                auto const& coarseLevel       = hierarchy->getPatchLevel(levelNumber - 1);
                ErefluxSchedules[levelNumber] = ErefluxAlgo.createSchedule(coarseLevel, level);
                HydroXrefluxSchedules[levelNumber]
                    = HydroXrefluxAlgo.createSchedule(coarseLevel, level);
                HydroYrefluxSchedules[levelNumber]
                    = HydroYrefluxAlgo.createSchedule(coarseLevel, level);
                HydroZrefluxSchedules[levelNumber]
                    = HydroZrefluxAlgo.createSchedule(coarseLevel, level);

                // refinement
                magInitRefineSchedules[levelNumber] = BalgoInit.createSchedule(
                    level, nullptr, levelNumber - 1, hierarchy, &magneticRefinePatchStrategy_);

                densityInitRefiners_.registerLevel(hierarchy, level);
                momentumInitRefiners_.registerLevel(hierarchy, level);
                totalEnergyInitRefiners_.registerLevel(hierarchy, level);
            }
        }


        void regrid(hierarchy_ptr const& hierarchy, int const levelNumber,
                    level_ptr const& oldLevel, IPhysicalModel& model,
                    double const initDataTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            auto& mhdModel = static_cast<MHDModel&>(model);
            auto level     = hierarchy->getPatchLevel(levelNumber);

            bool isRegriddingL0 = levelNumber == 0 and oldLevel;

            magneticRegriding_(hierarchy, level, oldLevel, initDataTime);

            densityInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            momentumInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            totalEnergyInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);

            fillGhostsAtInitTime_(mhdModel, *level, initDataTime);

            // magPatchGhostsRefineSchedules[levelNumber]->fillData(initDataTime);
            // elecPatchGhostsRefineSchedules[levelNumber]->fillData(initDataTime);
        }


        std::string fineModelName() const override { return MHDModel::model_name; }

        std::string coarseModelName() const override { return MHDModel::model_name; }

        std::unique_ptr<IMessengerInfo> emptyInfoFromCoarser() override
        {
            return std::make_unique<MHDMessengerInfo>();
        }

        std::unique_ptr<IMessengerInfo> emptyInfoFromFiner() override
        {
            return std::make_unique<MHDMessengerInfo>();
        }

        void initLevel(IPhysicalModel& model, level_t& level, double const initDataTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            auto levelNumber = level.getLevelNumber();

            auto& mhdModel = static_cast<MHDModel&>(model);

            magInitRefineSchedules[levelNumber]->fillData(initDataTime);
            densityInitRefiners_.fill(levelNumber, initDataTime);
            momentumInitRefiners_.fill(levelNumber, initDataTime);
            totalEnergyInitRefiners_.fill(levelNumber, initDataTime);

            fillGhostsAtInitTime_(mhdModel, level, initDataTime);
        }

        void firstStep(IPhysicalModel& model, level_t& level, double const /*currentTime*/,
                       double const prevCoarserTIme, double const newCoarserTime) final
        {
            temporalTransfer_.firstStep(level, hierarchy_, prevCoarserTIme, newCoarserTime);
        }


        void lastStep(IPhysicalModel& model, level_t& level) final {}


        // the old state used for time interpolation is snapshotted by SolverMHD::prepareStep
        void prepareStep(IPhysicalModel& /*model*/, level_t& /*level*/,
                         double /*currentTime*/) final
        {
        }

        void fillRootGhosts(IPhysicalModel& model, level_t& level, double const initDataTime) final
        {
            auto& mhdModel = static_cast<MHDModel&>(model);
            temporalTransfer_.levelMade(level, hierarchy_, initDataTime);
            fillConservativeGhosts(mhdModel.state, level, initDataTime,
                                   core::RKStageContext::endOfStep(initDataTime));

            // the faces on both sides of a periodic boundary are one face, initialised at two
            // coordinates: make the patches holding them agree once, before anything reads B.
            // Only after the refill: the max runs over whole ghost overlaps, and ghosts still
            // holding the initializer's values would leak into the interior.
            magMaxModelRefiners_.fill(mhdModel.state.B, level.getLevelNumber(), initDataTime);
        }

        void synchronize(level_t& level) final {}

        void reflux(int const coarserLevelNumber, int const fineLevelNumber,
                    double const syncTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            ErefluxSchedules[fineLevelNumber]->coarsenData();
            HydroXrefluxSchedules[fineLevelNumber]->coarsenData();
            HydroYrefluxSchedules[fineLevelNumber]->coarsenData();
            HydroZrefluxSchedules[fineLevelNumber]->coarsenData();

            EpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
            HydroXpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
            HydroYpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
            HydroZpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
        }

        void postSynchronize(IPhysicalModel& model, level_t& level, double const time) override
        {
            // The ghosts for B are obtained in the solver's reflux_euler. For B, this is because
            // refluxing is done through faraday which is computed on the ghost box for the other
            // quantities, the ghosts are filled in the end of the euler step anyways.
        }

        void fillConservativeGhosts(MHDStateT& state, level_t const& level, double const fillTime,
                                    core::RKStageContext const& context)
        {
            setNaNsOnConservativeGhosts_(state, level);
            temporalTransfer_.prepareTarget(state, level, context);
            fillConservativeGhosts_(state, level, fillTime);
        }

        void fillMagneticFluxesXGhosts(VecFieldT& Fx_B, level_t const& level, double const fillTime)
        {
            setNaNsOnVecfieldGhosts(Fx_B, level);
            magFluxesXGhostRefiners_.fill(Fx_B, level.getLevelNumber(), fillTime);
        }

        void fillMagneticFluxesYGhosts(VecFieldT& Fy_B, level_t const& level, double const fillTime)
        {
            setNaNsOnVecfieldGhosts(Fy_B, level);
            magFluxesYGhostRefiners_.fill(Fy_B, level.getLevelNumber(), fillTime);
        }

        void fillMagneticFluxesZGhosts(VecFieldT& Fz_B, level_t const& level, double const fillTime)
        {
            setNaNsOnVecfieldGhosts(Fz_B, level);
            magFluxesZGhostRefiners_.fill(Fz_B, level.getLevelNumber(), fillTime);
        }

        void fillElectricGhosts(VecFieldT& E, level_t const& level, double const fillTime)
        {
            setNaNsOnVecfieldGhosts(E, level);
            elecGhostsRefiners_.fill(E, level.getLevelNumber(), fillTime);
        }

        std::string name() override { return stratName; }


    private:
        // The init and regrid schedules interpolate in space only, reading coarser levels at their
        // latest time. A level made between two coarser syncs gets ghosts from coarser data ahead
        // of its init time: refill them, interpolated in time as in a step. The refill reads the
        // coarser levels' old data, which their solver sets when they are made
        // (MultiPhysicsIntegrator::initializeLevelData).
        void fillGhostsAtInitTime_(MHDModel& model, level_t& level, double const initDataTime)
        {
            bool const refill = temporalTransfer_.levelMade(level, hierarchy_, initDataTime);
            if (level.getLevelNumber() == rootLevelNumber || !refill)
                return;
            fillConservativeGhosts(model.state, level, initDataTime,
                                   core::RKStageContext::endOfStep(initDataTime));
        }

        void setNaNsOnConservativeGhosts_(MHDStateT& state, level_t const& level)
        {
            setNaNsOnFieldGhosts(state.rho, level);
            setNaNsOnVecfieldGhosts(state.rhoV, level);
            setNaNsOnFieldGhosts(state.Etot, level);
            setNaNsOnVecfieldGhosts(state.B, level);
        }

        void fillConservativeGhosts_(MHDStateT& state, level_t const& level, double const fillTime)
        {
            magGhostsRefiners_.fill(state.B, level.getLevelNumber(), fillTime);
            rhoGhostsRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumGhostsRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyGhostsRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);
        }

        // without a coarse-fine path, ghosts come from the same level and its periodic images only
        void registerNoCoarseFineGhostComms_(MHDMessengerInfo const& info)
        {
            registerGhostRefinePatchStrategies_(elecPatchStrats, info.ghostElectric);
            for (std::size_t i = 0; i < info.ghostElectric.size(); ++i)
                elecGhostsRefiners_.addStaticRefiner(info.ghostElectric[i], nullptr,
                                                     info.ghostElectric[i], noOverlapFillPattern,
                                                     elecPatchStrats[i]);

            registerGhostRefinePatchStrategies_(rhoPatchStrats, info.ghostDensity);
            for (std::size_t i = 0; i < info.ghostDensity.size(); ++i)
                rhoGhostsRefiners_.addStaticRefiner(
                    info.ghostDensity[i], nullptr, info.ghostDensity[i],
                    nonOverwriteFieldFillPattern, rhoPatchStrats[i]);
            registerGhostRefinePatchStrategies_(momentumPatchStrats, info.ghostMomentum);
            for (std::size_t i = 0; i < info.ghostMomentum.size(); ++i)
                momentumGhostsRefiners_.addStaticRefiner(
                    info.ghostMomentum[i], nullptr, info.ghostMomentum[i],
                    nonOverwriteInteriorTFfillPattern, momentumPatchStrats[i]);
            registerGhostRefinePatchStrategies_(totalEnergyPatchStrats, info.ghostTotalEnergy);
            for (std::size_t i = 0; i < info.ghostTotalEnergy.size(); ++i)
                totalEnergyGhostsRefiners_.addStaticRefiner(
                    info.ghostTotalEnergy[i], nullptr, info.ghostTotalEnergy[i],
                    nonOverwriteFieldFillPattern, totalEnergyPatchStrats[i]);

            magFluxesXGhostRefiners_.addStaticRefiners(info.ghostMagneticFluxesX, nullptr,
                                                       info.ghostMagneticFluxesX,
                                                       nonOverwriteInteriorTFfillPattern);
            magFluxesYGhostRefiners_.addStaticRefiners(info.ghostMagneticFluxesY, nullptr,
                                                       info.ghostMagneticFluxesY,
                                                       nonOverwriteInteriorTFfillPattern);
            magFluxesZGhostRefiners_.addStaticRefiners(info.ghostMagneticFluxesZ, nullptr,
                                                       info.ghostMagneticFluxesZ,
                                                       nonOverwriteInteriorTFfillPattern);

            registerGhostRefinePatchStrategies_(magPatchStrats, info.ghostMagnetic);
            for (std::size_t i = 0; i < info.ghostMagnetic.size(); ++i)
            {
                auto const& key = info.ghostMagnetic[i];
                magGhostsRefiners_.addStaticRefiner(
                    key, nullptr, key, nonOverwriteInteriorTFfillPattern, magPatchStrats[i]);
            }
        }

        void makeRefineOperators_(RefinementConfig const& config)
        {
            auto fieldKernel = [&] {
                return std::make_shared<KernelFieldRefineOperator<GridLayoutT, GridT>>(
                    makeRefineKernel<GridLayoutT, GridT>(config.order));
            };
            auto vecKernel = [&] {
                return std::make_shared<KernelVecFieldRefineOperator<VectorFieldDataT>>(
                    makeRefineKernel<GridLayoutT, GridT>(config.order));
            };
            auto magKernel = [&] {
                return std::make_shared<KernelVecFieldRefineOperator<VectorFieldDataT>>(
                    makeMagneticRefineKernel<GridLayoutT, GridT>(config.order));
            };

            mhdFluxRefineOp_     = fieldKernel();
            mhdVecFluxRefineOp_  = vecKernel();
            mhdFieldRefineOp_    = fieldKernel();
            mhdVecFieldRefineOp_ = vecKernel();
            EfieldRefineOp_      = vecKernel();
            BfieldRegridOp_      = magKernel();
        }

        // Maybe we also need conservative ghost refiners for amr operations, actually quite
        // likely
        void registerGhostComms_(std::unique_ptr<MHDMessengerInfo> const& info)
        {
            // static refinement for E because in MHD it is a temporary, so keeping its
            // state updated after each regrid is not a priority. However if we do not correctly
            // refine on regrid, the post regrid state is not up to date (in our case it will be nan
            // since we nan-initialise) and thus is is better to rely on static refinement, which
            // uses the state after computation of CT.
            // The refiners for the electric field only serve for filling ghosts at physical
            // boundaries.
            registerGhostRefinePatchStrategies_(elecPatchStrats, info->ghostElectric);
            for (std::size_t i = 0; i < info->ghostElectric.size(); ++i)
                elecGhostsRefiners_.addStaticRefiner(info->ghostElectric[i], nullptr,
                                                     info->ghostElectric[i], noOverlapFillPattern,
                                                     elecPatchStrats[i]);

            // each ghost refiner gets its own patch strategy so that physical-boundary
            // ghosts are filled by the registered boundary conditions during schedule fills
            registerGhostRefinePatchStrategies_(rhoPatchStrats, info->ghostDensity);
            registerGhostRefinePatchStrategies_(momentumPatchStrats, info->ghostMomentum);
            registerGhostRefinePatchStrategies_(totalEnergyPatchStrats, info->ghostTotalEnergy);

            if constexpr (linearInTime)
            {
                auto const& old = info->oldState;

                for (std::size_t i = 0; i < info->ghostDensity.size(); ++i)
                    rhoGhostsRefiners_.addTimeRefiner(
                        info->ghostDensity[i], info->modelDensity, old.rho(), mhdFieldRefineOp_,
                        fieldTimeOp_, info->ghostDensity[i], nonOverwriteFieldFillPattern,
                        rhoPatchStrats[i]);
                for (std::size_t i = 0; i < info->ghostMomentum.size(); ++i)
                    momentumGhostsRefiners_.addTimeRefiner(
                        info->ghostMomentum[i], info->modelMomentum, old.rhoV(),
                        mhdVecFieldRefineOp_, vecFieldTimeOp_, info->ghostMomentum[i],
                        nonOverwriteInteriorTFfillPattern, momentumPatchStrats[i]);
                for (std::size_t i = 0; i < info->ghostTotalEnergy.size(); ++i)
                    totalEnergyGhostsRefiners_.addTimeRefiner(
                        info->ghostTotalEnergy[i], info->modelTotalEnergy, old.Etot(),
                        mhdFieldRefineOp_, fieldTimeOp_, info->ghostTotalEnergy[i],
                        nonOverwriteFieldFillPattern, totalEnergyPatchStrats[i]);
            }
            else
            {
                auto const& assembled = temporalTransfer_.assembledStateNames();

                for (std::size_t i = 0; i < info->ghostDensity.size(); ++i)
                    rhoGhostsRefiners_.addStaticRefiner(
                        info->ghostDensity[i], assembled.rho(), mhdFieldRefineOp_,
                        info->ghostDensity[i], nonOverwriteFieldFillPattern, rhoPatchStrats[i]);
                for (std::size_t i = 0; i < info->ghostMomentum.size(); ++i)
                    momentumGhostsRefiners_.addStaticRefiner(
                        info->ghostMomentum[i], assembled.rhoV(), mhdVecFieldRefineOp_,
                        info->ghostMomentum[i], nonOverwriteInteriorTFfillPattern,
                        momentumPatchStrats[i]);
                for (std::size_t i = 0; i < info->ghostTotalEnergy.size(); ++i)
                    totalEnergyGhostsRefiners_.addStaticRefiner(
                        info->ghostTotalEnergy[i], assembled.Etot(), mhdFieldRefineOp_,
                        info->ghostTotalEnergy[i], nonOverwriteFieldFillPattern,
                        totalEnergyPatchStrats[i]);
            }

            magFluxesXGhostRefiners_.addStaticRefiners(
                info->ghostMagneticFluxesX, mhdVecFluxRefineOp_, info->ghostMagneticFluxesX,
                nonOverwriteInteriorTFfillPattern);

            magFluxesYGhostRefiners_.addStaticRefiners(
                info->ghostMagneticFluxesY, mhdVecFluxRefineOp_, info->ghostMagneticFluxesY,
                nonOverwriteInteriorTFfillPattern);

            magFluxesZGhostRefiners_.addStaticRefiners(
                info->ghostMagneticFluxesZ, mhdVecFluxRefineOp_, info->ghostMagneticFluxesZ,
                nonOverwriteInteriorTFfillPattern);

            // we need a separate patch strategy for each refiner so that each one can register
            // their required ids
            registerGhostRefinePatchStrategies_(magPatchStrats, info->ghostMagnetic);

            for (std::size_t i = 0; i < info->ghostMagnetic.size(); ++i)
            {
                if constexpr (linearInTime)
                    magGhostsRefiners_.addTimeRefiner(
                        info->ghostMagnetic[i], info->modelMagnetic, info->oldState.B(),
                        BfieldRegridOp_, vecFieldTimeOp_, info->ghostMagnetic[i],
                        nonOverwriteInteriorTFfillPattern, magPatchStrats[i]);
                else
                    magGhostsRefiners_.addStaticRefiner(
                        info->ghostMagnetic[i], temporalTransfer_.assembledStateNames().B(),
                        BfieldRegridOp_, info->ghostMagnetic[i], nonOverwriteInteriorTFfillPattern,
                        magPatchStrats[i]);
            }

            magMaxModelRefiners_.addStaticRefiner(
                info->modelMagnetic, info->modelMagnetic, nullptr, info->modelMagnetic,
                std::make_shared<
                    TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());
        }




        /**
         * @brief Register a list of refine patch strategy pointers corresponding to a list of
         * keys.
         *
         * @tparam RefinePatchStrategyT type inheriting from SAMRAI's `RefinePatchStrategy`
         * @param patchStrategies the list of refine patch strategy pointers.
         * @param keys the list of keys.
         */
        template<typename RefinePatchStrategyT>
        void registerGhostRefinePatchStrategies_(
            std::vector<std::shared_ptr<RefinePatchStrategyT>>& patchStrategies,
            std::vector<std::string> const& keys)
        {
            patchStrategies.clear(); // registerQuantities must be idempotent
            patchStrategies.reserve(keys.size());
            for (auto const& key : keys)
            {
                auto&& [id]     = resourcesManager_->getIDsList(key);
                auto patchStrat = std::make_shared<RefinePatchStrategyT>(*boundaryManager_);
                patchStrat->registerIDs(id);
                patchStrategies.push_back(patchStrat);
            }
        }


        // should this use conservative quantities ? When should we do the initial conversion ?
        // Maybe mhd_init
        void registerInitComms_(std::unique_ptr<MHDMessengerInfo> const& info)
        {
            // Give the init refiners the model-state moment patch strategies (index 0 of each
            // ghost-strategy list: ghostX[0] == modelX == initX). The InitField schedule already
            // fills interior + coarse-fine from the coarser level via createSchedule(level,
            // nullptr, coarser, hierarchy, patchStrat) — the same call B uses in BalgoInit — and
            // with a patch strategy it now also fills the physical-boundary ghosts. So a freshly
            // created / regridded refined level touching a physical boundary carries valid moment
            // ghosts before the first flux. Default (non-overwrite) fill pattern is kept: overwrite
            // is a B/face-centered concern and corrupts the cell-centered moment interior fill.
            std::shared_ptr<SAMRAI::xfer::RefinePatchStrategy> rhoInitStrat
                = rhoPatchStrats.empty() ? nullptr : rhoPatchStrats[0];
            std::shared_ptr<SAMRAI::xfer::RefinePatchStrategy> momentumInitStrat
                = momentumPatchStrats.empty() ? nullptr : momentumPatchStrats[0];
            std::shared_ptr<SAMRAI::xfer::RefinePatchStrategy> totalEnergyInitStrat
                = totalEnergyPatchStrats.empty() ? nullptr : totalEnergyPatchStrats[0];

            densityInitRefiners_.addStaticRefiners(info->initDensity, mhdFieldRefineOp_,
                                                   info->initDensity, nullptr, rhoInitStrat);

            momentumInitRefiners_.addStaticRefiners(info->initMomentum, mhdVecFieldRefineOp_,
                                                    info->initMomentum, nullptr, momentumInitStrat);

            totalEnergyInitRefiners_.addStaticRefiners(info->initTotalEnergy, mhdFieldRefineOp_,
                                                       info->initTotalEnergy, nullptr,
                                                       totalEnergyInitStrat);
        }


        void magneticRegriding_(hierarchy_ptr const& hierarchy, level_ptr const& level,
                                level_ptr const& oldLevel, double const initDataTime)
        {
            auto magSchedule = BregridAlgo.createSchedule(
                level, oldLevel, level->getNextCoarserHierarchyLevelNumber(), hierarchy,
                &magneticRefinePatchStrategy_);

            magSchedule->fillData(initDataTime);
        }

        /** * @brief setNaNsFieldOnGhosts sets NaNs on the ghost nodes of the field
         *
         * NaNs are set on all ghost nodes, patch ghost or level ghost nodes
         * so that the refinement operators can know nodes at NaN have not been
         * touched by schedule copy.
         *
         * This is needed when the schedule copy is done before refinement
         * as a result of FieldVariable::fineBoundaryRepresentsVariable=false
         */
        void setNaNsOnFieldGhosts(FieldT& field)
        {
            auto const nGhosts = GridLayoutT::nDNbrGhosts(core::QtyCentering::primal)[0];
            field[core::NdArrayMask{0, nGhosts - 1}]
                = std::numeric_limits<typename FieldT::type>::quiet_NaN();
        }

        void setNaNsOnFieldGhosts(FieldT& field, level_t const& level)
        {
            for ([[maybe_unused]] auto& patch : resourcesManager_->enumerate(level, field))
                setNaNsOnFieldGhosts(field);
        }

        void setNaNsOnVecfieldGhosts(VecFieldT& vf, level_t const& level)
        {
            for ([[maybe_unused]] auto& patch : resourcesManager_->enumerate(level, vf))
                for (auto& component : vf)
                    setNaNsOnFieldGhosts(component);
        }


        TemporalTransfer temporalTransfer_;

        using rm_t = typename MHDModel::resources_manager_type;
        std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager_;
        std::shared_ptr<BoundaryManagerT> boundaryManager_;
        hierarchy_ptr hierarchy_; // the temporal transfer brackets a level with its coarser one
        int const firstLevel_;

        using InitRefinerPool             = RefinerPool<rm_t, RefinerType::InitField>;
        using GhostRefinerPool            = RefinerPool<rm_t, RefinerType::GhostField>;
        using PatchGhostRefinerPool       = RefinerPool<rm_t, RefinerType::PatchGhostField>;
        using InitDomPartRefinerPool      = RefinerPool<rm_t, RefinerType::InitInteriorPart>;
        using VecFieldGhostMaxRefinerPool = RefinerPool<rm_t, RefinerType::PatchVecFieldBorderMax>;


        SAMRAI::xfer::RefineAlgorithm BalgoPatchGhost; //
        SAMRAI::xfer::RefineAlgorithm BalgoInit;
        SAMRAI::xfer::RefineAlgorithm BregridAlgo;
        SAMRAI::xfer::RefineAlgorithm EalgoPatchGhost; //
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> magInitRefineSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> magGhostsRefineSchedules; //
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>
            magPatchGhostsRefineSchedules; //
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> elecPatchGhostsRefineSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>
            magSharedNodeRefineSchedules; //

        SAMRAI::xfer::CoarsenAlgorithm ErefluxAlgo{SAMRAI::tbox::Dimension{dimension}};
        SAMRAI::xfer::CoarsenAlgorithm HydroXrefluxAlgo{SAMRAI::tbox::Dimension{dimension}};
        SAMRAI::xfer::CoarsenAlgorithm HydroYrefluxAlgo{SAMRAI::tbox::Dimension{dimension}};
        SAMRAI::xfer::CoarsenAlgorithm HydroZrefluxAlgo{SAMRAI::tbox::Dimension{dimension}};

        SAMRAI::xfer::RefineAlgorithm EpatchGhostRefluxedAlgo;
        SAMRAI::xfer::RefineAlgorithm HydroXpatchGhostRefluxedAlgo;
        SAMRAI::xfer::RefineAlgorithm HydroYpatchGhostRefluxedAlgo;
        SAMRAI::xfer::RefineAlgorithm HydroZpatchGhostRefluxedAlgo;

        std::map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> ErefluxSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> HydroXrefluxSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> HydroYrefluxSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> HydroZrefluxSchedules;

        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> EpatchGhostRefluxedSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>
            HydroXpatchGhostRefluxedSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>
            HydroYpatchGhostRefluxedSchedules;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>
            HydroZpatchGhostRefluxedSchedules;

        PatchGhostRefinerPool elecGhostsRefiners_{resourcesManager_};
        GhostRefinerPool rhoGhostsRefiners_{resourcesManager_};
        // GhostRefinerPool velGhostsRefiners_{resourcesManager_};
        // GhostRefinerPool pressureGhostsRefiners_{resourcesManager_};
        GhostRefinerPool momentumGhostsRefiners_{resourcesManager_};
        GhostRefinerPool totalEnergyGhostsRefiners_{resourcesManager_};
        GhostRefinerPool magFluxesXGhostRefiners_{resourcesManager_};
        GhostRefinerPool magFluxesYGhostRefiners_{resourcesManager_};
        GhostRefinerPool magFluxesZGhostRefiners_{resourcesManager_};

        GhostRefinerPool magGhostsRefiners_{resourcesManager_};
        // root level only: the initial B equalize in fillRootGhosts
        VecFieldGhostMaxRefinerPool magMaxModelRefiners_{resourcesManager_};

        InitRefinerPool densityInitRefiners_{resourcesManager_};
        InitRefinerPool momentumInitRefiners_{resourcesManager_};
        InitRefinerPool totalEnergyInitRefiners_{resourcesManager_};

        // SynchronizerPool<rm_t> densitySynchronizers_{resourcesManager_};
        // SynchronizerPool<rm_t> momentumSynchronizers_{resourcesManager_};
        // SynchronizerPool<rm_t> magnetoSynchronizers_{resourcesManager_};
        // SynchronizerPool<rm_t> totalEnergySynchronizers_{resourcesManager_};

        using RefOp_ptr     = std::shared_ptr<SAMRAI::hier::RefineOperator>;
        using CoarsenOp_ptr = std::shared_ptr<SAMRAI::hier::CoarsenOperator>;
        using TimeOp_ptr    = std::shared_ptr<SAMRAI::hier::TimeInterpolateOperator>;

        using FieldTimeInterp = FieldLinearTimeInterpolate<GridLayoutT, GridT>;

        using VecFieldTimeInterp
            = VecFieldLinearTimeInterpolate<GridLayoutT, GridT, core::MHDQuantity>;

        template<typename Policy>
        using FieldCoarseningOp = FieldCoarsenOperator<GridLayoutT, GridT, Policy>;

        template<typename Policy>
        using VecFieldCoarsenOp
            = VecFieldCoarsenOperator<GridLayoutT, GridT, Policy, core::MHDQuantity>;

        using MHDFluxCoarsenOp       = FieldCoarseningOp<MHDFluxCoarsener<dimension>>;
        using MHDVecFluxCoarsenOp    = VecFieldCoarsenOp<MHDFluxCoarsener<dimension>>;
        using ElectricFieldCoarsenOp = VecFieldCoarsenOp<ElectricFieldCoarsener<dimension>>;

        SynchronizerPool<rm_t> electroSynchronizers_{resourcesManager_};

        // built in the ctor body (makeRefineOperators_) from the composite runtime kernels.
        RefOp_ptr mhdFluxRefineOp_;
        RefOp_ptr mhdVecFluxRefineOp_;
        RefOp_ptr mhdFieldRefineOp_;
        RefOp_ptr mhdVecFieldRefineOp_;
        RefOp_ptr EfieldRefineOp_;
        RefOp_ptr BfieldRegridOp_;

        TimeOp_ptr fieldTimeOp_{std::make_shared<FieldTimeInterp>()};
        TimeOp_ptr vecFieldTimeOp_{std::make_shared<VecFieldTimeInterp>()};

        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension /*, rank=1*/>;
        using FieldFillPattern_t       = FieldFillPattern<dimension>;

        std::shared_ptr<FieldFillPattern_t> nonOverwriteFieldFillPattern
            = std::make_shared<FieldFillPattern<dimension>>(); // stateless (mostly)

        std::shared_ptr<TensorFieldFillPattern_t> nonOverwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>();

        std::shared_ptr<NoOverlapFillPattern<dimension>> noOverlapFillPattern
            = std::make_shared<NoOverlapFillPattern<dimension>>();

        std::shared_ptr<TensorFieldFillPattern_t> overwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>(
                /*overwrite_interior=*/true);

        CoarsenOp_ptr mhdFluxCoarseningOp_{std::make_shared<MHDFluxCoarsenOp>()};
        CoarsenOp_ptr mhdVecFluxCoarseningOp_{std::make_shared<MHDVecFluxCoarsenOp>()};
        CoarsenOp_ptr electricFieldCoarseningOp_{std::make_shared<ElectricFieldCoarsenOp>()};

        using FieldRefinePatchStrategyT = FieldRefinePatchStrategy<FieldDataT, BoundaryManagerT>;
        using VectorFieldRefinePatchStrategyT
            = FieldRefinePatchStrategy<VectorFieldDataT, BoundaryManagerT>;
        using MagneticRefinePatchStrategyT
            = MagneticRefinePatchStrategy<VectorFieldDataT, BoundaryManagerT>;
        using FieldRefinePatchStrategyList
            = std::vector<std::shared_ptr<FieldRefinePatchStrategyT>>;
        using VectorFieldRefinePatchStrategyList
            = std::vector<std::shared_ptr<VectorFieldRefinePatchStrategyT>>;
        using MagneticRefinePatchStrategyList
            = std::vector<std::shared_ptr<MagneticRefinePatchStrategyT>>;

        MagneticRefinePatchStrategyT magneticRefinePatchStrategy_{*boundaryManager_};

        FieldRefinePatchStrategyList rhoPatchStrats;
        FieldRefinePatchStrategyList totalEnergyPatchStrats;
        VectorFieldRefinePatchStrategyList momentumPatchStrats;
        VectorFieldRefinePatchStrategyList elecPatchStrats;
        MagneticRefinePatchStrategyList magPatchStrats;
    };

} // namespace amr
} // namespace PHARE
#endif
