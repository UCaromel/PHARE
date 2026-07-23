#ifndef PHARE_MHD_MESSENGER_HPP
#define PHARE_MHD_MESSENGER_HPP

#include "core/def/phare_mpi.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include "amr/data/field/refine/field_refiner.hpp"
#include "amr/data/field/coarsening/electric_field_coarsener.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/mhd_flux_coarsener.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/composite_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_composite_refiner.hpp"
#include "amr/messengers/refinement_config.hpp"
#include "amr/data/field/refine/electric_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_field_regrider.hpp"
#include "amr/data/field/refine/mhd_field_refiner.hpp"
#include "amr/data/field/refine/mhd_flux_refiner.hpp"
#include "amr/data/field/time_interpolate/field_linear_time_interpolate.hpp"
#include "amr/messengers/refiner.hpp"
#include "amr/messengers/refiner_pool.hpp"
#include "amr/messengers/synchronizer_pool.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/messenger_info.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/data/field/refine/magnetic_refine_patch_strategy.hpp"
#include "amr/data/field/refine/adpt_magnetic_refine_patch_strategy.hpp"
#include "amr/data/field/refine/magnetic_patch_strategy_base.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"

#include "core/data/vecfield/vecfield.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/def/phare_mpi.hpp"
#include "core/models/mhd_state_increment.hpp"
#include "core/numerics/mc2011/mc2011_reconstruction.hpp"

#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/RefineOperator.h"
#include "SAMRAI/hier/CoarsenOperator.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace PHARE
{
namespace amr
{
    // UseMC2011Temporal: compile-time gate, resolved in phare_solver.hpp from the
    // selected time integrator (true only for SSPRK4_5). When true, registerGhostComms_
    // swaps the rho/rhoV/Etot/B C-F ghost refiners for a static refiner sourced from
    // mc2011Assembled_ (Tier 2/3 output) instead of the 2nd-order linear-in-time path.
    template<typename MHDModel, bool UseMC2011Temporal = false>
    class MHDMessenger : public IMessenger<typename MHDModel::Interface>
    {
        using amr_types   = PHARE::amr::SAMRAI_Types;
        using level_t     = amr_types::level_t;
        using patch_t     = amr_types::patch_t;
        using hierarchy_t = amr_types::hierarchy_t;

        using IPhysicalModel    = MHDModel::Interface;
        using FieldT            = MHDModel::field_type;
        using VecFieldT         = MHDModel::vecfield_type;
        using MHDStateT         = MHDModel::state_type;
        using GridLayoutT       = MHDModel::gridlayout_type;
        using GridT             = MHDModel::grid_type;
        using ResourcesManagerT = MHDModel::resources_manager_type;
        using VectorFieldDataT  = TensorFieldData<1, GridLayoutT, GridT, core::MHDQuantity>;
        using KIncrementT       = core::MHDStateIncrement<VecFieldT>;

        static constexpr auto dimension = MHDModel::dimension;

    public:
        static constexpr std::size_t rootLevelNumber = 0;
        static inline std::string const stratName    = "MHDModel-MHDModel";

        MHDMessenger(std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager,
                     int const firstLevel, RefinementConfig const& refinementConfig = {})
            : resourcesManager_{std::move(resourcesManager)}
            , firstLevel_{firstLevel}
            , config_{refinementConfig}
        {
            makeRefineOperators_(refinementConfig);
            magneticRefinePatchStrategy_ = makeMagneticPatchStrategy_();

            // moment ghosts are primitive quantities
            resourcesManager_->registerResources(rhoOld_);
            resourcesManager_->registerResources(Vold_);
            resourcesManager_->registerResources(Pold_);

            resourcesManager_->registerResources(rhoVold_);
            resourcesManager_->registerResources(EtotOld_);

            resourcesManager_->registerResources(Bold_);

            // MC2011 temporal reconstruction: stateN_/unp1_ mirror the
            // SSPRK4_5Integrator's own same-named fields by construction (MHDState and
            // MHDStateIncrement share the `name + "_rho"` naming scheme, so these lean
            // conserved-quad views resolve to the integrator's full stage states) --
            // registerResources is name-keyed and idempotent (resources_manager.hpp),
            // so whichever of {solver, messenger} registers first wins and both end up
            // viewing the same underlying PatchData. For any other integrator
            // (TVDRK2/TVDRK3/Euler), this messenger is the sole registrant of whatever
            // the integrator doesn't name itself: harmless unused allocation, gated
            // out of the ghost-fill path (registerGhostComms_ never references these
            // for those configs).
            resourcesManager_->registerResources(state1_);
            resourcesManager_->registerResources(state2_);
            resourcesManager_->registerResources(state3_);
            resourcesManager_->registerResources(state4_);
            resourcesManager_->registerResources(unp1_);
            resourcesManager_->registerResources(mc2011Assembled_);

            // also magnetic fluxes ? or should we use static refiners instead ?
        }

        virtual ~MHDMessenger() = default;

        void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override
        {
            resourcesManager_->allocate(rhoOld_, patch, allocateTime);
            resourcesManager_->allocate(Vold_, patch, allocateTime);
            resourcesManager_->allocate(Pold_, patch, allocateTime);

            resourcesManager_->allocate(rhoVold_, patch, allocateTime);
            resourcesManager_->allocate(EtotOld_, patch, allocateTime);

            resourcesManager_->allocate(Bold_, patch, allocateTime);

            // allocate_ is guarded by patch.checkAllocated(id) (name-keyed), so this
            // is a harmless no-op if the solver already allocated these for this
            // patch (order-independent -- see registerResources comment above).
            resourcesManager_->allocate(state1_, patch, allocateTime);
            resourcesManager_->allocate(state2_, patch, allocateTime);
            resourcesManager_->allocate(state3_, patch, allocateTime);
            resourcesManager_->allocate(state4_, patch, allocateTime);
            resourcesManager_->allocate(unp1_, patch, allocateTime);
            resourcesManager_->allocate(mc2011Assembled_, patch, allocateTime);
        }


        void
        registerQuantities(std::unique_ptr<IMessengerInfo> fromCoarserInfo,
                           [[maybe_unused]] std::unique_ptr<IMessengerInfo> fromFinerInfo) override
        {
            std::unique_ptr<MHDMessengerInfo> mhdInfo{
                dynamic_cast<MHDMessengerInfo*>(fromFinerInfo.release())};

            auto b_id = resourcesManager_->getID(mhdInfo->modelMagnetic);

            if (!b_id)
            {
                throw std::runtime_error(
                    "MHDMessengerStrategy: missing magnetic field variable IDs");
            }

            magneticRefinePatchStrategy_->registerIDs(*b_id);

            BalgoPatchGhost.registerRefine(*b_id, *b_id, *b_id, BfieldRefineOp_,
                                           nonOverwriteInteriorTFfillPattern);

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

            // MC2011 sanity check: catches any future rename drift between this
            // messenger's hardcoded "state1".."unp1" base names (needed at
            // construction, before mhdInfo exists) and the solver's own same-named
            // fields. Skipped (mhdInfo left default/empty) for every integrator
            // except SSPRK4_5.
            if (!mhdInfo->stageState1.rho.empty()
                && mhdInfo->stageState1.rho != state1_.rho.name())
                throw std::runtime_error("MHDMessenger: MC2011 state1 field name mismatch");
            if (!mhdInfo->stageState2.rho.empty()
                && mhdInfo->stageState2.rho != state2_.rho.name())
                throw std::runtime_error("MHDMessenger: MC2011 state2 field name mismatch");
            if (!mhdInfo->stageState3.rho.empty()
                && mhdInfo->stageState3.rho != state3_.rho.name())
                throw std::runtime_error("MHDMessenger: MC2011 state3 field name mismatch");
            if (!mhdInfo->stageState4.rho.empty()
                && mhdInfo->stageState4.rho != state4_.rho.name())
                throw std::runtime_error("MHDMessenger: MC2011 state4 field name mismatch");
            if (!mhdInfo->unp1.rho.empty() && mhdInfo->unp1.rho != unp1_.rho.name())
                throw std::runtime_error("MHDMessenger: MC2011 unp1 field name mismatch");

            registerGhostComms_(mhdInfo);
            registerInitComms_(mhdInfo);
        }



        void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                           int const levelNumber) override
        {
            auto const level = hierarchy->getPatchLevel(levelNumber);

            // magPatchGhostsRefineSchedules[levelNumber]
            //     = BalgoPatchGhost.createSchedule(level, &magneticRefinePatchStrategy_);

            // elecPatchGhostsRefineSchedules[levelNumber] = EalgoPatchGhost.createSchedule(level);

            EpatchGhostRefluxedSchedules[levelNumber]
                = EpatchGhostRefluxedAlgo.createSchedule(level);
            HydroXpatchGhostRefluxedSchedules[levelNumber]
                = HydroXpatchGhostRefluxedAlgo.createSchedule(level);
            HydroYpatchGhostRefluxedSchedules[levelNumber]
                = HydroYpatchGhostRefluxedAlgo.createSchedule(level);
            HydroZpatchGhostRefluxedSchedules[levelNumber]
                = HydroZpatchGhostRefluxedAlgo.createSchedule(level);

            rhoGhostsRefiners_.registerLevel(hierarchy, level);
            momentumGhostsRefiners_.registerLevel(hierarchy, level);
            totalEnergyGhostsRefiners_.registerLevel(hierarchy, level);
            rhoMaxRefiners_.registerLevel(hierarchy, level);
            momentumMaxRefiners_.registerLevel(hierarchy, level);
            totalEnergyMaxRefiners_.registerLevel(hierarchy, level);
            rhoModelMaxRefiners_.registerLevel(hierarchy, level);
            momentumModelMaxRefiners_.registerLevel(hierarchy, level);
            totalEnergyModelMaxRefiners_.registerLevel(hierarchy, level);

            // magFluxesXGhostRefiners_.registerLevel(hierarchy, level);
            // magFluxesYGhostRefiners_.registerLevel(hierarchy, level);
            // magFluxesZGhostRefiners_.registerLevel(hierarchy, level);

            magGhostsRefiners_.registerLevel(hierarchy, level);
            magMaxRefiners_.registerLevel(hierarchy, level);
            magMaxModelRefiners_.registerLevel(hierarchy, level);

            if (levelNumber != rootLevelNumber)
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
                    level, nullptr, levelNumber - 1, hierarchy,
                    magneticRefinePatchStrategy_.get());

                densityInitRefiners_.registerLevel(hierarchy, level);
                momentumInitRefiners_.registerLevel(hierarchy, level);
                totalEnergyInitRefiners_.registerLevel(hierarchy, level);
            }
        }


        void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                    int const levelNumber,
                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                    IPhysicalModel& model, double const initDataTime) override
        {
            auto& mhdModel = static_cast<MHDModel&>(model);
            auto level     = hierarchy->getPatchLevel(levelNumber);

            bool isRegriddingL0 = levelNumber == 0 and oldLevel;

            magneticRegriding_(hierarchy, level, oldLevel, initDataTime);
            magMaxModelRefiners_.fill(mhdModel.state.B, level->getLevelNumber(), initDataTime);

            densityInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            momentumInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            totalEnergyInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            rhoModelMaxRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            momentumModelMaxRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            totalEnergyModelMaxRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);

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

        void initLevel(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       double const initDataTime) override
        {
            auto levelNumber = level.getLevelNumber();

            auto& mhdModel = static_cast<MHDModel&>(model);

            magInitRefineSchedules[levelNumber]->fillData(initDataTime);
            densityInitRefiners_.fill(levelNumber, initDataTime);
            momentumInitRefiners_.fill(levelNumber, initDataTime);
            totalEnergyInitRefiners_.fill(levelNumber, initDataTime);
        }

        void firstStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                       double const currentTime, double const prevCoarserTIme,
                       double const newCoarserTime) final
        {
            // MC2011 temporal reconstruction: track the coarse step's time bracket and
            // a handle on the coarser level, so assembleMC2011_ (Step 4 wires the actual
            // call sites) can read that level's persisted stage states at any chi in
            // [0,1] for the duration of this level's subcycle. Mirrors
            // HybridHybridMessengerStrategy::firstStep()'s beforePushCoarseTime_/
            // afterPushCoarseTime_ pattern, plus a coarser-level handle (needed here
            // because, unlike the hybrid case, we must WRITE mc2011Assembled_ onto the
            // coarse level's own patches before the static refiner can read it).
            auto const levelNumber = level.getLevelNumber();
            if (levelNumber == rootLevelNumber)
                return;

            beforePushCoarseTime_[levelNumber] = prevCoarserTIme;
            afterPushCoarseTime_[levelNumber]  = newCoarserTime;
            coarserLevels_[levelNumber]        = hierarchy->getPatchLevel(levelNumber - 1);
        }


        // MC2011: chi = fraction of the coarse step's [prevCoarserTime, newCoarserTime]
        // bracket (recorded by firstStep()) reached by this level's own step starting at
        // stepStartTime. Fixed for a whole RK step (all 5 stage/final ghost-fills of one
        // SSPRK4_5Integrator::operator() call share one chi, per full-derivation.md S5.4).
        // Root level has no coarser bracket; callers must not use the result there (see
        // assembleMC2011_'s own rootLevelNumber guard) -- 0.0 is a harmless placeholder.
        double mc2011Chi(level_t const& level, double const stepStartTime) const
        {
            auto const levelNumber = level.getLevelNumber();
            if (levelNumber == rootLevelNumber)
                return 0.0;

            double const start = beforePushCoarseTime_.at(levelNumber);
            double const end   = afterPushCoarseTime_.at(levelNumber);
            return (stepStartTime - start) / (end - start);
        }


        void lastStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level) final {}


        void prepareStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                         double currentTime) final
        {
            auto& mhdModel = static_cast<MHDModel&>(model);
            for (auto& patch : level)
            {
                auto dataOnPatch = resourcesManager_->setOnPatch(
                    *patch, mhdModel.state.rho, mhdModel.state.V, mhdModel.state.P,
                    mhdModel.state.rhoV, mhdModel.state.Etot, mhdModel.state.B, rhoOld_, Vold_,
                    Pold_, rhoVold_, EtotOld_, Bold_);

                resourcesManager_->setTime(rhoOld_, *patch, currentTime);
                resourcesManager_->setTime(Vold_, *patch, currentTime);
                resourcesManager_->setTime(Pold_, *patch, currentTime);
                resourcesManager_->setTime(rhoVold_, *patch, currentTime);
                resourcesManager_->setTime(EtotOld_, *patch, currentTime);
                resourcesManager_->setTime(Bold_, *patch, currentTime);

                rhoOld_.copyData(mhdModel.state.rho);
                Vold_.copyData(mhdModel.state.V);
                Pold_.copyData(mhdModel.state.P);
                rhoVold_.copyData(mhdModel.state.rhoV);
                EtotOld_.copyData(mhdModel.state.Etot);
                Bold_.copyData(mhdModel.state.B);
            }
        }

        void fillRootGhosts(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                            double const initDataTime) final
        {
        }

        void synchronize(SAMRAI::hier::PatchLevel& level) final {}

        void reflux(int const coarserLevelNumber, int const fineLevelNumber,
                    double const syncTime) override
        {
            ErefluxSchedules[fineLevelNumber]->coarsenData();
            HydroXrefluxSchedules[fineLevelNumber]->coarsenData();
            HydroYrefluxSchedules[fineLevelNumber]->coarsenData();
            HydroZrefluxSchedules[fineLevelNumber]->coarsenData();

            EpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
            HydroXpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
            HydroYpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
            HydroZpatchGhostRefluxedSchedules[coarserLevelNumber]->fillData(syncTime);
        }

        void postSynchronize(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                             double const time) override
        {
            // The ghosts for B are obtained in the solver's reflux_euler. For B, this is because
            // refluxing is done through faraday which is computed on the ghost box for the other
            // quantities, the ghosts are filled in the end of the euler step anyways.
        }

        void fillMomentsGhosts(MHDStateT& state, level_t const& level, double const fillTime)
        {
            setNaNsOnFieldGhosts(state.rho, level);
            setNaNsOnVecfieldGhosts(state.rhoV, level);
            setNaNsOnFieldGhosts(state.Etot, level);

            if constexpr (UseMC2011Temporal)
            {
                // k-less fill (post-reflux euler path): the static refiners below are
                // sourced from mc2011Assembled_, which must be (re-)assembled at this
                // fillTime or the fill reads stale/NaN data -- see assembleAtChi_.
                auto const levelNumber = level.getLevelNumber();
                if (levelNumber != rootLevelNumber)
                    assembleAtChi_(levelNumber, fillTime);

                selfAssembleMomentsMC2011_(state, level);
            }

            rhoGhostsRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumGhostsRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyGhostsRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);

            rhoMaxRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumMaxRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyMaxRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);
        }

        // MC2011 overload: stageIndex in [0..4] (0..3 = state1_..state4_/Y2..Y5, 4 = the
        // final blend), chi from MHDMessenger::mc2011Chi (fixed for the whole RK step),
        // dtFine = this level's own RK step size. Used only by SSPRK4_5Integrator's 5
        // ghost-fill call sites; additive overload, the 3-arg signature above is untouched
        // so TVDRK2/TVDRK3/Euler are unaffected. When UseMC2011Temporal is false, this
        // still compiles (SSPRK4_5Integrator is only ever paired with the true
        // instantiation, per phare_solver.hpp, so the false branch is simply unreachable
        // in practice) but assembleMC2011_ is skipped via if constexpr.
        void fillMomentsGhosts(MHDStateT& state, level_t const& level, double const fillTime,
                               std::size_t const stageIndex, double const chi,
                               double const dtFine)
        {
            setNaNsOnFieldGhosts(state.rho, level);
            setNaNsOnVecfieldGhosts(state.rhoV, level);
            setNaNsOnFieldGhosts(state.Etot, level);

            if constexpr (UseMC2011Temporal)
            {
                auto const levelNumber = level.getLevelNumber();

                // Persist this level's own sweep dt (dtFine is the integrator's
                // dt = newTime - currentTime, identical across the 5 fills of one RK
                // step). When a finer level later assembles against THIS level, its
                // back-solve divides by this exact value -- recomputing t1 - t0 there
                // is not guaranteed bit-identical to the sweep's dt, and the S5.4
                // invariant (assembly dtC == sweep dt, bit-for-bit) would break.
                sweepDt_[levelNumber] = dtFine;

                if (levelNumber != rootLevelNumber)
                    assembleMC2011_(levelNumber, stageIndex, chi, dtFine);

                // The static refiners below are sourced from mc2011Assembled_ for BOTH
                // the coarse-fine portion (assembled above, on the coarser level's own
                // patches) AND the same-level/periodic portion of the same schedule
                // (SAMRAI fills same-level ghost overlaps from the source variable's
                // OWN level too). assembleMC2011_ never touches THIS level's patches
                // (root has no coarser bracket at all; a non-root level's patches are
                // only ever written when it acts as someone else's coarser neighbor) --
                // without this self-copy, same-level exchange reads mc2011Assembled_'s
                // NaN-initialized sentinel on every level. Interior-only copy: ghosts
                // were just NaN-sentineled above and aren't valid source data anyway.
                selfAssembleMomentsMC2011_(state, level);
            }

            rhoGhostsRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumGhostsRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyGhostsRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);

            rhoMaxRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumMaxRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyMaxRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);
        }

        // no point-value ghost fills: point-value quantities are local derived quantities,
        // computed on shrunk ghost boxes from average ghosts (see PointValueHandler).

        // void fillMagneticFluxesXGhosts(VecFieldT& Fx_B, level_t const& level, double const
        // fillTime)
        // {
        //     setNaNsOnVecfieldGhosts(Fx_B, level);
        //     magFluxesXGhostRefiners_.fill(Fx_B, level.getLevelNumber(), fillTime);
        // }
        //
        // void fillMagneticFluxesYGhosts(VecFieldT& Fy_B, level_t const& level, double const
        // fillTime)
        // {
        //     setNaNsOnVecfieldGhosts(Fy_B, level);
        //     magFluxesYGhostRefiners_.fill(Fy_B, level.getLevelNumber(), fillTime);
        // }
        //
        // void fillMagneticFluxesZGhosts(VecFieldT& Fz_B, level_t const& level, double const
        // fillTime)
        // {
        //     setNaNsOnVecfieldGhosts(Fz_B, level);
        //     magFluxesZGhostRefiners_.fill(Fz_B, level.getLevelNumber(), fillTime);
        // }

        void fillMagneticGhosts(VecFieldT& B, level_t const& level, double const fillTime)
        {
            PHARE_LOG_SCOPE(3, "MHDMessenger::fillMagneticGhosts");

            setNaNsOnVecfieldGhosts(B, level);

            if constexpr (UseMC2011Temporal)
            {
                // See the k-less fillMomentsGhosts just above -- same re-assembly
                // requirement for the mc2011Assembled_-sourced refiners.
                auto const levelNumber = level.getLevelNumber();
                if (levelNumber != rootLevelNumber)
                    assembleAtChi_(levelNumber, fillTime);

                selfAssembleMagneticMC2011_(B, level);
            }

            magGhostsRefiners_.fill(B, level.getLevelNumber(), fillTime);
            magMaxRefiners_.fill(B, level.getLevelNumber(), fillTime);
        }

        // MC2011 overload -- see fillMomentsGhosts's MC2011 overload just above for the
        // parameter semantics; the two are called back-to-back at the same stage/chi from
        // EulerUsingComputedFlux, each independently (re-)populating mc2011Assembled_ (no
        // cross-call memoization in this first cut, matching Step 3's scope note).
        void fillMagneticGhosts(VecFieldT& B, level_t const& level, double const fillTime,
                                std::size_t const stageIndex, double const chi,
                                double const dtFine)
        {
            PHARE_LOG_SCOPE(3, "MHDMessenger::fillMagneticGhosts");

            setNaNsOnVecfieldGhosts(B, level);

            if constexpr (UseMC2011Temporal)
            {
                auto const levelNumber = level.getLevelNumber();

                // See fillMomentsGhosts's MC2011 overload: persist the sweep dt.
                sweepDt_[levelNumber] = dtFine;

                if (levelNumber != rootLevelNumber)
                    assembleMC2011_(levelNumber, stageIndex, chi, dtFine);

                // See fillMomentsGhosts's MC2011 overload for the rationale: same-level
                // self-copy so the shared mc2011Assembled_ source is valid for the
                // same-level/periodic portion of magGhostsRefiners_'s schedule too.
                selfAssembleMagneticMC2011_(B, level);
            }

            magGhostsRefiners_.fill(B, level.getLevelNumber(), fillTime);
            magMaxRefiners_.fill(B, level.getLevelNumber(), fillTime);
        }

        std::string name() override { return stratName; }



    private:
        // Select the field-refinement operators once at construction. order==0 keeps the legacy
        // per-quantity policies (byte-identical to master); order==2/4 swaps in the composite
        // runtime kernels. B uses the shared-face magnetic kernel (interior stays Tóth-Roe).
        void makeRefineOperators_(RefinementConfig const& config)
        {
            if (config.order)
            {
                auto fieldKernel = [&] {
                    return std::make_shared<KernelFieldRefineOperator<GridLayoutT, GridT>>(
                        makeRefineKernel<GridLayoutT, GridT>(config.order, config.limiter));
                };
                auto vecKernel = [&] {
                    return std::make_shared<KernelVecFieldRefineOperator<VectorFieldDataT>>(
                        makeRefineKernel<GridLayoutT, GridT>(config.order, config.limiter));
                };
                auto magKernel = [&] {
                    return std::make_shared<KernelVecFieldRefineOperator<VectorFieldDataT>>(
                        makeMagneticRefineKernel<GridLayoutT, GridT>(config.order, config.limiter));
                };

                mhdFluxRefineOp_    = fieldKernel();
                mhdVecFluxRefineOp_ = vecKernel();
                mhdFieldRefineOp_   = fieldKernel();
                mhdVecFieldRefineOp_ = vecKernel();
                EfieldRefineOp_     = vecKernel();
                BfieldRefineOp_     = magKernel();
                BfieldRegridOp_     = magKernel();
            }
            else
            {
                mhdFluxRefineOp_    = std::make_shared<MHDFluxRefineOp>();
                mhdVecFluxRefineOp_ = std::make_shared<MHDVecFluxRefineOp>();
                mhdFieldRefineOp_   = std::make_shared<MHDFieldRefineOp>();
                mhdVecFieldRefineOp_ = std::make_shared<MHDVecFieldRefineOp>();
                EfieldRefineOp_     = std::make_shared<ElectricFieldRefineOp>();
                BfieldRefineOp_     = std::make_shared<MagneticFieldRefineOp>();
                BfieldRegridOp_     = std::make_shared<MagneticFieldRegridOp>();
            }
        }

        // B postprocess strategy paired with the operator selection above: order==0 keeps the
        // legacy Tóth-Roe interior fill; order==2/4 pairs the fill-all composite kernel with the
        // ADPT div-free touch-up (order-independent by construction).
        std::shared_ptr<MagneticPatchStrategyBase> makeMagneticPatchStrategy_() const
        {
            if (config_.order)
                return std::make_shared<
                    ADPTMagneticRefinePatchStrategy<ResourcesManagerT, VectorFieldDataT>>(
                    *resourcesManager_);
            return std::make_shared<
                MagneticRefinePatchStrategy<ResourcesManagerT, VectorFieldDataT>>(
                *resourcesManager_);
        }

        // Maybe we also need conservative ghost refiners for amr operations, actually quite
        // likely
        void registerGhostComms_(std::unique_ptr<MHDMessengerInfo> const& info)
        {
            // E no longer has a ghost-fill schedule: UpwindConstrainedTransport now computes
            // it self-sufficiently on physical box+1 (see upwind_constrained_transport.hpp),
            // replacing what used to be a static (non-temporal) refiner here.

            if constexpr (UseMC2011Temporal)
            {
                // MC2011: swap the 2nd-order linear-in-time C-F ghost fill for a static
                // refiner sourced from mc2011Assembled_ (Tier 2/3 output, freshly
                // populated by assembleMC2011_ just before each fill -- see
                // fillMomentsGhosts's MC2011 overload). One coarse-side scratch field
                // serves every stage/chi: the refiner just copies whatever is currently
                // in mc2011Assembled_ at fill time.
                std::vector<std::string> const rhoSrc(info->ghostDensity.size(),
                                                       mc2011Assembled_.rho.name());
                rhoGhostsRefiners_.addStaticRefiners(info->ghostDensity, rhoSrc,
                                                     mhdFieldRefineOp_, info->ghostDensity,
                                                     nonOverwriteFieldFillPattern);

                std::vector<std::string> const momentumSrc(info->ghostMomentum.size(),
                                                            mc2011Assembled_.rhoV.name());
                momentumGhostsRefiners_.addStaticRefiners(
                    info->ghostMomentum, momentumSrc, mhdVecFieldRefineOp_, info->ghostMomentum,
                    nonOverwriteInteriorTFfillPattern);

                std::vector<std::string> const totalEnergySrc(info->ghostTotalEnergy.size(),
                                                               mc2011Assembled_.Etot.name());
                totalEnergyGhostsRefiners_.addStaticRefiners(
                    info->ghostTotalEnergy, totalEnergySrc, mhdFieldRefineOp_,
                    info->ghostTotalEnergy, nonOverwriteFieldFillPattern);
            }
            else
            {
                rhoGhostsRefiners_.addTimeRefiners(info->ghostDensity, info->modelDensity,
                                                   rhoOld_.name(), mhdFieldRefineOp_,
                                                   fieldTimeOp_, nonOverwriteFieldFillPattern);

                momentumGhostsRefiners_.addTimeRefiners(
                    info->ghostMomentum, info->modelMomentum, rhoVold_.name(),
                    mhdVecFieldRefineOp_, vecFieldTimeOp_, nonOverwriteInteriorTFfillPattern);

                totalEnergyGhostsRefiners_.addTimeRefiners(
                    info->ghostTotalEnergy, info->modelTotalEnergy, EtotOld_.name(),
                    mhdFieldRefineOp_, fieldTimeOp_, nonOverwriteFieldFillPattern);
            }

            // always static, this is a max battle on time interpolated data already. single refiner
            // as all hydro quantities have same centering
            rhoMaxRefiners_.addStaticRefiners(
                info->ghostDensity, info->ghostDensity, nullptr, info->ghostDensity,
                std::make_shared<FieldGhostInterpOverlapFillPattern<GridLayoutT>>());

            momentumMaxRefiners_.addStaticRefiners(
                info->ghostMomentum, info->ghostMomentum, nullptr, info->ghostMomentum,
                std::make_shared<
                    TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());

            totalEnergyMaxRefiners_.addStaticRefiners(
                info->ghostTotalEnergy, info->ghostTotalEnergy, nullptr, info->ghostTotalEnergy,
                std::make_shared<FieldGhostInterpOverlapFillPattern<GridLayoutT>>());

            // model only version for regrid
            rhoModelMaxRefiners_.addStaticRefiner(
                info->modelDensity, info->modelDensity, nullptr, info->modelDensity,
                std::make_shared<FieldGhostInterpOverlapFillPattern<GridLayoutT>>());

            momentumModelMaxRefiners_.addStaticRefiner(
                info->modelMomentum, info->modelMomentum, nullptr, info->modelMomentum,
                std::make_shared<
                    TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());

            totalEnergyModelMaxRefiners_.addStaticRefiner(
                info->modelTotalEnergy, info->modelTotalEnergy, nullptr, info->modelTotalEnergy,
                std::make_shared<FieldGhostInterpOverlapFillPattern<GridLayoutT>>());

            // no point-value refiners: point-value quantities are local derived quantities,
            // computed on shrunk ghost boxes from the average ghosts filled above.

            // magFluxesXGhostRefiners_.addStaticRefiners(
            //     info->ghostMagneticFluxesX, mhdVecFluxRefineOp_, info->ghostMagneticFluxesX,
            //     nonOverwriteInteriorTFfillPattern);
            //
            // magFluxesYGhostRefiners_.addStaticRefiners(
            //     info->ghostMagneticFluxesY, mhdVecFluxRefineOp_, info->ghostMagneticFluxesY,
            //     nonOverwriteInteriorTFfillPattern);
            //
            // magFluxesZGhostRefiners_.addStaticRefiners(
            //     info->ghostMagneticFluxesZ, mhdVecFluxRefineOp_, info->ghostMagneticFluxesZ,
            //     nonOverwriteInteriorTFfillPattern);

            // we need a separate patch strategy for each refiner so that each one can register
            // their required ids
            magneticPatchStratPerGhostRefiner_ = [&]() {
                std::vector<std::shared_ptr<MagneticPatchStrategyBase>> result;

                result.reserve(info->ghostMagnetic.size());

                for (auto const& key : info->ghostMagnetic)
                {
                    auto&& [id] = resourcesManager_->getIDsList(key);

                    auto patch_strat = makeMagneticPatchStrategy_();

                    patch_strat->registerIDs(id);

                    result.push_back(patch_strat);
                }
                return result;
            }();

            for (size_t i = 0; i < info->ghostMagnetic.size(); ++i)
            {
                if constexpr (UseMC2011Temporal)
                {
                    // MC2011: static refiner sourced from mc2011Assembled_.B (see the
                    // rho/rhoV/Etot branch above for the shared rationale).
                    magGhostsRefiners_.addStaticRefiner(
                        info->ghostMagnetic[i], mc2011Assembled_.B.name(), BfieldRegridOp_,
                        info->ghostMagnetic[i], nonOverwriteInteriorTFfillPattern,
                        magneticPatchStratPerGhostRefiner_[i]);
                }
                else
                {
                    magGhostsRefiners_.addTimeRefiner(
                        info->ghostMagnetic[i], info->modelMagnetic, Bold_.name(),
                        BfieldRegridOp_, vecFieldTimeOp_, info->ghostMagnetic[i],
                        nonOverwriteInteriorTFfillPattern, magneticPatchStratPerGhostRefiner_[i]);
                }

                magMaxRefiners_.addStaticRefiner(
                    info->ghostMagnetic[i], info->ghostMagnetic[i], nullptr, info->ghostMagnetic[i],
                    std::make_shared<
                        TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());
            }

            magMaxModelRefiners_.addStaticRefiner(
                info->modelMagnetic, info->modelMagnetic, nullptr, info->modelMagnetic,
                std::make_shared<
                    TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());
        }




        // should this use conservative quantities ? When should we do the initial conversion ?
        // Maybe mhd_init
        void registerInitComms_(std::unique_ptr<MHDMessengerInfo> const& info)
        {
            densityInitRefiners_.addStaticRefiners(info->initDensity, mhdFieldRefineOp_,
                                                   info->initDensity);

            momentumInitRefiners_.addStaticRefiners(info->initMomentum, mhdVecFieldRefineOp_,
                                                    info->initMomentum);

            totalEnergyInitRefiners_.addStaticRefiners(info->initTotalEnergy, mhdFieldRefineOp_,
                                                       info->initTotalEnergy);
        }


        // Same-level counterpart to assembleMC2011_: copies state's own (interior, still
        // valid) rho/rhoV/Etot into mc2011Assembled_ on THIS level, so the same-level/
        // periodic part of rhoGhostsRefiners_/momentumGhostsRefiners_/
        // totalEnergyGhostsRefiners_'s schedules (which share the coarse-fine-only
        // mc2011Assembled_ source) has valid data to copy from. Disjoint from
        // assembleMC2011_'s writes, which target the COARSER level's patches only.
        void selfAssembleMomentsMC2011_(MHDStateT& state, level_t const& level)
        {
            for (auto& patch : level)
            {
                auto const& layout = amr::layoutFromPatch<GridLayoutT>(*patch);
                auto _ = resourcesManager_->setOnPatch(*patch, state.rho, state.rhoV, state.Etot,
                                                       mc2011Assembled_);

                layout.evalOnBox(mc2011Assembled_.rho, [&](auto const&... args) {
                    mc2011Assembled_.rho(args...) = state.rho(args...);
                });
                layout.evalOnBox(mc2011Assembled_.Etot, [&](auto const&... args) {
                    mc2011Assembled_.Etot(args...) = state.Etot(args...);
                });
                for (auto const component :
                    {core::Component::X, core::Component::Y, core::Component::Z})
                {
                    layout.evalOnBox(mc2011Assembled_.rhoV(component), [&](auto const&... args) {
                        mc2011Assembled_.rhoV(component)(args...) = state.rhoV(component)(args...);
                    });
                }
            }
        }

        // Same-level counterpart to assembleMC2011_ for B (see
        // selfAssembleMomentsMC2011_ above for the rationale).
        void selfAssembleMagneticMC2011_(VecFieldT& B, level_t const& level)
        {
            for (auto& patch : level)
            {
                auto const& layout = amr::layoutFromPatch<GridLayoutT>(*patch);
                auto _ = resourcesManager_->setOnPatch(*patch, B, mc2011Assembled_);

                for (auto const component :
                    {core::Component::X, core::Component::Y, core::Component::Z})
                {
                    layout.evalOnBox(mc2011Assembled_.B(component), [&](auto const&... args) {
                        mc2011Assembled_.B(component)(args...) = B(component)(args...);
                    });
                }
            }
        }

        // MC2011 Tier 2/3: assembles mc2011Assembled_ = U^(stageIndex) on the coarser
        // level's own patches, from that level's persisted coarse-step states -- y_n
        // (rhoOld_/rhoVold_/EtotOld_/Bold_, the prepareStep snapshot), the four stage
        // states state1_..state4_ (Butcher stages Y2..Y5) and the final-blend
        // snapshot unp1_ -- per state-backsolve derivation.md S5.1: the five stage
        // derivatives k_i and the Tier 1 split terms are recomputed per point
        // (backSolve + splitTerms, ~40 flops/point) instead of stored, then
        //
        //   ~y(chi)   = y_n + dtCoarse * sum_i b_i(chi) k_i,   b_i(t)=beta1_i t+beta2_i
        //               t^2+beta3_i t^3
        //   ~y'(chi)  = sum_i b_i'(chi) k_i
        //   ~y''(chi) = (1/dtCoarse) sum_i b_i''(chi) k_i
        //   U^(stage) = ~y(chi) + dtFine*gamma1[stage]*~y'(chi)
        //               + dtFine^2*gamma2[stage]*~y''(chi)
        //               + dtFine^3*(gamma3[stage]*splitB + gamma4[stage]*splitA)
        //
        // stageIndex in [0..3] = state1_..state4_ (Butcher stages Y2..Y5), 4 = the
        // final blended state. chi in [0,1] is the fine-substep boundary's fraction
        // of the coarse step. Both call paths -- the stage-path fills and
        // assembleAtChi_'s dtFine=0 reflux path -- route through this one function,
        // so they consume bit-identical back-solved k's by construction (S5.2), and
        // dtCoarse is the coarse sweep's own persisted dt (S5.4), not a t1 - t0
        // recompute. chi=1, stage=4 reproduces the stored unp1_ bit-for-bit
        // (derivation.md S3.3, gate G2).
        void assembleMC2011_(std::size_t const levelNumber, std::size_t const stageIndex,
                             double const chi, double const dtFine)
        {
            // Phase-3 seam asserts (state-backsolve derivation.md S5.7): dtFine == 0
            // is a VALID path (assembleAtChi_'s pure-CE reflux re-assembly), hence
            // >= 0, not > 0. chi may sit epsilon outside [0,1] only through the
            // (fillTime - t0)/(t1 - t0) division in assembleAtChi_ -- reject
            // anything beyond fp noise of the bracket endpoints.
            auto constexpr chiSlack = 1e-12;
            if (stageIndex > 4 or not(chi >= -chiSlack and chi <= 1.0 + chiSlack)
                or not(dtFine >= 0.0))
                throw std::runtime_error("MHDMessenger: assembleMC2011_ out-of-range arguments "
                                         "(stage " + std::to_string(stageIndex) + ", chi "
                                         + std::to_string(chi) + ", dtFine "
                                         + std::to_string(dtFine) + ")");

            auto const& coarseLevel = *coarserLevels_.at(levelNumber);
            double const dtCoarse   = sweepDt_.at(levelNumber - 1);
            if (not(dtCoarse > 0.0))
                throw std::runtime_error(
                    "MHDMessenger: assembleMC2011_ without a persisted coarse sweep dt (level "
                    + std::to_string(levelNumber) + ")");
            double const invDt2 = 1.0 / (dtCoarse * dtCoarse);

            auto assembleField = [&](auto const& yN, auto const& y1f, auto const& y2f,
                                     auto const& y3f, auto const& y4f, auto const& unp1f,
                                     auto& out, GridLayoutT const& layout) {
                layout.evalOnGhostBox(out, [&](auto const&... args) mutable {
                    auto const k = core::mc2011::backSolve(yN(args...), y1f(args...),
                                                           y2f(args...), y3f(args...),
                                                           y4f(args...), unp1f(args...), dtCoarse);
                    auto const [splitA, splitB] = core::mc2011::splitTerms(k, invDt2);
                    out(args...) = core::mc2011::reconstruct(yN(args...), k, splitA, splitB, chi,
                                                             dtCoarse, dtFine, stageIndex);
                });
            };

            for (auto& patch : coarseLevel)
            {
                auto const& layout = amr::layoutFromPatch<GridLayoutT>(*patch);
                auto _ = resourcesManager_->setOnPatch(*patch, rhoOld_, rhoVold_, EtotOld_, Bold_,
                                                       state1_, state2_, state3_, state4_, unp1_,
                                                       mc2011Assembled_);

                assembleField(rhoOld_, state1_.rho, state2_.rho, state3_.rho, state4_.rho,
                             unp1_.rho, mc2011Assembled_.rho, layout);
                assembleField(EtotOld_, state1_.Etot, state2_.Etot, state3_.Etot, state4_.Etot,
                             unp1_.Etot, mc2011Assembled_.Etot, layout);

                for (auto const component :
                    {core::Component::X, core::Component::Y, core::Component::Z})
                {
                    assembleField(rhoVold_(component), state1_.rhoV(component),
                                 state2_.rhoV(component), state3_.rhoV(component),
                                 state4_.rhoV(component), unp1_.rhoV(component),
                                 mc2011Assembled_.rhoV(component), layout);
                    assembleField(Bold_(component), state1_.B(component), state2_.B(component),
                                 state3_.B(component), state4_.B(component), unp1_.B(component),
                                 mc2011Assembled_.B(component), layout);
                }
            }
        }

        // F2: k-less ghost-fill entry points (the solver's post-reflux euler) carry no
        // stage/chi/dtFine, but the static refiners they use are sourced from
        // mc2011Assembled_. Re-assemble at chi = (fillTime - t0)/(t1 - t0) with
        // dtFine = 0: every gamma term in reconstruct() is dtFine-multiplied, so this
        // degenerates to the pure continuous extension ~y(chi) and stageIndex is
        // irrelevant (4 = final blend, by convention).
        void assembleAtChi_(std::size_t const levelNumber, double const fillTime)
        {
            // Phase-3 seam assert: a usable coarse time bracket must exist before
            // any chi can be formed from it (t1 == t0 would divide by zero and
            // means the coarse push times were never recorded for this level).
            double const t0 = beforePushCoarseTime_.at(levelNumber);
            double const t1 = afterPushCoarseTime_.at(levelNumber);
            if (not(t1 > t0))
                throw std::runtime_error(
                    "MHDMessenger: assembleAtChi_ without a coarse time bracket (level "
                    + std::to_string(levelNumber) + ": t0 " + std::to_string(t0) + ", t1 "
                    + std::to_string(t1) + ")");
            assembleMC2011_(levelNumber, /*stageIndex=*/4, (fillTime - t0) / (t1 - t0),
                            /*dtFine=*/0.0);
        }


        void magneticRegriding_(std::shared_ptr<hierarchy_t> const& hierarchy,
                                std::shared_ptr<level_t> const& level,
                                std::shared_ptr<level_t> const& oldLevel, double const initDataTime)
        {
            auto magSchedule = BregridAlgo.createSchedule(
                level, oldLevel, level->getNextCoarserHierarchyLevelNumber(), hierarchy,
                magneticRefinePatchStrategy_.get());

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
        void setNaNsOnFieldGhosts(FieldT& field, patch_t const& patch)
        {
            auto const qty         = field.physicalQuantity();
            using qty_t            = std::decay_t<decltype(qty)>;
            using field_geometry_t = FieldGeometry<GridLayoutT, qty_t>;

            auto const box    = patch.getBox();
            auto const layout = layoutFromPatch<GridLayoutT>(patch);

            // we need to remove the box from the ghost box
            // to use SAMRAI::removeIntersections we do some conversions to
            // samrai box.
            // not gbox is a fieldBox (thanks to the layout)

            auto const gbox  = layout.AMRGhostBoxFor(field.physicalQuantity());
            auto const sgbox = samrai_box_from(gbox);
            auto const fbox  = field_geometry_t::toFieldBox(box, qty, layout);

            // we have field samrai boxes so we can now remove one from the other
            SAMRAI::hier::BoxContainer ghostLayerBoxes{};
            ghostLayerBoxes.removeIntersections(sgbox, fbox);

            // and now finally set the NaNs on the ghost boxes
            for (auto const& gb : ghostLayerBoxes)
                for (auto const& index : layout.AMRToLocal(phare_box_from<dimension>(gb)))
                    field(index) = std::numeric_limits<typename VecFieldT::value_type>::quiet_NaN();
        }

        void setNaNsOnFieldGhosts(FieldT& field, level_t const& level)
        {
            for (auto& patch : resourcesManager_->enumerate(level, field))
                setNaNsOnFieldGhosts(field, *patch);
        }

        void setNaNsOnVecfieldGhosts(VecFieldT& vf, level_t const& level)
        {
            for (auto& patch : resourcesManager_->enumerate(level, vf))
                for (auto& component : vf)
                    setNaNsOnFieldGhosts(component, *patch);
        }


        FieldT rhoOld_{stratName + "rhoOld", core::MHDQuantity::Scalar::rho};
        VecFieldT Vold_{stratName + "Vold", core::MHDQuantity::Vector::V};
        FieldT Pold_{stratName + "Pold", core::MHDQuantity::Scalar::P};

        VecFieldT rhoVold_{stratName + "rhoVold", core::MHDQuantity::Vector::rhoV};
        FieldT EtotOld_{stratName + "EtotOld", core::MHDQuantity::Scalar::Etot};

        VecFieldT Bold_{stratName + "Bold", core::MHDQuantity::Vector::B};

        // MC2011 temporal reconstruction. state1_..state4_/unp1_ are name-shared
        // mirrors of SSPRK4_5Integrator's own same-named members (bare names, not
        // stratName-prefixed, so they resolve to the same underlying PatchData via
        // the ResourcesManager's name-keyed registration -- see the registerResources
        // comment in the ctor); lean KIncrementT views since only the conserved
        // rho/rhoV/Etot/B quads feed the back-solve. mc2011Assembled_ is
        // messenger-exclusive (uniquely named): the coarse-side scratch field Tier 3
        // writes into, consumed by a static refiner (rho/momentum/B only -- E has no
        // such refiner, see UpwindConstrainedTransport's self-sufficient box+1
        // computation).
        KIncrementT state1_{"state1"};
        KIncrementT state2_{"state2"};
        KIncrementT state3_{"state3"};
        KIncrementT state4_{"state4"};
        KIncrementT unp1_{"unp1"};
        KIncrementT mc2011Assembled_{stratName + "MC2011Assembled"};

        // Coarse step time bracket + coarser-level handle, populated by firstStep()
        // (keyed by fine levelNumber, since a deep hierarchy may have several levels
        // simultaneously mid-subcycle).
        std::unordered_map<std::size_t, double> beforePushCoarseTime_;
        std::unordered_map<std::size_t, double> afterPushCoarseTime_;
        std::unordered_map<std::size_t, std::shared_ptr<level_t>> coarserLevels_;

        // Each level's own RK-sweep dt, recorded by the MC2011 ghost-fill overloads
        // (keyed by the level that swept). assembleMC2011_ uses the coarser level's
        // entry as its back-solve divisor: the S5.4 invariant requires assembly's
        // dtCoarse to be the sweep's dt bit-for-bit, which a t1 - t0 recompute from
        // the firstStep() bracket does not guarantee.
        std::unordered_map<std::size_t, double> sweepDt_;

        using rm_t = typename MHDModel::resources_manager_type;
        std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager_;
        int const firstLevel_;

        using InitRefinerPool             = RefinerPool<rm_t, RefinerType::InitField>;
        using GhostRefinerPool            = RefinerPool<rm_t, RefinerType::GhostField>;
        using InitDomPartRefinerPool      = RefinerPool<rm_t, RefinerType::InitInteriorPart>;
        using FieldGhostMaxRefinerPool    = RefinerPool<rm_t, RefinerType::PatchFieldBorderMax>;
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

        GhostRefinerPool rhoGhostsRefiners_{resourcesManager_};
        GhostRefinerPool momentumGhostsRefiners_{resourcesManager_};
        GhostRefinerPool totalEnergyGhostsRefiners_{resourcesManager_};
        FieldGhostMaxRefinerPool rhoMaxRefiners_{resourcesManager_};
        VecFieldGhostMaxRefinerPool momentumMaxRefiners_{resourcesManager_};
        FieldGhostMaxRefinerPool totalEnergyMaxRefiners_{resourcesManager_};
        FieldGhostMaxRefinerPool rhoModelMaxRefiners_{resourcesManager_};
        VecFieldGhostMaxRefinerPool momentumModelMaxRefiners_{resourcesManager_};
        FieldGhostMaxRefinerPool totalEnergyModelMaxRefiners_{resourcesManager_};


        // GhostRefinerPool magFluxesXGhostRefiners_{resourcesManager_};
        // GhostRefinerPool magFluxesYGhostRefiners_{resourcesManager_};
        // GhostRefinerPool magFluxesZGhostRefiners_{resourcesManager_};

        GhostRefinerPool magGhostsRefiners_{resourcesManager_};
        VecFieldGhostMaxRefinerPool magMaxRefiners_{resourcesManager_};
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

        template<typename Policy>
        using FieldRefineOp = FieldRefineOperator<GridLayoutT, GridT, Policy>;

        template<typename Policy>
        using VecFieldRefineOp = VecFieldRefineOperator<VectorFieldDataT, Policy>;

        using DefaultVecFieldRefineOp = VecFieldRefineOp<DefaultFieldRefiner<dimension>>;
        using MagneticFieldRefineOp   = VecFieldRefineOp<MagneticFieldRefiner<dimension>>;
        using MagneticFieldRegridOp   = VecFieldRefineOp<MagneticFieldRegrider<dimension>>;
        using ElectricFieldRefineOp   = VecFieldRefineOp<ElectricFieldRefiner<dimension>>;

        using MHDFluxRefineOp     = FieldRefineOp<MHDFluxRefiner<dimension>>;
        using MHDVecFluxRefineOp  = VecFieldRefineOp<MHDFluxRefiner<dimension>>;
        using MHDFieldRefineOp    = FieldRefineOp<MHDFieldRefiner<dimension>>;
        using MHDVecFieldRefineOp = VecFieldRefineOp<MHDFieldRefiner<dimension>>;

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

        // built in the ctor body (makeRefineOperators_): legacy policies when order==0,
        // composite Linear/Cubic kernels when order==2/4.
        RefOp_ptr mhdFluxRefineOp_;
        RefOp_ptr mhdVecFluxRefineOp_;
        RefOp_ptr mhdFieldRefineOp_;
        RefOp_ptr mhdVecFieldRefineOp_;
        RefOp_ptr EfieldRefineOp_;
        RefOp_ptr BfieldRefineOp_;
        RefOp_ptr BfieldRegridOp_;

        TimeOp_ptr fieldTimeOp_{std::make_shared<FieldTimeInterp>()};
        TimeOp_ptr vecFieldTimeOp_{std::make_shared<VecFieldTimeInterp>()};

        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension /*, rank=1*/>;
        using FieldFillPattern_t       = FieldFillPattern<dimension>;

        std::shared_ptr<FieldFillPattern_t> nonOverwriteFieldFillPattern
            = std::make_shared<FieldFillPattern<dimension>>(); // stateless (mostly)

        std::shared_ptr<TensorFieldFillPattern_t> nonOverwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>();

        std::shared_ptr<TensorFieldFillPattern_t> overwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>(
                /*overwrite_interior=*/true);

        CoarsenOp_ptr mhdFluxCoarseningOp_{std::make_shared<MHDFluxCoarsenOp>()};
        CoarsenOp_ptr mhdVecFluxCoarseningOp_{std::make_shared<MHDVecFluxCoarsenOp>()};
        CoarsenOp_ptr electricFieldCoarseningOp_{std::make_shared<ElectricFieldCoarsenOp>()};

        RefinementConfig config_;

        std::shared_ptr<MagneticPatchStrategyBase> magneticRefinePatchStrategy_;

        std::vector<std::shared_ptr<MagneticPatchStrategyBase>>
            magneticPatchStratPerGhostRefiner_;
    };

} // namespace amr
} // namespace PHARE
#endif
