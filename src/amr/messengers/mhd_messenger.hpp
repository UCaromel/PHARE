#ifndef PHARE_MHD_MESSENGER_HPP
#define PHARE_MHD_MESSENGER_HPP

#include "phare_mpi.hpp"
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
#include "amr/messengers/mhd_messenger_fwd.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/data/field/refine/magnetic_refine_patch_strategy.hpp"
#include "core/numerics/rk_stage_context.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/debug/shared_mismatch.hpp"


#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/RefineOperator.h"
#include "SAMRAI/hier/CoarsenOperator.h"

#include <memory>
#include <string>
#include <type_traits>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel, typename TemporalTransfer>
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
                     int const firstLevel, RefinementConfig const& refinementConfig = {})
            : temporalTransfer_{resourcesManager}
            , resourcesManager_{std::move(resourcesManager)}
            , firstLevel_{firstLevel}
        {
            makeRefineOperators_(refinementConfig);

            resourcesManager_->registerResources(temporalTransfer_);
        }

        virtual ~MHDMessenger() = default;

        void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override
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



        void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                           int const levelNumber) override
        {
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
            magMaxRefiners_.registerLevel(hierarchy, level);
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


        void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                    int const levelNumber,
                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                    IPhysicalModel& model, double const initDataTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            auto& mhdModel = static_cast<MHDModel&>(model);
            auto level     = hierarchy->getPatchLevel(levelNumber);
            mmRegridTime_[levelNumber] = initDataTime;

            bool isRegriddingL0 = levelNumber == 0 and oldLevel;

            if (debug::mm_delta_enabled() and oldLevel)
            {
                auto const* p0 = &**level->begin();
                auto const* q0 = &**oldLevel->begin();
                *debug::mm_log() << "REGRIDPTR L" << levelNumber << " t=" << initDataTime
                                 << " sameLevel=" << (level.get() == oldLevel.get())
                                 << " samePatch0=" << (p0 == q0) << " nNew="
                                 << level->getLocalNumberOfPatches()
                                 << " nOld=" << oldLevel->getLocalNumberOfPatches() << "\n";
                // pre-regrid L2 (old level) snapshot for a post-regrid diff
                mmPre_ = std::make_unique<std::array<debug::MMSnap, 4>>(std::array{
                    debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.B, *oldLevel),
                    debug::mm_snap_field<GridLayoutT>(*resourcesManager_, mhdModel.state.rho,
                                                      *oldLevel),
                    debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.rhoV, *oldLevel),
                    debug::mm_snap_field<GridLayoutT>(*resourcesManager_, mhdModel.state.Etot,
                                                      *oldLevel)});
            }
            if (debug::mm_delta_enabled() and oldLevel)
            {
                // what the freshly allocated new level holds before any regrid fill
                mmCountNaN_("prefill:new", *level, mhdModel.state, initDataTime);
                if (debug::mm_flag("PHARE_MM_NAN_NEWGHOSTS"))
                {
                    auto& st = mhdModel.state;
                    setNaNsOnFieldGhosts(st.rho, *level);
                    setNaNsOnVecfieldGhosts(st.rhoV, *level);
                    setNaNsOnFieldGhosts(st.Etot, *level);
                    setNaNsOnVecfieldGhosts(st.B, *level);
                }
            }
            // MM: PHARE_MM_NAN_OLDGHOSTS=1 -> do new-level ghosts come from old-level ghosts?
            if (oldLevel and debug::mm_flag("PHARE_MM_NAN_OLDGHOSTS"))
            {
                auto& st = mhdModel.state;
                setNaNsOnFieldGhosts(st.rho, *oldLevel);
                setNaNsOnVecfieldGhosts(st.rhoV, *oldLevel);
                setNaNsOnFieldGhosts(st.Etot, *oldLevel);
                setNaNsOnVecfieldGhosts(st.B, *oldLevel);
            }
            magneticRegriding_(hierarchy, level, oldLevel, initDataTime);
            debug::mm_vec<GridLayoutT>("MSG:regrid_preMax", *resourcesManager_, mhdModel.state.B,
                                       *level, initDataTime);
            if (!debug::mm_disable_magmax())
            {
                debug::MMSnap before;
                if (debug::mm_delta_enabled())
                    before = debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.B,
                                                         *level);
                magMaxModelRefiners_.fill(mhdModel.state.B, level->getLevelNumber(),
                                          initDataTime);
                if (debug::mm_delta_enabled())
                    debug::mm_snap_diff(
                        "MSG:magMaxRegrid", before,
                        debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.B, *level),
                        *level, initDataTime);
            }

            densityInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            momentumInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            totalEnergyInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            if (debug::mm_delta_enabled() and oldLevel)
                mmCountNaN_("postregrid:new", *level, mhdModel.state, initDataTime);
            if (mmPre_ and debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.B, *level)
                                   .vals.size()
                               == (*mmPre_)[0].vals.size())
            {
                auto& rm = *resourcesManager_;
                debug::mm_snap_diff("MSG:regridVsOld_B", (*mmPre_)[0],
                                    debug::mm_snap<GridLayoutT>(rm, mhdModel.state.B, *level),
                                    *level, initDataTime);
                debug::mm_snap_diff("MSG:regridVsOld_rho", (*mmPre_)[1],
                                    debug::mm_snap_field<GridLayoutT>(rm, mhdModel.state.rho,
                                                                      *level),
                                    *level, initDataTime);
                debug::mm_snap_diff("MSG:regridVsOld_Etot", (*mmPre_)[3],
                                    debug::mm_snap_field<GridLayoutT>(rm, mhdModel.state.Etot,
                                                                      *level),
                                    *level, initDataTime);
            }
            mmPre_.reset();
            debug::mm_field<GridLayoutT>("MSG:regrid_post", *resourcesManager_,
                                         mhdModel.state.rho, *level, initDataTime);
            debug::mm_vec<GridLayoutT>("MSG:regrid_post", *resourcesManager_, mhdModel.state.rhoV,
                                       *level, initDataTime);

            // MM: PHARE_MM_SRC_PROBE=1 -> which coarse data feed the new L2 (regrid) and the
            // time-interpolated refill: poison L1 ghosts / L1 all / L0 all with NaN, rerun, count
            if constexpr (requires { temporalTransfer_.assembleLevelAt(std::size_t{0}, 0.0); })
                if (debug::mm_flag("PHARE_MM_SRC_PROBE") and levelNumber >= 2 and oldLevel
                    and debug::mm_delta_enabled())
                    srcProbe_(hierarchy, level, oldLevel, mhdModel, initDataTime);

            // MM: PHARE_MM_REGRID_TI=1 -> time-interpolated regrid prototype: put every level
            // <= L-2 at the regrid time (MC2011, dtFine=0), rerun the same regrid schedules, diff
            if constexpr (requires { temporalTransfer_.assembleLevelAt(std::size_t{0}, 0.0); })
                if (debug::mm_flag("PHARE_MM_REGRID_TI") and levelNumber >= 2
                    and debug::mm_delta_enabled())
                {
                    auto& st   = mhdModel.state;
                    auto& rm   = *resourcesManager_;
                    auto a_B   = debug::mm_snap<GridLayoutT>(rm, st.B, *level);
                    auto a_rho = debug::mm_snap_field<GridLayoutT>(rm, st.rho, *level);
                    auto a_rv  = debug::mm_snap<GridLayoutT>(rm, st.rhoV, *level);
                    auto a_et  = debug::mm_snap_field<GridLayoutT>(rm, st.Etot, *level);

                    auto& asm_ = temporalTransfer_.assembled();
                    std::vector<std::vector<std::vector<double>>> saved; // [lvl][field...]
                    auto fieldsOf = [&](auto& S) {
                        return std::array{&S.rho,     &S.Etot,    &S.rhoV(core::Component::X),
                                          &S.rhoV(core::Component::Y),
                                          &S.rhoV(core::Component::Z), &S.B(core::Component::X),
                                          &S.B(core::Component::Y),    &S.B(core::Component::Z)};
                    };
                    bool ok = true;
                    for (int j = 0; j + 2 <= levelNumber; ++j)
                    {
                        if (!temporalTransfer_.assembleLevelAt(static_cast<std::size_t>(j),
                                                                initDataTime))
                        {
                            ok = false;
                            break;
                        }
                        auto& sv = saved.emplace_back();
                        for (auto& patch : *hierarchy->getPatchLevel(j))
                        {
                            auto const layout = layoutFromPatch<GridLayoutT>(*patch);
                            auto _  = rm.setOnPatch(*patch, st.rho, st.rhoV, st.Etot, st.B, asm_);
                            auto dF = fieldsOf(st);
                            auto sF = fieldsOf(asm_);
                            for (std::size_t f = 0; f < dF.size(); ++f)
                            {
                                auto& d = *dF[f];
                                sv.emplace_back(d.data(), d.data() + d.size());
                                auto const& src = *sF[f];
                                layout.evalOnBox(d, [&](auto const&... i) { d(i...) = src(i...); });
                            }
                        }
                    }
                    if (ok)
                    {
                        // refine ops only write NaN destinations: rerun from an all-NaN level
                        auto discard = mmPoison_(*level, st, 1);
                        magneticRegriding_(hierarchy, level, oldLevel, initDataTime);
                        densityInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
                        momentumInitRefiners_.regrid(hierarchy, levelNumber, oldLevel,
                                                     initDataTime);
                        totalEnergyInitRefiners_.regrid(hierarchy, levelNumber, oldLevel,
                                                        initDataTime);
                        debug::mm_snap_diff("MSG:regridTI_B", a_B,
                                            debug::mm_snap<GridLayoutT>(rm, st.B, *level), *level,
                                            initDataTime);
                        debug::mm_snap_diff("MSG:regridTI_rho", a_rho,
                                            debug::mm_snap_field<GridLayoutT>(rm, st.rho, *level),
                                            *level, initDataTime);
                        debug::mm_snap_diff("MSG:regridTI_rhoV", a_rv,
                                            debug::mm_snap<GridLayoutT>(rm, st.rhoV, *level),
                                            *level, initDataTime);
                        debug::mm_snap_diff("MSG:regridTI_Etot", a_et,
                                            debug::mm_snap_field<GridLayoutT>(rm, st.Etot, *level),
                                            *level, initDataTime);
                    }
                    else
                        *debug::mm_log() << "MSG:regridTI skipped L" << levelNumber
                                         << " t=" << initDataTime << "\n";
                    // restore the coarse levels
                    for (std::size_t j = 0; j < saved.size(); ++j)
                    {
                        std::size_t k = 0;
                        for (auto& patch : *hierarchy->getPatchLevel(static_cast<int>(j)))
                        {
                            auto _  = rm.setOnPatch(*patch, st.rho, st.rhoV, st.Etot, st.B);
                            auto dF = fieldsOf(st);
                            for (auto* d : dF)
                            {
                                auto const& v = saved[j][k++];
                                std::copy(v.begin(), v.end(), d->data());
                            }
                        }
                    }
                }

            // MM: PHARE_MM_FIX_REGRID=1 -> candidate fix: refill the new level's ghosts with the
            // time-interpolated conservative ghost fill at the regrid time (no logging needed)
            if (debug::mm_flag("PHARE_MM_FIX_REGRID") and levelNumber > 0 and oldLevel)
                fillConservativeGhosts(mhdModel.state, *level, initDataTime,
                                       core::RKStageContext::endOfStep(initDataTime));

            // MM: PHARE_MM_REGRID_REFILL=1 -> post-regrid ghosts vs a fresh temporal ghost fill
            if (debug::mm_flag("PHARE_MM_REGRID_REFILL") and debug::mm_delta_enabled())
            {
                auto& st   = mhdModel.state;
                auto& rm   = *resourcesManager_;
                auto b_rho = debug::mm_snap_field<GridLayoutT>(rm, st.rho, *level);
                auto b_rv  = debug::mm_snap<GridLayoutT>(rm, st.rhoV, *level);
                auto b_et  = debug::mm_snap_field<GridLayoutT>(rm, st.Etot, *level);
                auto b_B   = debug::mm_snap<GridLayoutT>(rm, st.B, *level);
                try
                {
                    fillConservativeGhosts(st, *level, initDataTime,
                                           core::RKStageContext::endOfStep(initDataTime));
                    debug::mm_snap_diff("MSG:regridRefill_B", b_B,
                                        debug::mm_snap<GridLayoutT>(rm, st.B, *level), *level,
                                        initDataTime);
                    debug::mm_snap_diff("MSG:regridRefill_rho", b_rho,
                                        debug::mm_snap_field<GridLayoutT>(rm, st.rho, *level),
                                        *level, initDataTime);
                    debug::mm_snap_diff("MSG:regridRefill_rhoV", b_rv,
                                        debug::mm_snap<GridLayoutT>(rm, st.rhoV, *level), *level,
                                        initDataTime);
                    debug::mm_snap_diff("MSG:regridRefill_Etot", b_et,
                                        debug::mm_snap_field<GridLayoutT>(rm, st.Etot, *level),
                                        *level, initDataTime);
                }
                catch (std::exception const& e)
                {
                    *debug::mm_log() << "MSG:regridRefill skipped L" << levelNumber
                                     << " t=" << initDataTime << ": " << e.what() << "\n";
                }
            }

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
            if (!temporalTransfer_.hasCoarseFine())
                return;

            auto levelNumber = level.getLevelNumber();

            auto& mhdModel = static_cast<MHDModel&>(model);

            magInitRefineSchedules[levelNumber]->fillData(initDataTime);
            debug::mm_vec<GridLayoutT>("MSG:initLevel", *resourcesManager_, mhdModel.state.B,
                                       level, initDataTime);
            densityInitRefiners_.fill(levelNumber, initDataTime);
            momentumInitRefiners_.fill(levelNumber, initDataTime);
            totalEnergyInitRefiners_.fill(levelNumber, initDataTime);
        }

        void firstStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                       double const currentTime, double const prevCoarserTIme,
                       double const newCoarserTime) final
        {
            temporalTransfer_.firstStep(level, hierarchy, prevCoarserTIme, newCoarserTime);

            // MM: PHARE_MM_REFLUX_PROBE=1 -> at a step that starts at a coarser sync time (all
            // reflux of that sync done), refill the ghosts from the coarser state at that time
            // (chi = 0: the coarser step's start state, snapshotted after its reflux), diff
            // against the ghosts the step would use, restore them
            bool const probe = debug::mm_flag("PHARE_MM_REFLUX_PROBE") and debug::mm_delta_enabled();
            // MM: PHARE_MM_REFLUX_FIX=1 keeps the refilled ghosts (prototype fix (ii))
            bool const fix = debug::mm_flag("PHARE_MM_REFLUX_FIX");
            if (level.getLevelNumber() > 0 and (probe or fix))
            {
                auto& st = static_cast<MHDModel&>(model).state;
                auto& rm = *resourcesManager_;
                debug::MMSnap b_B, b_rho, b_rv, b_et;
                if (probe)
                {
                    b_B   = debug::mm_snap<GridLayoutT>(rm, st.B, level);
                    b_rho = debug::mm_snap_field<GridLayoutT>(rm, st.rho, level);
                    b_rv  = debug::mm_snap<GridLayoutT>(rm, st.rhoV, level);
                    b_et  = debug::mm_snap_field<GridLayoutT>(rm, st.Etot, level);
                }
                fillConservativeGhosts(st, level, currentTime,
                                       core::RKStageContext::endOfStep(currentTime));
                if (probe)
                {
                    auto const rg = mmRegridTime_.contains(level.getLevelNumber())
                                    and mmRegridTime_[level.getLevelNumber()] == currentTime;
                    std::string const sfx = rg ? "_rg" : "";
                    debug::mm_snap_diff("MSG:refluxProbe_B" + sfx, b_B,
                                        debug::mm_snap<GridLayoutT>(rm, st.B, level), level,
                                        currentTime);
                    debug::mm_snap_diff("MSG:refluxProbe_rho" + sfx, b_rho,
                                        debug::mm_snap_field<GridLayoutT>(rm, st.rho, level),
                                        level, currentTime);
                    debug::mm_snap_diff("MSG:refluxProbe_rhoV" + sfx, b_rv,
                                        debug::mm_snap<GridLayoutT>(rm, st.rhoV, level), level,
                                        currentTime);
                    debug::mm_snap_diff("MSG:refluxProbe_Etot" + sfx, b_et,
                                        debug::mm_snap_field<GridLayoutT>(rm, st.Etot, level),
                                        level, currentTime);
                    if (!fix)
                    {
                        debug::mm_restore<GridLayoutT>(rm, st.B, level, b_B);
                        debug::mm_restore_field<GridLayoutT>(rm, st.rho, level, b_rho);
                        debug::mm_restore<GridLayoutT>(rm, st.rhoV, level, b_rv);
                        debug::mm_restore_field<GridLayoutT>(rm, st.Etot, level, b_et);
                    }
                }
            }
        }


        void lastStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level) final {}


        // the old state used for time interpolation is snapshotted by SolverMHD::prepareStep
        void prepareStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                         double /*currentTime*/) final
        {
        }

        void fillRootGhosts(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                            double const initDataTime) final
        {
            auto& mhdModel = static_cast<MHDModel&>(model);
            debug::mm_vec<GridLayoutT>("MSG:root_pre", *resourcesManager_, mhdModel.state.B, level,
                                       initDataTime);
            // MM: one-time equalize of initial B (periodic images initialised at two coords)
            if (debug::mm_flag("PHARE_MM_INITMAX"))
            {
                debug::MMSnap before;
                if (debug::mm_delta_enabled())
                    before = debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.B,
                                                         level);
                magMaxModelRefiners_.fill(mhdModel.state.B, level.getLevelNumber(), initDataTime);
                if (debug::mm_delta_enabled())
                    debug::mm_snap_diff(
                        "MSG:magMaxInit", before,
                        debug::mm_snap<GridLayoutT>(*resourcesManager_, mhdModel.state.B, level),
                        level, initDataTime);
            }
            if (debug::mm_flag("PHARE_MM_ROOTFILL"))
            {
                fillConservativeGhosts(mhdModel.state, level, initDataTime,
                                       core::RKStageContext::endOfStep(initDataTime));
                debug::mm_vec<GridLayoutT>("MSG:rootfill", *resourcesManager_, mhdModel.state.B,
                                           level, initDataTime);
            }
        }

        void synchronize(SAMRAI::hier::PatchLevel& level) final {}

        void reflux(int const coarserLevelNumber, int const fineLevelNumber,
                    double const syncTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;
            // MM: PHARE_MM_NOREFLUX=1 -> skip the fine-to-coarse flux correction
            if (debug::mm_flag("PHARE_MM_NOREFLUX"))
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

        void postSynchronize(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                             double const time) override
        {
            // The ghosts for B are obtained in the solver's reflux_euler. For B, this is because
            // refluxing is done through faraday which is computed on the ghost box for the other
            // quantities, the ghosts are filled in the end of the euler step anyways.
        }

        // MM: PHARE_MM_REFLUX_FIX (i), called by SolverMHD::reflux on the refluxed level
        void mmOverwriteFinal(MHDStateT& state, level_t const& level)
        {
            temporalTransfer_.overwriteFinal(state, level);
        }

        void fillConservativeGhosts(MHDStateT& state, level_t const& level, double const fillTime,
                                    core::RKStageContext const& context)
        {
            setNaNsOnConservativeGhosts_(state, level);
            temporalTransfer_.prepareTarget(state, level, context);
            fillConservativeGhosts_(state, level, fillTime);

            // MM: PHARE_MM_STAGE_PROBE=1 -> refill with EVERY coarser level assembled at the
            // stage target and diff (stale levels <= L-2 in the default MC2011 fill)
            if constexpr (requires { temporalTransfer_.prepareTargetAll(state, level, context); })
                if (debug::mm_flag("PHARE_MM_STAGE_PROBE") and level.getLevelNumber() >= 2
                    and debug::mm_delta_enabled())
                {
                    auto& rm   = *resourcesManager_;
                    auto b_B   = debug::mm_snap<GridLayoutT>(rm, state.B, level);
                    auto b_rho = debug::mm_snap_field<GridLayoutT>(rm, state.rho, level);
                    auto b_rv  = debug::mm_snap<GridLayoutT>(rm, state.rhoV, level);
                    auto b_et  = debug::mm_snap_field<GridLayoutT>(rm, state.Etot, level);
                    setNaNsOnConservativeGhosts_(state, level);
                    temporalTransfer_.prepareTargetAll(state, level, context);
                    fillConservativeGhosts_(state, level, fillTime);
                    debug::mm_snap_diff("MSG:stageAll_B", b_B,
                                        debug::mm_snap<GridLayoutT>(rm, state.B, level), level,
                                        fillTime);
                    debug::mm_snap_diff("MSG:stageAll_rho", b_rho,
                                        debug::mm_snap_field<GridLayoutT>(rm, state.rho, level),
                                        level, fillTime);
                    debug::mm_snap_diff("MSG:stageAll_rhoV", b_rv,
                                        debug::mm_snap<GridLayoutT>(rm, state.rhoV, level), level,
                                        fillTime);
                    debug::mm_snap_diff("MSG:stageAll_Etot", b_et,
                                        debug::mm_snap_field<GridLayoutT>(rm, state.Etot, level),
                                        level, fillTime);
                }
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
        // ---- MM source probe ------------------------------------------------------------------
        static auto mmFieldsOf_(auto& S)
        {
            return std::array{&S.rho,
                              &S.Etot,
                              &S.rhoV(core::Component::X),
                              &S.rhoV(core::Component::Y),
                              &S.rhoV(core::Component::Z),
                              &S.B(core::Component::X),
                              &S.B(core::Component::Y),
                              &S.B(core::Component::Z)};
        }
        static constexpr std::array<char const*, 8> mmFieldNames_{"rho", "Etot", "rhoVx", "rhoVy",
                                                                  "rhoVz", "Bx",   "By",   "Bz"};

        // saves then NaNs `which` (0 = ghosts only, 1 = everything) of state S on level lvl
        template<typename S>
        std::vector<std::vector<double>> mmPoison_(level_t& lvl, S& st, int const which)
        {
            std::vector<std::vector<double>> saved;
            auto& rm = *resourcesManager_;
            for (auto& patch : lvl)
            {
                auto const layout = layoutFromPatch<GridLayoutT>(*patch);
                auto _            = rm.setOnPatch(*patch, st);
                for (auto* f : mmFieldsOf_(st))
                {
                    saved.emplace_back(f->data(), f->data() + f->size());
                    if (which == 1)
                        std::fill(f->data(), f->data() + f->size(),
                                  std::numeric_limits<double>::quiet_NaN());
                    else if (which == 0)
                        setNaNsOnFieldGhosts(*f, *patch);
                }
            }
            return saved;
        }
        template<typename S>
        void mmRestore_(level_t& lvl, S& st, std::vector<std::vector<double>> const& saved)
        {
            auto& rm      = *resourcesManager_;
            std::size_t k = 0;
            for (auto& patch : lvl)
            {
                auto _ = rm.setOnPatch(*patch, st);
                for (auto* f : mmFieldsOf_(st))
                {
                    auto const& v = saved[k++];
                    std::copy(v.begin(), v.end(), f->data());
                }
            }
        }
        // NaN counts on the new level, interior vs ghost, per field
        void mmCountNaN_(std::string const& label, level_t& lvl, MHDStateT& st, double t)
        {
            auto& rm = *resourcesManager_;
            std::array<std::size_t, 8> nIn{}, nGh{};
            for (auto& patch : lvl)
            {
                auto const layout = layoutFromPatch<GridLayoutT>(*patch);
                auto _            = rm.setOnPatch(*patch, st.rho, st.rhoV, st.Etot, st.B);
                auto fs           = mmFieldsOf_(st);
                for (std::size_t i = 0; i < fs.size(); ++i)
                {
                    auto& f         = *fs[i];
                    auto const ibox = layout.AMRBoxFor(f);
                    for (auto const& a : layout.AMRGhostBoxFor(f))
                        if (std::isnan(f(layout.AMRToLocal(a))))
                            ++(core::isIn(a, ibox) ? nIn[i] : nGh[i]);
                }
            }
            auto& out = *debug::mm_log();
            out << "SRC L" << lvl.getLevelNumber() << " t=" << t << " " << label;
            for (std::size_t i = 0; i < 8; ++i)
                out << " " << mmFieldNames_[i] << ":in=" << nIn[i] << ",gh=" << nGh[i];
            out << "\n";
            out.flush();
        }

        void srcProbe_(std::shared_ptr<hierarchy_t> const& hierarchy,
                       std::shared_ptr<level_t> const& level,
                       std::shared_ptr<level_t> const& oldLevel, MHDModel& mhdModel,
                       double const t)
        {
            auto& st         = mhdModel.state;
            auto& asm_       = temporalTransfer_.assembled();
            auto const L     = level->getLevelNumber();
            auto& L1         = *hierarchy->getPatchLevel(L - 1);
            auto& L0         = *hierarchy->getPatchLevel(L - 2);
            auto regridAgain = [&] {
                magneticRegriding_(hierarchy, level, oldLevel, t);
                densityInitRefiners_.regrid(hierarchy, L, oldLevel, t);
                momentumInitRefiners_.regrid(hierarchy, L, oldLevel, t);
                totalEnergyInitRefiners_.regrid(hierarchy, L, oldLevel, t);
            };
            auto newSaved = mmPoison_(*level, st, 2); // which=2: save only, no poison
            // --- regrid path: sources are the model state of the coarser levels
            for (auto [name, lvl, which] : {std::tuple{"regrid:none", &L1, 2}, std::tuple{"regrid:L1ghosts", &L1, 0},
                                            std::tuple{"regrid:L1all", &L1, 1},
                                            std::tuple{"regrid:L0all", &L0, 1}})
            {
                auto sv = mmPoison_(*lvl, st, which);
                auto sn = mmPoison_(*level, st, 1); // start the rerun from an all-NaN new level
                regridAgain();
                mmCountNaN_(name, *level, st, t);
                mmRestore_(*lvl, st, sv);
            }
            // --- refill path: sources are the MC2011 assembled state
            auto const ctx = core::RKStageContext::endOfStep(t);
            for (auto [name, lvl, which] : {std::tuple{"refill:none", &L1, 2}, std::tuple{"refill:L1ghosts", &L1, 0},
                                            std::tuple{"refill:L1all", &L1, 1},
                                            std::tuple{"refill:L0all", &L0, 1}})
            {
                setNaNsOnConservativeGhosts_(st, *level);
                temporalTransfer_.prepareTarget(st, *level, ctx);
                auto sv = mmPoison_(*lvl, asm_, which);
                fillConservativeGhosts_(st, *level, t);
                mmCountNaN_(name, *level, st, t);
                mmRestore_(*lvl, asm_, sv);
            }
            mmRestore_(*level, st, newSaved);
        }
        // ----------------------------------------------------------------------------------------

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
            debug::mm_vec<GridLayoutT>("MSG:Bghost_preMax", *resourcesManager_, state.B, level,
                                       fillTime);
            if (!debug::mm_disable_magmax())
            {
                debug::MMSnap before;
                if (debug::mm_delta_enabled())
                    before = debug::mm_snap<GridLayoutT>(*resourcesManager_, state.B, level);
                auto const t0 = std::chrono::steady_clock::now();
                magMaxRefiners_.fill(state.B, level.getLevelNumber(), fillTime);
                debug::mm_timer().seconds += std::chrono::duration<double>(
                                                 std::chrono::steady_clock::now() - t0)
                                                 .count();
                if (++debug::mm_timer().calls % 20 == 0 and debug::mm_flag("PHARE_MM_TIME"))
                    std::fprintf(stderr, "MMTIME magMax calls=%zu seconds=%.3f\n",
                                 debug::mm_timer().calls, debug::mm_timer().seconds);
                if (debug::mm_delta_enabled())
                    debug::mm_snap_diff("MSG:magMax", before,
                                        debug::mm_snap<GridLayoutT>(*resourcesManager_, state.B,
                                                                    level),
                                        level, fillTime);
            }
            rhoGhostsRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumGhostsRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyGhostsRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);
            debug::mm_field<GridLayoutT>("MSG:mfill", *resourcesManager_, state.rho, level,
                                         fillTime);
            debug::mm_vec<GridLayoutT>("MSG:mfill", *resourcesManager_, state.rhoV, level,
                                       fillTime);
            debug::mm_field<GridLayoutT>("MSG:mfill", *resourcesManager_, state.Etot, level,
                                         fillTime);
        }

        // without a coarse-fine path, ghosts come from the same level and its periodic images only
        void registerNoCoarseFineGhostComms_(MHDMessengerInfo const& info)
        {
            elecGhostsRefiners_.addStaticRefiners(info.ghostElectric, nullptr, info.ghostElectric,
                                                  nonOverwriteInteriorTFfillPattern);
            rhoGhostsRefiners_.addStaticRefiners(info.ghostDensity, nullptr, info.ghostDensity,
                                                 nonOverwriteFieldFillPattern);
            momentumGhostsRefiners_.addStaticRefiners(
                info.ghostMomentum, nullptr, info.ghostMomentum, nonOverwriteInteriorTFfillPattern);
            totalEnergyGhostsRefiners_.addStaticRefiners(info.ghostTotalEnergy, nullptr,
                                                         info.ghostTotalEnergy,
                                                         nonOverwriteFieldFillPattern);

            magFluxesXGhostRefiners_.addStaticRefiners(info.ghostMagneticFluxesX, nullptr,
                                                       info.ghostMagneticFluxesX,
                                                       nonOverwriteInteriorTFfillPattern);
            magFluxesYGhostRefiners_.addStaticRefiners(info.ghostMagneticFluxesY, nullptr,
                                                       info.ghostMagneticFluxesY,
                                                       nonOverwriteInteriorTFfillPattern);
            magFluxesZGhostRefiners_.addStaticRefiners(info.ghostMagneticFluxesZ, nullptr,
                                                       info.ghostMagneticFluxesZ,
                                                       nonOverwriteInteriorTFfillPattern);

            for (auto const& key : info.ghostMagnetic)
            {
                magGhostsRefiners_.addStaticRefiner(key, nullptr, key,
                                                    nonOverwriteInteriorTFfillPattern);
                magMaxRefiners_.addStaticRefiner(
                    key, key, nullptr, key,
                    std::make_shared<
                        TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());
            }
        }

        // Select field-refinement operators once at construction.
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
            elecGhostsRefiners_.addStaticRefiners(info->ghostElectric, EfieldRefineOp_,
                                                  info->ghostElectric,
                                                  nonOverwriteInteriorTFfillPattern);

            if constexpr (linearInTime)
            {
                auto const& old = info->oldState;

                rhoGhostsRefiners_.addTimeRefiners(info->ghostDensity, info->modelDensity,
                                                   old.rho(), mhdFieldRefineOp_, fieldTimeOp_,
                                                   nonOverwriteFieldFillPattern);
                momentumGhostsRefiners_.addTimeRefiners(
                    info->ghostMomentum, info->modelMomentum, old.rhoV(), mhdVecFieldRefineOp_,
                    vecFieldTimeOp_, nonOverwriteInteriorTFfillPattern);
                totalEnergyGhostsRefiners_.addTimeRefiners(
                    info->ghostTotalEnergy, info->modelTotalEnergy, old.Etot(), mhdFieldRefineOp_,
                    fieldTimeOp_, nonOverwriteFieldFillPattern);
            }
            else
            {
                auto const& assembled = temporalTransfer_.assembledStateNames();

                for (auto const& ghost : info->ghostDensity)
                    rhoGhostsRefiners_.addStaticRefiner(ghost, assembled.rho(), mhdFieldRefineOp_,
                                                        ghost, nonOverwriteFieldFillPattern);
                for (auto const& ghost : info->ghostMomentum)
                    momentumGhostsRefiners_.addStaticRefiner(ghost, assembled.rhoV(),
                                                             mhdVecFieldRefineOp_, ghost,
                                                             nonOverwriteInteriorTFfillPattern);
                for (auto const& ghost : info->ghostTotalEnergy)
                    totalEnergyGhostsRefiners_.addStaticRefiner(ghost, assembled.Etot(),
                                                                mhdFieldRefineOp_, ghost,
                                                                nonOverwriteFieldFillPattern);
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
            magneticPatchStratPerGhostRefiner_ = [&]() {
                std::vector<std::shared_ptr<MagneticRefinePatchStrategy<VectorFieldDataT>>> result;

                result.reserve(info->ghostMagnetic.size());

                for (auto const& key : info->ghostMagnetic)
                {
                    auto&& [id] = resourcesManager_->getIDsList(key);

                    auto patch_strat
                        = std::make_shared<MagneticRefinePatchStrategy<VectorFieldDataT>>();

                    patch_strat->registerIDs(id);

                    result.push_back(patch_strat);
                }
                return result;
            }();

            for (size_t i = 0; i < info->ghostMagnetic.size(); ++i)
            {
                if constexpr (linearInTime)
                    magGhostsRefiners_.addTimeRefiner(
                        info->ghostMagnetic[i], info->modelMagnetic, info->oldState.B(),
                        BfieldRegridOp_, vecFieldTimeOp_, info->ghostMagnetic[i],
                        nonOverwriteInteriorTFfillPattern, magneticPatchStratPerGhostRefiner_[i]);
                else
                    magGhostsRefiners_.addStaticRefiner(
                        info->ghostMagnetic[i], temporalTransfer_.assembledStateNames().B(),
                        BfieldRegridOp_, info->ghostMagnetic[i], nonOverwriteInteriorTFfillPattern,
                        magneticPatchStratPerGhostRefiner_[i]);

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


        void magneticRegriding_(std::shared_ptr<hierarchy_t> const& hierarchy,
                                std::shared_ptr<level_t> const& level,
                                std::shared_ptr<level_t> const& oldLevel, double const initDataTime)
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


        TemporalTransfer temporalTransfer_;
        std::unique_ptr<std::array<debug::MMSnap, 4>> mmPre_;
        std::unordered_map<int, double> mmRegridTime_;

        using rm_t = typename MHDModel::resources_manager_type;
        std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager_;
        int const firstLevel_;

        using InitRefinerPool             = RefinerPool<rm_t, RefinerType::InitField>;
        using GhostRefinerPool            = RefinerPool<rm_t, RefinerType::GhostField>;
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

        GhostRefinerPool elecGhostsRefiners_{resourcesManager_};
        GhostRefinerPool rhoGhostsRefiners_{resourcesManager_};
        // GhostRefinerPool velGhostsRefiners_{resourcesManager_};
        // GhostRefinerPool pressureGhostsRefiners_{resourcesManager_};
        GhostRefinerPool momentumGhostsRefiners_{resourcesManager_};
        GhostRefinerPool totalEnergyGhostsRefiners_{resourcesManager_};
        GhostRefinerPool magFluxesXGhostRefiners_{resourcesManager_};
        GhostRefinerPool magFluxesYGhostRefiners_{resourcesManager_};
        GhostRefinerPool magFluxesZGhostRefiners_{resourcesManager_};

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

        std::shared_ptr<TensorFieldFillPattern_t> overwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>(
                /*overwrite_interior=*/true);

        CoarsenOp_ptr mhdFluxCoarseningOp_{std::make_shared<MHDFluxCoarsenOp>()};
        CoarsenOp_ptr mhdVecFluxCoarseningOp_{std::make_shared<MHDVecFluxCoarsenOp>()};
        CoarsenOp_ptr electricFieldCoarseningOp_{std::make_shared<ElectricFieldCoarsenOp>()};

        MagneticRefinePatchStrategy<VectorFieldDataT> magneticRefinePatchStrategy_{};

        std::vector<std::shared_ptr<MagneticRefinePatchStrategy<VectorFieldDataT>>>
            magneticPatchStratPerGhostRefiner_;
    };

} // namespace amr
} // namespace PHARE
#endif
