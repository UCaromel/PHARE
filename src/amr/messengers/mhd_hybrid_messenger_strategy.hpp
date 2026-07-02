#ifndef PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP
#define PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP

#include "amr/messengers/cross_model_fill_context.hpp"
#include "amr/messengers/dispatching_refine_patch_strategy.hpp"
#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "amr/messengers/mhd_hybrid_particle_spawn_strategy.hpp"
#include "amr/messengers/mhd_hybrid/mhd_hybrid_reflux_comms.hpp"
#include "amr/messengers/hybrid_hybrid/hybrid_border_comms.hpp"
#include "amr/messengers/refiner_pool.hpp"
#include "amr/messengers/synchronizer_pool.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/magnetic_field_coarsener.hpp"
#include "amr/data/field/coarsening/mhd_field_coarsener.hpp"
#include "amr/data/particles/particles_variable_fill_pattern.hpp"
#include "core/physical_quantities.hpp"
#include "core/data/electrons/electrons.hpp"
#include "core/numerics/interpolator/interpolator.hpp"
#include "amr/data/tensorfield/tensor_field_data.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/magnetic_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_field_init_refiner.hpp"
#include "amr/data/field/refine/electric_field_refiner.hpp"
#include "amr/data/field/refine/mhd_field_refiner.hpp"
#include "amr/data/field/refine/mhd_flux_refiner.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "initializer/data_provider.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"

#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefineSchedule.h>
#include <SAMRAI/xfer/PatchLevelBorderFillPattern.h>
#include <SAMRAI/xfer/PatchLevelFullFillPattern.h>
#include <SAMRAI/xfer/PatchLevelInteriorFillPattern.h>

#include "amr/data/particles/particles_data.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel, typename HybridModel, typename RefinementParams>
    class MHDHybridMessengerStrategy : public HybridMessengerStrategy<HybridModel>
    {
        static constexpr std::size_t dimension       = HybridModel::dimension;
        static constexpr std::size_t rootLevelNumber = 0;

        using IonsT          = decltype(std::declval<HybridModel>().state.ions);
        using VecFieldT      = decltype(std::declval<HybridModel>().state.electromag.E);
        using TensorFieldT   = typename std::decay_t<IonsT>::tensorfield_type;
        using IPhysicalModel = typename HybridModel::Interface;

        using HybridGridLayoutT = typename HybridModel::gridlayout_type;
        using HybridGridT       = typename HybridModel::grid_type;
        using MHDGridLayoutT    = typename MHDModel::gridlayout_type;
        using MHDGridT          = typename MHDModel::grid_type;

        using RMType = typename HybridModel::resources_manager_type;

        // VecFieldData type for Hybrid B (needed by BfieldComms / MagneticRefinePatchStrategy)
        using HybVectorFieldDataT
            = TensorFieldData<1, HybridGridLayoutT, HybridGridT, core::PhysicalQuantity>;

        template<typename Policy>
        using MHDVecFieldRefineOp = VecFieldRefineOperator<MHDGridLayoutT, MHDGridT, Policy>;

        // Same-type MHD refine ops passed to refluxComms_ and ghost fills
        using MHDERefineOp       = MHDVecFieldRefineOp<ElectricFieldRefiner<dimension>>;
        using MHDMagRefineOp     = MHDVecFieldRefineOp<MagneticFieldRefiner<dimension>>;
        using MHDMagInitRefineOp = MHDVecFieldRefineOp<MagneticFieldInitRefiner<dimension>>;
        using MHDFluxRefineOp = FieldRefineOperator<MHDGridLayoutT, MHDGridT,
                                                    MHDFluxRefiner<dimension>>;
        using MHDVecFluxRefineOp = MHDVecFieldRefineOp<MHDFluxRefiner<dimension>>;

        // Prim-field refine ops for the cons→prim→fine spawn pipeline
        using MHDScalarPrimRefineOp
            = FieldRefineOperator<MHDGridLayoutT, MHDGridT, MHDFieldRefiner<dimension>>;
        using MHDVecPrimRefineOp = MHDVecFieldRefineOp<MHDFieldRefiner<dimension>>;

        // Covered-interior sync ops: cell-average coarsening of the ddd staging fields
        // onto the MHD conservatives (the MHD-MHD hydro sync operator), and
        // face-flux-preserving magnetic coarsening on the shared B id (MHD-MHD verbatim).
        // Channels must be same-centering: SAMRAI CoarsenSchedule coarsens src into a
        // src-typed temporary coarse level (CoarsenSchedule.cpp passes source_id for both
        // operator args) and copies temp→dst afterwards, so a cross-centering coarsen
        // operator never sees the real dst. MHD template types serve both sides —
        // FieldData/TensorFieldData types are shared between the models (same precedent
        // as MHDHybridRefluxComms).
        using MHDFieldCoarsenOp
            = FieldCoarsenOperator<MHDGridLayoutT, MHDGridT, MHDFieldCoarsener<dimension>>;
        using MHDVecFieldCoarsenOp
            = VecFieldCoarsenOperator<MHDGridLayoutT, MHDGridT, MHDFieldCoarsener<dimension>,
                                      core::PhysicalQuantity>;
        using MagneticFieldCoarsenOp
            = VecFieldCoarsenOperator<MHDGridLayoutT, MHDGridT, MagneticFieldCoarsener<dimension>,
                                      core::PhysicalQuantity>;
        using CoarsenOp_ptr = std::shared_ptr<SAMRAI::hier::CoarsenOperator>;

        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension>;

        using FieldT = VecFieldT::field_type;
        static constexpr std::size_t interpOrder = HybridGridLayoutT::interp_order;
        using rm_t                   = RMType;
        using DomainGhostPartRefinerPool = RefinerPool<rm_t, RefinerType::ExteriorGhostParticles>;

        using MHDFieldDataT    = FieldData<MHDGridLayoutT, MHDGridT>;
        using MHDVecFieldDataT = TensorFieldData<1, MHDGridLayoutT, MHDGridT, core::PhysicalQuantity>;
        using ParticleArrayT   = typename IonsT::particle_array_type;
        using ParticlesDataT        = ParticlesData<ParticleArrayT>;
        using ParticleSpawnStrategy = MHDHybridParticleSpawnStrategy<
            MHDFieldDataT, MHDVecFieldDataT, ParticlesDataT, MHDGridLayoutT>;

    public:
        static inline std::string const stratName = "MHDModel-HybridModel";

        MHDHybridMessengerStrategy(std::shared_ptr<RMType> const& rm, int const firstLevel,
                                   std::shared_ptr<CrossModelFillContext> crossModelContext
                                   = nullptr)
            : HybridMessengerStrategy<HybridModel>{stratName}
            , resourcesManager_{rm}
            , firstLevel_{firstLevel}
            , crossModelContext_{std::move(crossModelContext)}
            , magComms_{*rm}
        {
            resourcesManager_->registerResources(sumVec_);
            resourcesManager_->registerResources(sumField_);
            resourcesManager_->registerResources(syncRho_);
            resourcesManager_->registerResources(syncRhoV_);
            resourcesManager_->registerResources(syncEtot_);
            // primRho_/primV_/primP_ are NOT registered here: they become aliases of the
            // MHD model's rho/V/P at registerQuantities time (shareResources in
            // registerInitComms_), where the model field names are first known.
            resourcesManager_->registerResources(diagSumField_);
            resourcesManager_->registerResources(diagSumVec_);
            resourcesManager_->registerResources(diagSumTensor_);
        }

        void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override
        {
            resourcesManager_->allocate(sumVec_, patch, allocateTime);
            resourcesManager_->allocate(sumField_, patch, allocateTime);
            resourcesManager_->allocate(syncRho_, patch, allocateTime);
            resourcesManager_->allocate(syncRhoV_, patch, allocateTime);
            resourcesManager_->allocate(syncEtot_, patch, allocateTime);
            resourcesManager_->allocate(primRho_, patch, allocateTime);
            resourcesManager_->allocate(primV_, patch, allocateTime);
            resourcesManager_->allocate(primP_, patch, allocateTime);
            resourcesManager_->allocate(diagSumField_, patch, allocateTime);
            resourcesManager_->allocate(diagSumVec_, patch, allocateTime);
            resourcesManager_->allocate(diagSumTensor_, patch, allocateTime);
        }

        void registerQuantities(std::unique_ptr<IMessengerInfo> fromCoarserInfo,
                                std::unique_ptr<IMessengerInfo> fromFinerInfo) override
        {
            std::unique_ptr<MHDMessengerInfo> mhdInfo{
                dynamic_cast<MHDMessengerInfo*>(fromCoarserInfo.release())};
            std::unique_ptr<HybridMessengerInfo> hybridInfo{
                dynamic_cast<HybridMessengerInfo*>(fromFinerInfo.release())};
            if (!mhdInfo)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: fromCoarserInfo is not MHDMessengerInfo");
            if (!hybridInfo)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: fromFinerInfo is not HybridMessengerInfo");

            registerGhostComms_(mhdInfo, hybridInfo);
            registerInitComms_(mhdInfo, hybridInfo);
            registerSyncComms_(mhdInfo);
            refluxComms_.registerQuantities(*mhdInfo, *hybridInfo, *resourcesManager_,
                                            mhdERefineOp_, mhdFluxRefineOp_, mhdVecFluxRefineOp_,
                                            nonOverwriteInteriorTFfillPattern_);
            if (crossModelContext_)
                registerCrossModelSlots_();
        }

        void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                           int const levelNumber) override
        {
            hierarchy_ = hierarchy;
            refluxComms_.registerLevel(levelNumber, hierarchy);

            if (levelNumber != rootLevelNumber)
            {
                auto const level = hierarchy->getPatchLevel(levelNumber);

                magComms_.magInitRefineSchedules_[levelNumber]
                    = magComms_.BalgoInit.createSchedule(level, nullptr, levelNumber - 1, hierarchy,
                                                         &magComms_.magneticRefinePatchStrategy_);

                eInitComms_.createSchedule(levelNumber, level, levelNumber - 1, hierarchy);

                // patchGhost: same-level peer exchange (src=hyb_*_id, dst=hyb_*_id, same level).
                // Uses SAMRAI overload A: createSchedule(dst, src) — no hierarchy, no
                // next_coarser_level → no coarser sub-schedule created → hyb_*_id never looked
                // up on the coarse MHD level where it is not allocated.
                // levelGhost: coarse→fine from MHD (InitField+BorderFillPattern, src_level=nullptr
                // → no same-level transactions; BorderFillPattern → coarse-fine boundary ghost
                // cells only). Periodic-domain-boundary ghosts and same-level peer ghosts are
                // expected to be covered by patchGhost. nonOverwrite interior pattern
                // (registerRefine) preserves the evolved Hybrid interior.
                // In fill*Ghosts the levelGhost schedule runs BEFORE patchGhost so same-level
                // exact data wins on overlap (x-periodic edges) over coarse interpolation.
                // NOTE: periodic-x fine ghosts are NOT filled by the coarse→fine pass (the
                // src_level=nullptr form builds no periodic coarse image); the same-level
                // patchGhost pass covers them (ghost scans clean after both passes). This split
                // pair now serves strategy TEMPORARIES only — model fields use the one-pass
                // shared-id GhostField form below.
                auto ghostBorderFill
                    = std::make_shared<SAMRAI::xfer::PatchLevelBorderFillPattern>();
                // One schedule pair per field (keyed by name). B levelGhost passes the field's
                // own MagneticRefinePatchStrategy (div-correction); E/J levelGhost take none.
                // Model B (shared ID): GhostField form — same-level transactions read the
                // shared ID on hybrid peers, the coarse leg reads it on the MHD level where
                // it IS allocated. Includes periodic images (full schedule, unlike the
                // src_level=nullptr InitField form used for temporaries below).
                magModelGhostSchedules_[levelNumber] = magModelGhostAlgo_.createSchedule(
                    level, levelNumber - 1, hierarchy,
                    magStratPerField_.at(modelMagneticKey_).get());
                // Model E/J (shared IDs): same one-pass GhostField form, no patch strategy.
                eModelGhostSchedules_[levelNumber]
                    = eModelGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                currentModelGhostSchedules_[levelNumber]
                    = currentModelGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                for (auto& [key, algo] : magPatchGhostAlgos_)
                    magPatchGhostSchedules_[key][levelNumber] = algo.createSchedule(level, nullptr);
                for (auto& [key, algo] : magLevelGhostAlgos_)
                    magLevelGhostSchedules_[key][levelNumber] = algo.createSchedule(
                        ghostBorderFill, level, nullptr, levelNumber - 1, hierarchy,
                        magStratPerField_.at(key).get());
                for (auto& [key, algo] : ePatchGhostAlgos_)
                    ePatchGhostSchedules_[key][levelNumber] = algo.createSchedule(level, nullptr);
                for (auto& [key, algo] : eLevelGhostAlgos_)
                    eLevelGhostSchedules_[key][levelNumber] = algo.createSchedule(
                        ghostBorderFill, level, nullptr, levelNumber - 1, hierarchy);
                for (auto& [key, algo] : currentPatchGhostAlgos_)
                    currentPatchGhostSchedules_[key][levelNumber]
                        = algo.createSchedule(level, nullptr);
                for (auto& [key, algo] : currentLevelGhostAlgos_)
                    currentLevelGhostSchedules_[key][levelNumber] = algo.createSchedule(
                        ghostBorderFill, level, nullptr, levelNumber - 1, hierarchy);

                // Prim-field refine + postprocessRefine spawn schedules (one per particle set).
                // Domain: interior-only fill → postprocessRefine receives interior boxes only →
                // spawn domainParticles in interior cells exclusively.
                // PatchLevelInteriorFillPattern required: default PatchLevelFullFillPattern
                // would include ghost cells in fine_box → postprocessRefine spawns domainParticles
                // in ghost cells → HybridLevelInitializer::depositParticles stencil OOB crash.
                auto interiorFillPattern
                    = std::make_shared<SAMRAI::xfer::PatchLevelInteriorFillPattern>();
                primDomainSchedules_[levelNumber] = primAlgoDomain_.createSchedule(
                    interiorFillPattern, level, nullptr, levelNumber - 1, hierarchy,
                    spawnStrategyFor_(spawnDispatcherDomain_, spawnStratDomain_));
                // Old/New: ghost-only fill from coarse MHD.
                // PatchLevelBorderFillPattern restricts fill to coarse-fine ghost region →
                // postprocessRefine receives ghost boxes only → spawn levelGhost particles.
                // src_level=nullptr (InitField form) is a CHOICE here, not a constraint:
                // the prim ids are shared with the MHD model, so a same-level src lookup
                // would be legal — but ghost-particle paths stay init-form (respawned
                // fresh each episode), mirroring HybridHybrid where regrid-form ghost
                // copies hit occasional SAMRAI MPI-module failures (PHAREHUB #604).
                auto borderFillPattern
                    = std::make_shared<SAMRAI::xfer::PatchLevelBorderFillPattern>();
                primOldSchedules_[levelNumber] = primAlgoOld_.createSchedule(
                    borderFillPattern, level, nullptr, levelNumber - 1, hierarchy,
                    spawnStrategyFor_(spawnDispatcherOld_, spawnStratOld_));
                primNewSchedules_[levelNumber] = primAlgoNew_.createSchedule(
                    borderFillPattern, level, nullptr, levelNumber - 1, hierarchy,
                    spawnStrategyFor_(spawnDispatcherNew_, spawnStratNew_));

                domainGhostPartRefiners_.registerLevel(hierarchy, level);
                borderComms_.registerLevel(levelNumber, hierarchy);

                // Covered-interior sync schedules (fine hybrid → coarse MHD)
                densitySynchronizers_.registerLevel(hierarchy, level);
                momentumSynchronizers_.registerLevel(hierarchy, level);
                magnetoSynchronizers_.registerLevel(hierarchy, level);
                totalEnergySynchronizers_.registerLevel(hierarchy, level);
            }
        }

        std::unique_ptr<IMessengerInfo> emptyInfoFromCoarser() override
        { return std::make_unique<MHDMessengerInfo>(); }

        std::unique_ptr<IMessengerInfo> emptyInfoFromFiner() override
        { return std::make_unique<HybridMessengerInfo>(); }

        std::string fineModelName() const override { return HybridModel::model_name; }
        std::string coarseModelName() const override { return MHDModel::model_name; }

        virtual ~MHDHybridMessengerStrategy() = default;

        void firstStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& /*hierarchy*/,
                       double const currentTime, double const prevCoarserTime,
                       double const newCoarserTime) override
        {
            if (level.getLevelNumber() == rootLevelNumber)
                return;

            if (newCoarserTime < prevCoarserTime)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy::firstStep: prevCoarserTime ("
                    + std::to_string(prevCoarserTime) + ") should be < newCoarserTime ("
                    + std::to_string(newCoarserTime) + ")");

            int const lvl     = level.getLevelNumber();
            auto& hybridModel = static_cast<HybridModel&>(model);
            auto& ions        = hybridModel.state.ions;

            populateSpawnStrategies_(ions);
            consToPrim_(lvl - 1);
            clearParticleBuckets_(level, model, false, false, true);
            primNewSchedules_.at(lvl)->fillData(currentTime);

            beforePushCoarseTime_[static_cast<std::size_t>(lvl)] = prevCoarserTime;
            afterPushCoarseTime_[static_cast<std::size_t>(lvl)]  = newCoarserTime;
        }

        void lastStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level) override
        {
            if (level.getLevelNumber() == rootLevelNumber)
                return;

            auto& hybridModel = static_cast<HybridModel&>(model);
            auto& ions        = hybridModel.state.ions;
            for (auto& patch : level)
            {
                auto dataOnPatch = resourcesManager_->setOnPatch(*patch, ions);
                for (auto& pop : ions)
                {
                    auto& levelGhostParticlesOld = pop.levelGhostParticlesOld();
                    auto& levelGhostParticlesNew = pop.levelGhostParticlesNew();
                    auto& levelGhostParticles    = pop.levelGhostParticles();

                    std::swap(levelGhostParticlesNew, levelGhostParticlesOld);
                    levelGhostParticlesNew.clear();
                    levelGhostParticles = levelGhostParticlesOld;
                }
            }

            // Staging for the covered-interior sync that follows in
            // standardLevelSynchronization → synchronize(): lastStep fires after the final
            // fine substep and nothing mutates ions/B/Pe before synchronize.
            assembleSyncFields_(hybridModel, level);
        }

        void prepareStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                         double /*currentTime*/) final
        {
        }

        void fillRootGhosts(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                            double const /*initDataTime*/) final
        {
            // Root level is MHD-only; no cross-model root ghost fill needed.
        }

        // initLevel: fill B and E from MHD, then spawn fine particles from refined MHD
        // primitives (prim-field schedules + postprocessRefine, see ParticleSpawnStrategy).
        void initLevel(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       double const initDataTime) override
        {
            int const lvl     = level.getLevelNumber();
            auto& hybridModel = dynamic_cast<HybridModel&>(model);
            magComms_.magInitRefineSchedules_.at(lvl)->fillData(initDataTime);
            eInitComms_.fill(lvl, initDataTime);

            if (lvl != rootLevelNumber)
            {
                auto& ions = hybridModel.state.ions;
                populateSpawnStrategies_(ions);
                consToPrim_(lvl - 1);
                clearParticleBuckets_(level, model, true, true, false);
                primDomainSchedules_.at(lvl)->fillData(initDataTime);
                primOldSchedules_.at(lvl)->fillData(initDataTime);
                copyLevelGhostOldToPushable_(level, model);
            }
        }

        void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                    int const levelNumber,
                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                    IPhysicalModel& model, double const initDataTime) override
        {
            auto const level  = hierarchy->getPatchLevel(levelNumber);
            auto& hybridModel = dynamic_cast<HybridModel&>(model);
            auto& ions        = hybridModel.state.ions;

            // B regrid: single-pass schedule (copy from old fine level where it overlaps,
            // refine from coarse MHD elsewhere) on the SHARED B id — divB-conserving,
            // identical to the HybridHybrid path.
            magComms_.magneticRegriding_(hierarchy, level, oldLevel, initDataTime);

            eInitComms_.fill(levelNumber, initDataTime);

            if (levelNumber != rootLevelNumber)
                regridParticles_(hierarchy, level, oldLevel, model, ions, initDataTime);
        }

        void fillMagneticGhosts(VecFieldT& B, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(B, level, *resourcesManager_);
            int const lvl = level.getLevelNumber();
            // Dispatch by field name: the schedule fills the IDs it was registered with, so we must
            // select the schedule(s) registered for THIS field (e.g. "EM_B" vs "EMPred_B").
            // Model B (shared ID): single GhostField-form pass (peer copy + coarse interp +
            // periodic images). Temporaries: split pair — levelGhost (coarse MHD, BorderFill)
            // first → fills y CF ghosts; patchGhost (same-level) second → exact same-level
            // data wins on overlap.
            if (B.name() == modelMagneticKey_)
            {
                magModelGhostSchedules_.at(lvl)->fillData(fillTime);
            }
            else
            {
                magLevelGhostSchedules_.at(B.name()).at(lvl)->fillData(fillTime);
                magPatchGhostSchedules_.at(B.name()).at(lvl)->fillData(fillTime);
            }
        }

        void fillElectricGhosts(VecFieldT& E, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(E, level, *resourcesManager_);
            int const lvl = level.getLevelNumber();
            if (E.name() == modelElectricKey_)
            {
                eModelGhostSchedules_.at(lvl)->fillData(fillTime);
            }
            else
            {
                eLevelGhostSchedules_.at(E.name()).at(lvl)->fillData(fillTime);
                ePatchGhostSchedules_.at(E.name()).at(lvl)->fillData(fillTime);
            }
        }

        void fillCurrentGhosts(VecFieldT& J, SAMRAI::hier::PatchLevel const& level,
                               double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(J, level, *resourcesManager_);
            int const lvl = level.getLevelNumber();
            if (J.name() == modelCurrentKey_)
            {
                currentModelGhostSchedules_.at(lvl)->fillData(fillTime);
            }
            else
            {
                currentLevelGhostSchedules_.at(J.name()).at(lvl)->fillData(fillTime);
                currentPatchGhostSchedules_.at(J.name()).at(lvl)->fillData(fillTime);
            }
        }

        void fillIonGhostParticles(IonsT& ions, SAMRAI::hier::PatchLevel& level,
                                   double const fillTime) override
        {
            domainGhostPartRefiners_.fill(level.getLevelNumber(), fillTime);
            for (auto patch : resourcesManager_->enumerate(level, ions))
                for (auto& pop : ions)
                    pop.patchGhostParticles().clear();
        }

        void fillIonPopMomentGhosts(IonsT& ions, SAMRAI::hier::PatchLevel& level,
                                    double const afterPushTime) override
        {
            if (level.getLevelNumber() == rootLevelNumber)
                return;

            auto const alpha = timeInterpCoef_(afterPushTime, level.getLevelNumber());
            if (alpha < 0 || alpha > 1)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy::fillIonPopMomentGhosts: invalid alpha = "
                    + std::to_string(alpha) + " on level "
                    + std::to_string(level.getLevelNumber()));

            for (auto const& patch : level)
            {
                auto dataOnPatch = resourcesManager_->setOnPatch(*patch, ions);
                auto layout      = layoutFromPatch<HybridGridLayoutT>(*patch);
                for (auto& pop : ions)
                {
                    auto& levelGhostOld = pop.levelGhostParticlesOld();
                    interpolate_(makeRange(levelGhostOld),
                                 pop.particleDensity(), pop.chargeDensity(), pop.flux(),
                                 layout, 1.0 - alpha);

                    auto& levelGhostNew = pop.levelGhostParticlesNew();
                    interpolate_(makeRange(levelGhostNew),
                                 pop.particleDensity(), pop.chargeDensity(), pop.flux(),
                                 layout, alpha);
                }
            }
        }

        void fillFluxBorders(IonsT& ions, SAMRAI::hier::PatchLevel& level,
                             double const fillTime) override
        { borderComms_.fillFluxBorders(ions, level, sumVec_, fillTime); }

        void fillDensityBorders(IonsT& ions, SAMRAI::hier::PatchLevel& level,
                                double const fillTime) override
        { borderComms_.fillDensityBorders(ions, level, sumField_, fillTime); }

        void fillIonBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& level,
                            double const fillTime) override
        { borderComms_.fillIonBorders(level, fillTime); }

        // Covered-interior overwrite: coarsen end-of-subcycle hybrid data (assembled in
        // lastStep) onto the covered MHD interior — rho/rhoV/Etot via fused ppp→ddd
        // restriction, B via the face-flux-preserving coarsener on the shared id.
        void synchronize(SAMRAI::hier::PatchLevel& level) final
        {
            auto const levelNumber = level.getLevelNumber();
            densitySynchronizers_.sync(levelNumber);
            momentumSynchronizers_.sync(levelNumber);
            magnetoSynchronizers_.sync(levelNumber);
            totalEnergySynchronizers_.sync(levelNumber);
        }

        // reflux: coarsen Hybrid flux sums into MHD flux sums + ghost refill;
        // SolverMHD::reflux then applies the textbook difference (timeFluxes − fluxSum).
        void reflux(int const coarserLevelNumber, int const fineLevelNumber,
                    double const syncTime) override
        { refluxComms_.reflux(fineLevelNumber, coarserLevelNumber, syncTime); }

        void postSynchronize(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                             double const /*time*/) override
        {
            // No-op: SolverMHD::reflux ends with coarse moments+B ghost refills, after both
            // the covered-interior overwrite (synchronize) and the CF-band correction —
            // same reliance as MHD-MHD.
        }

    private:
        std::shared_ptr<RMType> resourcesManager_;
        int const firstLevel_;
        std::shared_ptr<CrossModelFillContext> crossModelContext_; // null only in tests

        // Same-type MHD refine ops for refluxComms_ ghost fills
        std::shared_ptr<MHDERefineOp> mhdERefineOp_{std::make_shared<MHDERefineOp>()};
        std::shared_ptr<MHDMagRefineOp> mhdMagRefineOp_{std::make_shared<MHDMagRefineOp>()};
        std::shared_ptr<MHDMagInitRefineOp> mhdMagInitRefineOp_{
            std::make_shared<MHDMagInitRefineOp>()};
        std::shared_ptr<MHDFluxRefineOp> mhdFluxRefineOp_{std::make_shared<MHDFluxRefineOp>()};
        std::shared_ptr<MHDVecFluxRefineOp> mhdVecFluxRefineOp_{
            std::make_shared<MHDVecFluxRefineOp>()};

        // Fill patterns
        std::shared_ptr<TensorFieldFillPattern_t> nonOverwriteInteriorTFfillPattern_{
            std::make_shared<TensorFieldFillPattern_t>()};
        std::shared_ptr<TensorFieldFillPattern_t> overwriteInteriorTFfillPattern_{
            std::make_shared<TensorFieldFillPattern_t>(true)};

        // B fills: MagneticRefinePatchStrategy provides div-correction after interpolation.
        BfieldComms<RMType, HybVectorFieldDataT> magComms_;

        // E fills
        EfieldComms eInitComms_;

        // Reflux: 4 channels (E, HydroX, HydroY, HydroZ), cross-type coarsen + MHD ghost refill
        MHDHybridRefluxComms<MHDModel, HybridModel> refluxComms_;

        // Ghost fill algos + schedule maps (B/E/J ghost fills), keyed by FIELD NAME.
        // A RefineSchedule fills the patch-data IDs its algorithm was registered with — NOT the
        // VecField passed to fill*Ghosts. The solver calls fill*Ghosts on several distinct fields
        // per step (model B "EM_B" + predictor B "EMPred_B"; model E "EM_E" + avg E "EMAvg_E";
        // model J). Each needs its OWN schedule pair, dispatched by vec.name() in fill*Ghosts —
        // mirrors HybridHybrid's per-ghostMagnetic-entry refiners (refiner_pool.hpp:132). The
        // field-name list comes from hybridInfo->ghost{Magnetic,Electric,Current} (model fields
        // pushed by HybridModel, temporaries appended by SolverPPC::fillMessengerInfo).
        //
        // Each field gets two algos: patchGhost (Hybrid peer exchange, src=field id, nullptr op →
        // no coarse→fine) and levelGhost (MHD→Hybrid, src=mhd_*_id, InitField+FullFillPattern). A
        // single GhostField-form algo with src=mhd_*_id would generate same-level copy transactions
        // that try to read mhd_*_id from Hybrid patches (not allocated) → crash.
        using RefineAlgoMap = std::map<std::string, SAMRAI::xfer::RefineAlgorithm>;
        using RefineSchedMap
            = std::map<std::string, std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>>;
        using MagStratT = MagneticRefinePatchStrategy<RMType, HybVectorFieldDataT>;

        RefineAlgoMap magPatchGhostAlgos_;
        RefineAlgoMap magLevelGhostAlgos_;
        RefineSchedMap magPatchGhostSchedules_;
        RefineSchedMap magLevelGhostSchedules_;
        // Model B is SHARED with MHD (same patchdata ID on all levels, see shareResources),
        // so its ghost fill collapses to ONE GhostField-form schedule: same-level peer copy
        // + coarse→fine interpolation in a single pass (the coarse leg reads the shared ID
        // on the MHD level, which IS allocated there — the split-pair workaround only exists
        // for the strategy temporaries, which live on the hybrid level alone).
        SAMRAI::xfer::RefineAlgorithm magModelGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> magModelGhostSchedules_;
        std::string modelMagneticKey_;
        // One MagneticRefinePatchStrategy per B field (each registers its own id for div-correction)
        std::map<std::string, std::shared_ptr<MagStratT>> magStratPerField_;
        RefineAlgoMap ePatchGhostAlgos_;
        RefineAlgoMap eLevelGhostAlgos_;
        RefineSchedMap ePatchGhostSchedules_;
        RefineSchedMap eLevelGhostSchedules_;
        // Model E/J are SHARED with MHD (same patchdata ID, see shareResources) — same
        // one-pass GhostField collapse as model B above, no patch strategy needed.
        SAMRAI::xfer::RefineAlgorithm eModelGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> eModelGhostSchedules_;
        std::string modelElectricKey_;
        RefineAlgoMap currentPatchGhostAlgos_;
        RefineAlgoMap currentLevelGhostAlgos_;
        RefineSchedMap currentPatchGhostSchedules_;
        RefineSchedMap currentLevelGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm currentModelGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> currentModelGhostSchedules_;
        std::string modelCurrentKey_;

        // MHD conservative field IDs + heat capacity ratio (consumed by consToPrim_)
        int mhdRhoId_  = -1;
        int mhdRhoVId_ = -1;
        int mhdBId_    = -1;
        int mhdEtotId_ = -1;
        double gamma_  = 5.0 / 3.0;

        // Hierarchy reference for consToPrim_
        std::weak_ptr<SAMRAI::hier::PatchHierarchy> hierarchy_;

        // Time-interpolation brackets (set in firstStep, used in fillIonPopMomentGhosts)
        std::unordered_map<std::size_t, double> beforePushCoarseTime_;
        std::unordered_map<std::size_t, double> afterPushCoarseTime_;

        // Same-level escaped particle handling (mirrors HybridHybrid domainGhostPartRefiners_)
        DomainGhostPartRefinerPool domainGhostPartRefiners_{resourcesManager_};

        // Border fills: intra-Hybrid MPI accumulation (mirrors HybridHybrid borderComms_)
        HybridBorderComms<HybridModel> borderComms_{resourcesManager_};

        // Scratch fields for border fill sum operations
        VecFieldT sumVec_{"MHDHybrid_sumVec",    core::PhysicalQuantity::Vector::Hyb_V};
        FieldT    sumField_{"MHDHybrid_sumField", core::PhysicalQuantity::Scalar::Hyb_rho};

        // Covered-interior sync staging (fine hybrid levels, MHD ddd quantities): assembled
        // at fine cell centers in lastStep by corner-averaging primal-node integrands, then
        // cell-average coarsened onto the MHD model fields in synchronize(). The staging
        // must share the dst centering — SAMRAI CoarsenSchedule coarsens src into a
        // src-typed temporary coarse level, so cross-centering channels cannot work (see
        // MHDFieldCoarsenOp alias comment). B uses the shared id src==dst.
        FieldT    syncRho_{"MHDHybrid_syncRho",   core::PhysicalQuantity::Scalar::MHD_rho};
        VecFieldT syncRhoV_{"MHDHybrid_syncRhoV", core::PhysicalQuantity::Vector::MHD_rhoV};
        FieldT    syncEtot_{"MHDHybrid_syncEtot", core::PhysicalQuantity::Scalar::MHD_Etot};

        CoarsenOp_ptr mhdFieldCoarsenOp_{std::make_shared<MHDFieldCoarsenOp>()};
        CoarsenOp_ptr mhdVecFieldCoarsenOp_{std::make_shared<MHDVecFieldCoarsenOp>()};
        CoarsenOp_ptr magneticFieldCoarsenOp_{std::make_shared<MagneticFieldCoarsenOp>()};
        SynchronizerPool<RMType> densitySynchronizers_{resourcesManager_};
        SynchronizerPool<RMType> momentumSynchronizers_{resourcesManager_};
        SynchronizerPool<RMType> magnetoSynchronizers_{resourcesManager_};
        SynchronizerPool<RMType> totalEnergySynchronizers_{resourcesManager_};

        // Hybrid diagnostics ModelView temporaries (PHARE_sum*): in hybrid-only runs the
        // HybridHybrid strategy registers them; with MHD below there may be no such strategy,
        // so this one must. registerResources is keyed by name — both registering is fine.
        FieldT       diagSumField_{"PHARE_sumField", core::PhysicalQuantity::Scalar::Hyb_rho};
        VecFieldT    diagSumVec_{"PHARE_sumVec", core::PhysicalQuantity::Vector::Hyb_V};
        TensorFieldT diagSumTensor_{"PHARE_sumTensor", core::PhysicalQuantity::Tensor::M};

        // Fine strategy-owned primitive fields: coarse MHD (rho,V,P) refined here, then
        // postprocessRefine reads them to spawn Maxwellian particles.
        FieldT    primRho_{"MHDHybMess_prim_rho", core::PhysicalQuantity::Scalar::MHD_rho};
        VecFieldT primV_  {"MHDHybMess_prim_V",   core::PhysicalQuantity::Vector::MHD_V};
        FieldT    primP_  {"MHDHybMess_prim_P",   core::PhysicalQuantity::Scalar::MHD_P};

        std::shared_ptr<MHDScalarPrimRefineOp> mhdScalarPrimRefineOp_{
            std::make_shared<MHDScalarPrimRefineOp>()};
        std::shared_ptr<MHDVecPrimRefineOp> mhdVecPrimRefineOp_{
            std::make_shared<MHDVecPrimRefineOp>()};

        SAMRAI::xfer::RefineAlgorithm primAlgoDomain_, primAlgoOld_, primAlgoNew_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>
            primDomainSchedules_, primOldSchedules_, primNewSchedules_;

        // Regrid-only: null-op (copy-only) registrations of the domain particle ids,
        // scheduled against the old fine level so surviving regions keep their PIC history.
        SAMRAI::xfer::RefineAlgorithm partRegridAlgo_;

        ParticleSpawnStrategy spawnStratDomain_, spawnStratOld_, spawnStratNew_;

        // Nested-frame spawn fragments (one per population, Domain bucket, no interior
        // clip) — owned here, handed to the context by pop name in populateSpawnStrategies_;
        // the hybrid pools' dispatchers resolve them at MHD_Hyb interp frames. Instances
        // are reused across populate calls (initLevel/firstStep/regrid reconfigure them).
        std::map<std::string, std::shared_ptr<ParticleSpawnStrategy>> nestedSpawnFragments_;

        // Case-A dispatchers: the spawn schedules recurse through coarse-interp frames
        // that are pure-MHD pairs when levelNumber - 1 > firstLevel_; the schedule-bound
        // strategy is inherited into every frame, so it must no-op there (the raw spawn
        // postprocessRefine null-derefs on MHD interp patches — no particle ids). At the
        // top (coupled) frame the pair is MHD_Hyb and the raw strategy runs as before.
        // Stencil width 0 = the raw strategy's width. Only passed to createSchedule when
        // crossModelContext_ is set (null only in tests → raw strategies, today's path).
        DispatchingRefinePatchStrategy spawnDispatcherDomain_{
            crossModelContext_,
            [this](PairKind k) -> SAMRAI::xfer::RefinePatchStrategy* {
                return k == PairKind::MHD_Hyb ? &spawnStratDomain_ : nullptr;
            },
            SAMRAI::hier::IntVector::getZero(SAMRAI::tbox::Dimension{dimension})};
        DispatchingRefinePatchStrategy spawnDispatcherOld_{
            crossModelContext_,
            [this](PairKind k) -> SAMRAI::xfer::RefinePatchStrategy* {
                return k == PairKind::MHD_Hyb ? &spawnStratOld_ : nullptr;
            },
            SAMRAI::hier::IntVector::getZero(SAMRAI::tbox::Dimension{dimension})};
        DispatchingRefinePatchStrategy spawnDispatcherNew_{
            crossModelContext_,
            [this](PairKind k) -> SAMRAI::xfer::RefinePatchStrategy* {
                return k == PairKind::MHD_Hyb ? &spawnStratNew_ : nullptr;
            },
            SAMRAI::hier::IntVector::getZero(SAMRAI::tbox::Dimension{dimension})};

        SAMRAI::xfer::RefinePatchStrategy*
        spawnStrategyFor_(DispatchingRefinePatchStrategy& dispatcher, ParticleSpawnStrategy& raw)
        {
            if (crossModelContext_)
                return &dispatcher;
            return &raw;
        }

        int primRhoId_ = -1, primVId_ = -1, primPId_ = -1;
        int mhdVId_ = -1, mhdPId_ = -1;
        int modelEId_ = -1; // shared MHD/Hybrid E id (checked equal in registerGhostComms_)
        // hybrid solver ghost temporaries that cross the boundary at depth (Bpred, Eavg):
        // their VALUES are read by copy transactions on MHD patches → presence + mirror
        std::vector<int> crossMagTempIds_, crossElecTempIds_;
        std::vector<int> domainPartFineIds_;
        std::vector<int> lvlGhostOldFineIds_;
        std::vector<int> lvlGhostNewFineIds_;

        // Particle-to-moment interpolator
        core::Interpolator<dimension, interpOrder> interpolate_;

        // Convert MHD conservatives (rho, rhoV, B, Etot) → primitives (V, P) in-place on the
        // coarse level before each prim-refine schedule. B stays on the coarse model only.
        void consToPrim_(int const coarseLevelNumber)
        {
            auto hierarchy = hierarchy_.lock();
            if (!hierarchy)
                throw std::runtime_error("MHDHybridMessengerStrategy::consToPrim_: hierarchy expired");

            auto const coarseLevel = hierarchy->getPatchLevel(coarseLevelNumber);
            for (auto const& patch : *coarseLevel)
            {
                auto const layout = layoutFromPatch<MHDGridLayoutT>(*patch);
                core::ToPrimitiveConverter_ref<MHDGridLayoutT> toPrim{layout};
                auto& rho  = MHDFieldDataT::getField(*patch, mhdRhoId_);
                auto  rhoV = MHDVecFieldDataT::getTensorField(*patch, mhdRhoVId_);
                auto  B    = MHDVecFieldDataT::getTensorField(*patch, mhdBId_);
                auto& Etot = MHDFieldDataT::getField(*patch, mhdEtotId_);
                auto  V    = MHDVecFieldDataT::getTensorField(*patch, mhdVId_);
                auto& P    = MHDFieldDataT::getField(*patch, mhdPId_);
                toPrim(gamma_, rho, rhoV, B, Etot, V, P);
            }
        }

        void registerGhostComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo,
                                  std::unique_ptr<HybridMessengerInfo> const& hybridInfo)
        {
            auto mhd_b_id    = resourcesManager_->getID(mhdInfo->modelMagnetic);
            auto mhd_e_id    = resourcesManager_->getID(mhdInfo->modelElectric);
            auto mhd_j_id    = resourcesManager_->getID(mhdInfo->modelCurrent);
            auto mhd_rho_id  = resourcesManager_->getID(mhdInfo->modelDensity);
            auto mhd_rhoV_id = resourcesManager_->getID(mhdInfo->modelMomentum);
            auto mhd_Etot_id = resourcesManager_->getID(mhdInfo->modelTotalEnergy);
            auto hyb_b_id    = resourcesManager_->getID(hybridInfo->modelMagnetic);
            auto hyb_e_id    = resourcesManager_->getID(hybridInfo->modelElectric);
            auto hyb_j_id    = resourcesManager_->getID(hybridInfo->modelCurrent);
            if (!mhd_b_id or !mhd_e_id or !mhd_j_id or !mhd_rho_id or !mhd_rhoV_id
                or !mhd_Etot_id or !hyb_b_id or !hyb_e_id or !hyb_j_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing field IDs in registerGhostComms_");

            // B/E/J ghost fills split into two algos each, registered PER FIELD NAME (the solver
            // fills several distinct fields per step; each needs its own schedule pair — see member
            // decl comment). Field-name lists come from hybridInfo->ghost{Magnetic,Electric,Current}.
            //   sameLvl (patchGhost):    src=field id, nullptr op (no coarse→fine), GhostField form.
            //                            Fills patch ghost cells from neighboring Hybrid peers.
            //   coarseToFine (levelGhost): src=mhd_*_id, mhd*RefineOp_, InitField+FullFillPattern.
            //                            Fills level ghost cells by interpolation from coarser MHD.
            // scratch==dst throughout (SAMRAI-legal per RefineAlgorithm.h:62–74).
            modelMagneticKey_ = hybridInfo->modelMagnetic;
            crossMagTempIds_.clear();
            crossElecTempIds_.clear();
            for (auto const& key : hybridInfo->ghostMagnetic)
            {
                auto id = resourcesManager_->getID(key);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing B ghost field ID for " + key);
                auto strat = std::make_shared<MagStratT>(*resourcesManager_);
                strat->registerIDs(*id);
                magStratPerField_[key] = strat;
                if (key == modelMagneticKey_)
                {
                    // shared ID: one same-id algo serves both legs (peer copy on the fine
                    // level + interpolation from the coarse MHD level)
                    magModelGhostAlgo_.registerRefine(*id, *id, *id, mhdMagRefineOp_,
                                                      nonOverwriteInteriorTFfillPattern_);
                }
                else
                {
                    crossMagTempIds_.push_back(*id);
                    magPatchGhostAlgos_[key].registerRefine(*id, *id, *id, nullptr,
                                                            nonOverwriteInteriorTFfillPattern_);
                    magLevelGhostAlgos_[key].registerRefine(*id, *mhd_b_id, *id, mhdMagRefineOp_,
                                                            nonOverwriteInteriorTFfillPattern_);
                }
            }
            // Model E/J: shared patchdata ID with MHD (shareResources coupled path), same
            // one-pass GhostField collapse as model B. Temporaries keep the split pair.
            if (*mhd_e_id != *hyb_e_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: E is not shared between MHD and Hybrid "
                    "(shareResources not called?)");
            if (*mhd_j_id != *hyb_j_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: J is not shared between MHD and Hybrid "
                    "(shareResources not called?)");
            modelElectricKey_ = hybridInfo->modelElectric;
            modelCurrentKey_  = hybridInfo->modelCurrent;
            modelEId_         = *mhd_e_id;
            for (auto const& key : hybridInfo->ghostElectric)
            {
                auto id = resourcesManager_->getID(key);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing E ghost field ID for " + key);
                if (key == modelElectricKey_)
                {
                    eModelGhostAlgo_.registerRefine(*id, *id, *id, mhdERefineOp_,
                                                    nonOverwriteInteriorTFfillPattern_);
                }
                else
                {
                    crossElecTempIds_.push_back(*id);
                    ePatchGhostAlgos_[key].registerRefine(*id, *id, *id, nullptr,
                                                          nonOverwriteInteriorTFfillPattern_);
                    eLevelGhostAlgos_[key].registerRefine(*id, *mhd_e_id, *id, mhdERefineOp_,
                                                          nonOverwriteInteriorTFfillPattern_);
                }
            }
            for (auto const& key : hybridInfo->ghostCurrent)
            {
                auto id = resourcesManager_->getID(key);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing J ghost field ID for " + key);
                if (key == modelCurrentKey_)
                {
                    currentModelGhostAlgo_.registerRefine(*id, *id, *id, mhdERefineOp_,
                                                          nonOverwriteInteriorTFfillPattern_);
                }
                else
                {
                    currentPatchGhostAlgos_[key].registerRefine(*id, *id, *id, nullptr,
                                                                nonOverwriteInteriorTFfillPattern_);
                    currentLevelGhostAlgos_[key].registerRefine(*id, *mhd_j_id, *id,
                                                                mhdERefineOp_,
                                                                nonOverwriteInteriorTFfillPattern_);
                }
            }

            // Store fine hybrid ghost particle IDs for postprocessRefine spawn
            lvlGhostOldFineIds_.clear();
            for (auto const& name : hybridInfo->levelGhostParticlesOld)
            {
                auto id = resourcesManager_->getID(name);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing lvlGhostOld ID for " + name);
                lvlGhostOldFineIds_.push_back(*id);
            }
            lvlGhostNewFineIds_.clear();
            for (auto const& name : hybridInfo->levelGhostParticlesNew)
            {
                auto id = resourcesManager_->getID(name);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing lvlGhostNew ID for " + name);
                lvlGhostNewFineIds_.push_back(*id);
            }

            // Same-level escaped particle handling (mirrors HybridHybrid domainGhostPartRefiners_)
            domainGhostPartRefiners_.addStaticRefiners(
                hybridInfo->patchGhostParticles, nullptr, hybridInfo->patchGhostParticles,
                std::make_shared<ParticleDomainFromGhostFillPattern<HybridGridLayoutT>>());

            // Border fills: intra-Hybrid MPI accumulation (mirrors HybridHybrid borderComms_)
            borderComms_.registerInfo(*hybridInfo, sumVec_.name(), sumField_.name());
        }

        void registerInitComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo,
                                 std::unique_ptr<HybridMessengerInfo> const& hybridInfo)
        {
            auto mhd_rho_id  = resourcesManager_->getID(mhdInfo->modelDensity);
            auto mhd_rhoV_id = resourcesManager_->getID(mhdInfo->modelMomentum);
            auto mhd_B_id    = resourcesManager_->getID(mhdInfo->modelMagnetic);
            auto mhd_Etot_id = resourcesManager_->getID(mhdInfo->modelTotalEnergy);
            auto mhd_e_id    = resourcesManager_->getID(mhdInfo->modelElectric);
            if (!mhd_rho_id or !mhd_rhoV_id or !mhd_B_id or !mhd_Etot_id or !mhd_e_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing MHD IDs in registerInitComms_");

            // B init/regrid: the model B patchdata ID is SHARED between MHD and Hybrid
            // (shareResources in the simulator coupled path) — same-id registrations, exactly
            // the HybridHybrid pattern. The init refiner runs coarse→fine on the one ID; the
            // regrid algo copies old fine data where the old level overlaps and refines from
            // the coarse MHD level elsewhere (MagneticFieldRefiner skips already-copied,
            // non-NaN faces), divB-conserving by construction.
            auto hyb_b_id = resourcesManager_->getID(hybridInfo->modelMagnetic);
            if (!hyb_b_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing Hybrid B ID in registerInitComms_");
            if (*mhd_B_id != *hyb_b_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: B is not shared between MHD and Hybrid "
                    "(shareResources not called?)");
            magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            magComms_.BalgoInit.registerRefine(*hyb_b_id, *hyb_b_id, *hyb_b_id,
                                               mhdMagInitRefineOp_,
                                               overwriteInteriorTFfillPattern_);
            magComms_.BregridAlgo.registerRefine(*hyb_b_id, *hyb_b_id, *hyb_b_id,
                                                 mhdMagRefineOp_,
                                                 overwriteInteriorTFfillPattern_);

            // E init fills: model E is shared with MHD (invariant checked in
            // registerGhostComms_) → same-id registration, scratch==dst.
            for (auto const& eName : hybridInfo->initElectric)
            {
                auto hyb_e_id = resourcesManager_->getID(eName);
                if (!hyb_e_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing Hybrid E ID for " + eName);
                eInitComms_.algo.registerRefine(*hyb_e_id, *hyb_e_id, *hyb_e_id,
                                                mhdERefineOp_, nullptr);
            }

            // Cache MHD conservative IDs + EOS gamma for consToPrim_
            mhdRhoId_  = *mhd_rho_id;
            mhdRhoVId_ = *mhd_rhoV_id;
            mhdBId_    = *mhd_B_id;
            mhdEtotId_ = *mhd_Etot_id;

            // Cache MHD primitive IDs (V and P) for consToPrim_
            auto mhd_V_id = resourcesManager_->getID(mhdInfo->modelVelocity);
            auto mhd_P_id = resourcesManager_->getID(mhdInfo->modelPressure);
            if (!mhd_V_id or !mhd_P_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing MHD primitive IDs in registerInitComms_");
            mhdVId_ = *mhd_V_id;
            mhdPId_ = *mhd_P_id;

            // Prim fields are ALIASES of the coarse MHD model's rho/V/P (model = primary):
            // one patchdata id per prim, allocated on MHD levels by the model and on hybrid
            // levels by this messenger's allocate() (through the alias views). This id
            // coverage (coarse + old fine + new fine) is what makes the same-id regrid-form
            // prim schedule in regridParticles_ legal — exactly the phase-7 shared-B/E/J
            // move (see magneticRegriding_). Quantities match by construction (MHD_rho/
            // MHD_V/MHD_P on both sides; the name-based shareResources does not check).
            resourcesManager_->shareResources(mhdInfo->modelDensity, primRho_);
            resourcesManager_->shareResources(mhdInfo->modelVelocity, primV_);
            resourcesManager_->shareResources(mhdInfo->modelPressure, primP_);
            primRhoId_ = mhdRhoId_; // shared ids: aliases resolve to the model ids
            primVId_   = mhdVId_;
            primPId_   = mhdPId_;
            gamma_     = getHeatCapacityRatio_();

            // Wire spawn strategy field IDs (same prim fields for all three sets)
            spawnStratDomain_.setFieldIds(primRhoId_, primVId_, primPId_);
            spawnStratOld_.setFieldIds(primRhoId_, primVId_, primPId_);
            spawnStratNew_.setFieldIds(primRhoId_, primVId_, primPId_);

            // Same-id registrations (src == dst == scratch on the shared prim ids) in all
            // three algos: the interp leg reads the id on the coarse MHD level (model
            // allocation, consToPrim_-filled), the regrid copy leg on old hybrid patches
            // (messenger allocation) — no cross-id single-src-slot conflation possible.
            for (auto* algo : {&primAlgoDomain_, &primAlgoOld_, &primAlgoNew_})
            {
                algo->registerRefine(primRhoId_, primRhoId_, primRhoId_, mhdScalarPrimRefineOp_);
                algo->registerRefine(primVId_,   primVId_,   primVId_,   mhdVecPrimRefineOp_);
                algo->registerRefine(primPId_,   primPId_,   primPId_,   mhdScalarPrimRefineOp_);
            }

            // Store fine hybrid interior particle IDs for postprocessRefine domain spawn
            domainPartFineIds_.clear();
            for (auto const& name : hybridInfo->interiorParticles)
            {
                auto id = resourcesManager_->getID(name);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing interior particle ID for " + name);
                domainPartFineIds_.push_back(*id);
            }

            // Copy-only (null op) registrations for the regrid old-level particle copy:
            // ParticlesData::copy appends old domain particles into new patches where the
            // old level overlaps; no refine items → never fills from the MHD coarse level.
            for (auto const id : domainPartFineIds_)
                partRegridAlgo_.registerRefine(id, id, id, nullptr);
        }

        // Cross-boundary symmetric presence + EM value mirror (context slots; this
        // messenger is the sole writer). HybridHybrid schedules above the boundary
        // recurse into frames whose coarse side is an MHD level; their copy
        // transactions look up hybrid-only ids there:
        //   - per-population ParticlesData → allocated EMPTY on MHD patches (never
        //     read for values; presence stops the unchecked getPatchData null-deref)
        //   - Bpred/Eavg ghost temporaries → VALUES are read, so presence alone would
        //     expose NaN-init fields; the mirror keeps them equal to model B/E on MHD
        //     patches (shared ids, MHD-evolved)
        // Symmetrically, coupled prim schedules at depth reach hybrid levels above
        // firstHybridLevel where the shared prim rho/V/P ids are not allocated (the
        // coupled messenger's allocate() only covers its own level).
        void registerCrossModelSlots_()
        {
            auto const allocateIfAbsent
                = [](SAMRAI::hier::Patch& patch, int const id, double const t) {
                      if (!patch.checkAllocated(id))
                          patch.allocatePatchData(id, t);
                  };

            std::vector<int> particleIds = domainPartFineIds_;
            for (auto const* ids : {&lvlGhostOldFineIds_, &lvlGhostNewFineIds_})
                for (auto const id : *ids)
                    if (std::find(particleIds.begin(), particleIds.end(), id)
                        == particleIds.end())
                        particleIds.push_back(id);

            std::vector<int> emTempIds = crossMagTempIds_;
            emTempIds.insert(emTempIds.end(), crossElecTempIds_.begin(),
                             crossElecTempIds_.end());

            crossModelContext_->addMHDLevelPresence(
                [allocateIfAbsent, particleIds,
                 emTempIds](SAMRAI::hier::Patch& patch, double const t) {
                    for (auto const id : particleIds)
                        allocateIfAbsent(patch, id, t);
                    for (auto const id : emTempIds)
                        allocateIfAbsent(patch, id, t);
                });

            crossModelContext_->addHybridLevelPresence(
                [allocateIfAbsent, ids = std::array{primRhoId_, primVId_, primPId_}](
                    SAMRAI::hier::Patch& patch, double const t) {
                    for (auto const id : ids)
                        allocateIfAbsent(patch, id, t);
                });

            // TensorFieldData::copy — same concrete type on both sides (shared B/E are
            // hybrid-primary, simulator.hpp:349-351; Bpred/Eavg registered by the hybrid
            // solver), quantity asserted inside copy
            crossModelContext_->setMHDElectromagMirror(
                [this](SAMRAI::hier::Patch& patch, double const /*time*/) {
                    auto const mirror = [&](int const srcId, std::vector<int> const& dstIds) {
                        auto src = patch.getPatchData(srcId);
                        if (!src)
                            throw std::runtime_error("MHDHybridMessengerStrategy: EM mirror "
                                                     "source id not allocated on MHD patch");
                        for (auto const dstId : dstIds)
                        {
                            auto dst = patch.getPatchData(dstId);
                            if (!dst)
                                throw std::runtime_error(
                                    "MHDHybridMessengerStrategy: EM mirror destination id "
                                    "not allocated on MHD patch");
                            dst->copy(*src);
                        }
                    };
                    mirror(mhdBId_, crossMagTempIds_);
                    mirror(modelEId_, crossElecTempIds_);
                });
        }

        // Covered-interior sync channels: each coarse sync overwrites the covered MHD
        // interior with conservatively coarsened hybrid data. Moment channels are dst≠src
        // but same-centering (ddd staging assembled in lastStep → MHD ddd conservatives,
        // cell-average restriction); B is src==dst on the shared id with the
        // face-flux-preserving coarsener — MHD-MHD verbatim (the operator keeping
        // MHD-MHD divB-clean).
        void registerSyncComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo)
        {
            densitySynchronizers_.add(mhdInfo->modelDensity, syncRho_.name(),
                                      mhdFieldCoarsenOp_, mhdInfo->modelDensity);
            momentumSynchronizers_.add(mhdInfo->modelMomentum, syncRhoV_.name(),
                                       mhdVecFieldCoarsenOp_, mhdInfo->modelMomentum);
            totalEnergySynchronizers_.add(mhdInfo->modelTotalEnergy, syncEtot_.name(),
                                          mhdFieldCoarsenOp_, mhdInfo->modelTotalEnergy);
            magnetoSynchronizers_.add(mhdInfo->modelMagnetic, magneticFieldCoarsenOp_,
                                      mhdInfo->modelMagnetic);
        }

        // Assemble the ddd staging fields for the covered-interior sync, at fine cell
        // centers:
        //   rho:  fullPrimalToCellCenter projection of massDensity
        //   rhoV: corner average of massDensity × bulkVel (product BEFORE projection —
        //         conservative; project() is linear in one field so products must be
        //         corner-looped, reusing the fullPrimalToCellCenter weights)
        //   Etot: corner average of ½tr(M) + Pe/(γ−1), plus ½|B|²_center with B
        //         projected face→center (faceXToCellCenter — the MHD-side projection,
        //         consistent with how MHD consToPrim reads Etot back)
        // ½tr(M) is the exact ion kinetic+thermal energy density (second moment of f) —
        // no Pi_iso decomposition, anisotropy-safe; Pe/(γ−1) so MHD consToPrim recovers
        // total pressure. Center staging then cell-average coarsening composes to the
        // trapezoid (¼,½,¼) primal→dual restriction per direction — the fused form of
        // which SAMRAI cannot host (see MHDFieldCoarsenOp alias comment). Assembly runs
        // on the physical box only: corners and faces of interior cells are interior
        // nodes, and only covered coarse cells (fine interior) consume the result.
        // Momentum-tensor border nodes carry partial sums (pre-existing) → syncEtot_
        // slightly off at fine patch borders — recorded limitation.
        void assembleSyncFields_(HybridModel& hybridModel, SAMRAI::hier::PatchLevel& level)
        {
            auto& ions       = hybridModel.state.ions;
            auto& electromag = hybridModel.state.electromag;
            auto& electrons  = hybridModel.state.electrons;

            auto constexpr corners = HybridGridLayoutT::fullPrimalToCellCenter();

            for (auto& patch : level)
            {
                auto _ = resourcesManager_->setOnPatch(*patch, ions, electromag, electrons,
                                                       syncRho_, syncRhoV_, syncEtot_);
                auto const layout = layoutFromPatch<HybridGridLayoutT>(*patch);

                auto const& rho = ions.massDensity();
                auto const& V   = ions.velocity();
                auto const& Vx  = V(core::Component::X);
                auto const& Vy  = V(core::Component::Y);
                auto const& Vz  = V(core::Component::Z);
                auto const& MT  = ions.momentumTensor();
                auto const& Mxx = MT(core::Component::XX);
                auto const& Myy = MT(core::Component::YY);
                auto const& Mzz = MT(core::Component::ZZ);
                auto const& Pe  = electrons.pressure();
                auto const& B   = electromag.B;
                auto const& Bx  = B(core::Component::X);
                auto const& By  = B(core::Component::Y);
                auto const& Bz  = B(core::Component::Z);

                auto& rhoVx = syncRhoV_(core::Component::X);
                auto& rhoVy = syncRhoV_(core::Component::Y);
                auto& rhoVz = syncRhoV_(core::Component::Z);

                auto const at = [](auto const& f, auto const& p) {
                    if constexpr (dimension == 1)
                        return f(p[0]);
                    else if constexpr (dimension == 2)
                        return f(p[0], p[1]);
                    else
                        return f(p[0], p[1], p[2]);
                };

                layout.evalOnBox(syncRho_, [&](auto const&... args) mutable {
                    core::MeshIndex<dimension> const cell{args...};

                    double rhoVx_c{0.}, rhoVy_c{0.}, rhoVz_c{0.}, eth_c{0.};
                    for (auto const& wp : corners)
                    {
                        auto p = cell;
                        for (auto i = 0u; i < dimension; ++i)
                            p[i] += wp.indexes[i];

                        auto const rho_p = at(rho, p);
                        rhoVx_c += wp.coef * rho_p * at(Vx, p);
                        rhoVy_c += wp.coef * rho_p * at(Vy, p);
                        rhoVz_c += wp.coef * rho_p * at(Vz, p);
                        eth_c += wp.coef
                                 * (0.5 * (at(Mxx, p) + at(Myy, p) + at(Mzz, p))
                                    + at(Pe, p) / (gamma_ - 1.0));
                    }

                    auto const Bx_c = HybridGridLayoutT::project(
                        Bx, cell, HybridGridLayoutT::faceXToCellCenter());
                    auto const By_c = HybridGridLayoutT::project(
                        By, cell, HybridGridLayoutT::faceYToCellCenter());
                    auto const Bz_c = HybridGridLayoutT::project(
                        Bz, cell, HybridGridLayoutT::faceZToCellCenter());

                    syncRho_(args...)  = HybridGridLayoutT::project(rho, cell, corners);
                    rhoVx(args...)     = rhoVx_c;
                    rhoVy(args...)     = rhoVy_c;
                    rhoVz(args...)     = rhoVz_c;
                    syncEtot_(args...) = eth_c
                                         + 0.5 * (Bx_c * Bx_c + By_c * By_c + Bz_c * Bz_c);
                });
            }
        }

        // Particle-preserving regrid. Copy evolved PIC particles from the old fine level
        // where it overlaps the new one; spawn fresh Maxwellians from refined MHD prims
        // ONLY on genuinely new regions. The restriction is SAMRAI-native: the prim
        // schedule is regrid-form (src_level = oldLevel), and SAMRAI invokes refine ops
        // and postprocessRefine exclusively on its internal unfilled-box set = destination
        // minus oldLevel coverage (RefineSchedule::generateCommunicationSchedule computes
        // it via removeIntersections, RefineSchedule.cpp:3314-3435; refineScratchData
        // passes only coarse_to_unfilled boxes to ops and postprocessRefineBoxes,
        // RefineSchedule.cpp:2737-2778). No geometry arithmetic in PHARE.
        // Legality: prim ids are shared with the MHD model (see registerInitComms_), so
        // both schedule legs find the src id allocated (copy leg on old hybrid patches —
        // messenger allocate(); interp leg on coarse — model allocate()).
        // Behavior delta (accepted, by design): the copy leg also copies old prim FIELD
        // values on overlap regions — inert, no spawn happens there.
        // Ghost particles (Old) are respawned fresh on the whole new border, init-form —
        // same choice as HybridHybrid (regrid-form ghost copy hit SAMRAI MPI failures,
        // PHAREHUB #604).
        void regridParticles_(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                              std::shared_ptr<SAMRAI::hier::PatchLevel> const& level,
                              std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                              IPhysicalModel& model, IonsT const& ions,
                              double const initDataTime)
        {
            auto const levelNumber = level->getLevelNumber();

            populateSpawnStrategies_(ions);
            consToPrim_(levelNumber - 1);
            clearParticleBuckets_(*level, model, true, true, true);

            // Domain particles: copy from the old fine level where it overlaps the new
            // one (copy-only schedule — partRegridAlgo_ has null-op registrations), so
            // surviving regions keep their evolved PIC distributions across regrid.
            // PatchLevelInteriorFillPattern is REQUIRED (same as HybridHybrid's
            // InitInteriorPart regrid): the default full pattern includes dst ghost
            // regions, and ParticlesData::copy_'s DomainToGhosts leg would append
            // ghost-layer particles into domainParticles → duplicated mass across patch
            // seams + Updater::outsideGhostBox aborts after the first push.
            partRegridAlgo_
                .createSchedule(
                    std::make_shared<SAMRAI::xfer::PatchLevelInteriorFillPattern>(), level,
                    oldLevel)
                ->fillData(initDataTime);

            // Maxwellian spawn on new regions only: regrid-form schedule (src=oldLevel);
            // postprocessRefine (the spawn) fires only where the old level did NOT cover
            // — see doc block above. InteriorFillPattern as in initLevel (no ghost spawn).
            primAlgoDomain_
                .createSchedule(
                    std::make_shared<SAMRAI::xfer::PatchLevelInteriorFillPattern>(), level,
                    oldLevel, levelNumber - 1, hierarchy,
                    spawnStrategyFor_(spawnDispatcherDomain_, spawnStratDomain_))
                ->fillData(initDataTime);

            // levelGhostOld respawned fresh on the whole new border (HybridHybrid also
            // refills it with fill(), not regrid()).
            primOldSchedules_.at(levelNumber)->fillData(initDataTime);
            copyLevelGhostOldToPushable_(*level, model);
        }

        // Clear particle destination buckets before a spawn fillData episode. The spawn
        // strategy's postprocessRefine is append-only (called once per fill box — clearing
        // there keeps only the last box, losing e.g. the upper level-ghost band); the clear
        // belongs here, at the lifecycle point, mirroring HybridHybrid where transfers
        // append and clears happen in lastStep/fresh allocation.
        void clearParticleBuckets_(SAMRAI::hier::PatchLevel& level, IPhysicalModel& model,
                                   bool const domain, bool const ghostOld, bool const ghostNew)
        {
            auto& hybridModel = static_cast<HybridModel&>(model);
            auto& ions        = hybridModel.state.ions;
            for (auto& patch : level)
            {
                auto dataOnPatch = resourcesManager_->setOnPatch(*patch, ions);
                for (auto& pop : ions)
                {
                    if (domain)
                        pop.domainParticles().clear();
                    if (ghostOld)
                        pop.levelGhostParticlesOld().clear();
                    if (ghostNew)
                        pop.levelGhostParticlesNew().clear();
                }
            }
        }

        void copyLevelGhostOldToPushable_(SAMRAI::hier::PatchLevel& level, IPhysicalModel& model)
        {
            auto& hybridModel = static_cast<HybridModel&>(model);
            auto& ions        = hybridModel.state.ions;
            for (auto& patch : level)
            {
                auto dataOnPatch = resourcesManager_->setOnPatch(*patch, ions);
                for (auto& pop : ions)
                    pop.levelGhostParticles() = pop.levelGhostParticlesOld();
            }
        }

        static double getHeatCapacityRatio_()
        {
            auto const& simAlgo
                = PHARE::initializer::PHAREDictHandler::INSTANCE().dict()["simulation"]["algo"];
            return simAlgo["heat_capacity_ratio"].template to<double>();
        }

        static double getElectronTe_()
        {
            auto const& closure = PHARE::initializer::PHAREDictHandler::INSTANCE()
                                      .dict()["simulation"]["electrons"]["pressure_closure"];
            auto const name = closure["name"].template to<std::string>();
            if (name != "isothermal")
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: unsupported electron closure '" + name
                    + "' — spawn Pe requires isothermal");
            return closure["Te"].template to<double>();
        }

        void populateSpawnStrategies_(IonsT const& ions)
        {
            using Bucket = typename ParticleSpawnStrategy::SpawnTargetBucket;

            std::vector<typename ParticleSpawnStrategy::PopParams>
                domainParams, oldParams, newParams;
            std::vector<std::string> popNames;
            std::size_t popIdx = 0;
            double sumQ = 0.0, sumM = 0.0;
            std::uint32_t totalPPC = 0;
            for (auto const& pop : ions)
            {
                auto const& info    = pop.particleInitializerInfo();
                double const charge = info["charge"].template to<double>();
                double const mass   = pop.mass();
                auto const nbrPPC   = static_cast<std::uint32_t>(
                    info["nbr_part_per_cell"].template to<int>());
                std::optional<std::size_t> seed;
                if (info.contains("init") && info["init"].contains("seed"))
                    seed = info["init"]["seed"].template to<std::optional<std::size_t>>();

                sumQ += nbrPPC * charge;
                sumM += nbrPPC * mass;
                totalPPC += nbrPPC;

                domainParams.push_back({domainPartFineIds_[popIdx], charge, mass, nbrPPC, seed});
                oldParams.push_back({lvlGhostOldFineIds_[popIdx], charge, mass, nbrPPC, seed});
                newParams.push_back({lvlGhostNewFineIds_[popIdx], charge, mass, nbrPPC, seed});
                popNames.push_back(pop.name());
                ++popIdx;
            }

            // Pe(rho) = Te * Ne with Ne = rho * (qBar/mBar): the same charge density the
            // sync derives from the spawned particles, so subtracted Pe == re-added Pe.
            double const qOverM = sumQ / sumM;
            double const Te     = getElectronTe_();
            auto const pe       = [Te, qOverM](double rho) {
                return core::IsothermalElectronPressureClosure<std::decay_t<IonsT>>::pressure(
                    rho * qOverM, Te);
            };

            // Nested-frame spawn fragments: one single-population (Domain, no-clip) instance
            // per pop, resolved by the hybrid pools' dispatchers at MHD_Hyb interp frames
            // (step 6). One ParticlesData id per pop — Domain is the only bucket the copy
            // transactions read on interp temp levels. Do NOT reuse spawnStratOld_/New_
            // here: their buckets are the fine-level ghost lists, not what a nested frame
            // feeds. Fragment keys must match the hybrid pools' refiner names — both sides
            // derive from pop.name().
            if (crossModelContext_)
            {
                double const meanMass = sumM / totalPPC;
                for (std::size_t i = 0; i < domainParams.size(); ++i)
                {
                    auto& fragment = nestedSpawnFragments_[popNames[i]];
                    if (!fragment)
                        fragment = std::make_shared<ParticleSpawnStrategy>();
                    fragment->setSpawnMode(Bucket::Domain, /*clipToInterior=*/false);
                    fragment->setFieldIds(primRhoId_, primVId_, primPId_);
                    fragment->setPopulations({domainParams[i]});
                    // single-pop list: restore the across-population weight split
                    fragment->setSpawnWeights(totalPPC, meanMass);
                    fragment->setElectronPressure(pe);
                    crossModelContext_->setNestedSpawnFragment(popNames[i], fragment);
                }
            }

            spawnStratDomain_.setSpawnMode(Bucket::Domain, /*clipToInterior=*/true);
            spawnStratOld_.setSpawnMode(Bucket::LevelGhostOld, /*clipToInterior=*/true);
            spawnStratNew_.setSpawnMode(Bucket::LevelGhostNew, /*clipToInterior=*/true);
            spawnStratDomain_.setPopulations(std::move(domainParams));
            spawnStratOld_.setPopulations(std::move(oldParams));
            spawnStratNew_.setPopulations(std::move(newParams));
            spawnStratDomain_.setElectronPressure(pe);
            spawnStratOld_.setElectronPressure(pe);
            spawnStratNew_.setElectronPressure(pe);
        }

        double timeInterpCoef_(double const afterPushTime, std::size_t levelNumber)
        {
            return (afterPushTime - beforePushCoarseTime_[levelNumber])
                   / (afterPushCoarseTime_[levelNumber] - beforePushCoarseTime_[levelNumber]);
        }

    };

} // namespace amr
} // namespace PHARE

#endif
