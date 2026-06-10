#ifndef PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP
#define PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP

#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "amr/messengers/spawn_maxwellian_from_mhd.hpp"
#include "amr/messengers/mhd_hybrid_particle_spawn_strategy.hpp"
#include "amr/messengers/mhd_hybrid/mhd_hybrid_reflux_comms.hpp"
#include "amr/messengers/hybrid_hybrid/hybrid_border_comms.hpp"
#include "amr/messengers/refiner_pool.hpp"
#include "amr/data/particles/particles_variable_fill_pattern.hpp"
#include "core/physical_quantities.hpp"
#include "core/numerics/interpolator/interpolator.hpp"
#include "amr/data/tensorfield/tensor_field_data.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/magnetic_field_refiner.hpp"
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
#include "core/data/ions/ion_population/particle_pack.hpp"

#include <array>
#include <iostream>
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
        using MHDERefineOp    = MHDVecFieldRefineOp<ElectricFieldRefiner<dimension>>;
        using MHDMagRefineOp  = MHDVecFieldRefineOp<MagneticFieldRefiner<dimension>>;
        using MHDFluxRefineOp = FieldRefineOperator<MHDGridLayoutT, MHDGridT,
                                                    MHDFluxRefiner<dimension>>;
        using MHDVecFluxRefineOp = MHDVecFieldRefineOp<MHDFluxRefiner<dimension>>;

        // Prim-field refine ops for the cons→prim→fine spawn pipeline
        using MHDScalarPrimRefineOp
            = FieldRefineOperator<MHDGridLayoutT, MHDGridT, MHDFieldRefiner<dimension>>;
        using MHDVecPrimRefineOp = MHDVecFieldRefineOp<MHDFieldRefiner<dimension>>;

        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension>;

        using FieldT = VecFieldT::field_type;
        static constexpr std::size_t interpOrder = HybridGridLayoutT::interp_order;
        using rm_t                   = RMType;
        using DomainGhostPartRefinerPool = RefinerPool<rm_t, RefinerType::ExteriorGhostParticles>;
        using InitDomPartRefinerPool     = RefinerPool<rm_t, RefinerType::InitInteriorPart>;

        using CoarseToFineRefineOpOld = typename RefinementParams::CoarseToFineRefineOpOld;
        using CoarseToFineRefineOpNew = typename RefinementParams::CoarseToFineRefineOpNew;
        using InteriorParticleRefineOp = typename RefinementParams::InteriorParticleRefineOp;
        static auto constexpr LGRefT  = RefinerType::LevelBorderParticles;
        using RefOp_ptr               = std::shared_ptr<SAMRAI::hier::RefineOperator>;
        using LvlGhostPartRefinerPool = RefinerPool<rm_t, LGRefT>;

        using MHDFieldDataT    = FieldData<MHDGridLayoutT, MHDGridT>;
        using MHDVecFieldDataT = TensorFieldData<1, MHDGridLayoutT, MHDGridT, core::PhysicalQuantity>;
        using ParticleArrayT   = typename IonsT::particle_array_type;
        using ParticlesDataT        = ParticlesData<ParticleArrayT>;
        using ParticleSpawnStrategy = MHDHybridParticleSpawnStrategy<
            MHDFieldDataT, MHDVecFieldDataT, ParticlesDataT, MHDGridLayoutT>;

    public:
        static inline std::string const stratName = "MHDModel-HybridModel";

        MHDHybridMessengerStrategy(std::shared_ptr<RMType> const& rm, int const firstLevel)
            : HybridMessengerStrategy<HybridModel>{stratName}
            , resourcesManager_{rm}
            , firstLevel_{firstLevel}
            , magComms_{*rm}
        {
            resourcesManager_->registerResources(sumVec_);
            resourcesManager_->registerResources(sumField_);
            resourcesManager_->registerResources(primRho_);
            resourcesManager_->registerResources(primV_);
            resourcesManager_->registerResources(primP_);
            resourcesManager_->registerResources(diagSumField_);
            resourcesManager_->registerResources(diagSumVec_);
            resourcesManager_->registerResources(diagSumTensor_);
        }

        void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override
        {
            resourcesManager_->allocate(sumVec_, patch, allocateTime);
            resourcesManager_->allocate(sumField_, patch, allocateTime);
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
            refluxComms_.registerQuantities(*mhdInfo, *hybridInfo, *resourcesManager_,
                                            mhdERefineOp_, mhdFluxRefineOp_, mhdVecFluxRefineOp_,
                                            nonOverwriteInteriorTFfillPattern_);
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
                // NOTE: periodic-x fine ghosts are NOT filled by this coarse→fine pass regardless
                // of Border vs Full fill pattern — the src_level=nullptr form does not build the
                // periodic coarse image (tested 2026-06-07, FullFill gave identical periodic-x NaN).
                // Periodic-x is an OPEN PROBLEM; see branch handoff. BorderFill kept (fills y-ghosts).
                auto ghostBorderFill
                    = std::make_shared<SAMRAI::xfer::PatchLevelBorderFillPattern>();
                // One schedule pair per field (keyed by name). B levelGhost passes the field's
                // own MagneticRefinePatchStrategy (div-correction); E/J levelGhost take none.
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
                primDomainSchedules_[levelNumber]
                    = primAlgoDomain_.createSchedule(interiorFillPattern, level, nullptr,
                                                      levelNumber - 1, hierarchy,
                                                      &spawnStratDomain_);
                // Old/New: ghost-only fill from coarse MHD.
                // PatchLevelBorderFillPattern restricts fill to coarse-fine ghost region →
                // postprocessRefine receives ghost boxes only → spawn levelGhost particles.
                // src_level=nullptr (InitField form): no same-level copy transactions →
                // MHD src_ids (30/31/33) are never looked up on Hybrid fine patches → no crash.
                // (GhostField form "createSchedule(level, levelNumber-1, ...)" sets
                //  d_src_level=level and generates same-level copy transactions that try to
                //  read mhdVId_/etc. from Hybrid patches where they are not allocated → crash.)
                auto borderFillPattern
                    = std::make_shared<SAMRAI::xfer::PatchLevelBorderFillPattern>();
                primOldSchedules_[levelNumber]
                    = primAlgoOld_.createSchedule(borderFillPattern, level, nullptr,
                                                   levelNumber - 1, hierarchy, &spawnStratOld_);
                primNewSchedules_[levelNumber]
                    = primAlgoNew_.createSchedule(borderFillPattern, level, nullptr,
                                                   levelNumber - 1, hierarchy, &spawnStratNew_);

                domainGhostPartRefiners_.registerLevel(hierarchy, level);
                borderComms_.registerLevel(levelNumber, hierarchy);
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

        // initLevel: fill B and E from MHD, spawn coarse domain+ghost particles from MHD
        // conservatives, refine them coarse→fine via standard particle refiners.
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
                {
                    auto coarseLevel = hierarchy_.lock()->getPatchLevel(lvl - 1);
                    for (auto const& patch : *coarseLevel)
                        std::cout << "[DIAG initLevel prim] coarse patch " << patch->getLocalId()
                                  << " rhoAlloc=" << patch->checkAllocated(mhdRhoId_)
                                  << " VAlloc=" << patch->checkAllocated(mhdVId_)
                                  << " PAlloc=" << patch->checkAllocated(mhdPId_) << std::endl;
                    for (auto const& patch : level)
                        std::cout << "[DIAG initLevel prim] fine patch " << patch->getLocalId()
                                  << " primRhoAlloc=" << patch->checkAllocated(primRhoId_)
                                  << " primVAlloc=" << patch->checkAllocated(primVId_)
                                  << " primPAlloc=" << patch->checkAllocated(primPId_) << std::endl;
                }
                {
                    std::size_t domSz = 0;
                    for (auto& patch : level)
                    {
                        auto dataOnPatch = resourcesManager_->setOnPatch(*patch, ions);
                        for (auto& pop : ions)
                            domSz += pop.domainParticles().size();
                    }
                    std::cout << "[DIAG initLevel] PRE-domain-fill domainParts.size=" << domSz << std::endl;
                }
                primDomainSchedules_.at(lvl)->fillData(initDataTime);
                {
                    std::size_t domSz = 0;
                    for (auto& patch : level)
                    {
                        auto dataOnPatch = resourcesManager_->setOnPatch(*patch, ions);
                        for (auto& pop : ions)
                            domSz += pop.domainParticles().size();
                    }
                    std::cout << "[DIAG initLevel] POST-domain-fill domainParts.size=" << domSz << std::endl;
                }
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

            // B regrid is two-step (cross-model constraint, see BregridAlgo registration):
            // 1) fill the whole new level from coarse MHD B, exactly like initLevel;
            // 2) overwrite with old fine-level B wherever the old level overlaps.
            // INTERIM — not divB-conserving by design: the two passes are independent, so
            // nothing constrains face fluxes at the copy/refine seam. To be redesigned
            // (overplan phase 7); kept only to keep the advance pipeline crash-free.
            auto coarseFillSchedule = magComms_.BalgoInit.createSchedule(
                level, nullptr, levelNumber - 1, hierarchy,
                &magComms_.magneticRefinePatchStrategy_);
            coarseFillSchedule->fillData(initDataTime);
            auto oldCopySchedule = magComms_.BregridAlgo.createSchedule(level, oldLevel);
            oldCopySchedule->fillData(initDataTime);

            eInitComms_.fill(levelNumber, initDataTime);

            if (levelNumber != rootLevelNumber)
            {
                populateSpawnStrategies_(ions);
                consToPrim_(levelNumber - 1);
                primDomainSchedules_.at(levelNumber)->fillData(initDataTime);
                primOldSchedules_.at(levelNumber)->fillData(initDataTime);
                copyLevelGhostOldToPushable_(*level, model);
            }
        }

        void fillMagneticGhosts(VecFieldT& B, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(B, level, *resourcesManager_);
            int const lvl = level.getLevelNumber();
            // DIAG 6.A: unconditional entry print so we know WHICH field entered and on which level
            // before any crash. Absence of a postFillMag NaN line is only meaningful if the entry
            // print confirms the field entered. TEMPORARY.
            std::cout << "[DIAG enterFillMag] name=" << B.name() << " lvl=" << lvl << std::endl;
            // Dispatch by field name: the schedule fills the IDs it was registered with, so we must
            // select the pair registered for THIS field (e.g. "EM_B" vs "EMPred_B").
            // levelGhost (coarse MHD, BorderFill) first → fills y CF ghosts; patchGhost
            // (same-level) second → exact same-level data wins on overlap.
            magLevelGhostSchedules_.at(B.name()).at(lvl)->fillData(fillTime);
            // DIAG: per-pass scan AFTER the coarse-fine (levelGhost/BorderFill) pass, BEFORE the
            // same-level (patchGhost) pass. Periodic-x ghosts should still be NaN here (BorderFill
            // emits no periodic-x fill boxes). TEMPORARY.
            for (auto& patch : resourcesManager_->enumerate(level, B))
                for (auto& component : B)
                    diagScanFieldGhostsNonFinite<HybridGridLayoutT>(component, *patch,
                                                                    "postLevelGhost");
            magPatchGhostSchedules_.at(B.name()).at(lvl)->fillData(fillTime);
            // DIAG: per-pass scan AFTER the same-level (patchGhost) pass. This directly tests
            // whether createSchedule(level, nullptr) emits periodic-x transactions: if periodic-x
            // columns are FINITE here it does; if still NaN it does not -> escalate to shared-id.
            // TEMPORARY.
            for (auto& patch : resourcesManager_->enumerate(level, B))
                for (auto& component : B)
                    diagScanFieldGhostsNonFinite<HybridGridLayoutT>(component, *patch,
                                                                    "postPatchGhost");
        }

        void fillElectricGhosts(VecFieldT& E, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(E, level, *resourcesManager_);
            int const lvl = level.getLevelNumber();
            eLevelGhostSchedules_.at(E.name()).at(lvl)->fillData(fillTime);
            ePatchGhostSchedules_.at(E.name()).at(lvl)->fillData(fillTime);
        }

        void fillCurrentGhosts(VecFieldT& J, SAMRAI::hier::PatchLevel const& level,
                               double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(J, level, *resourcesManager_);
            int const lvl = level.getLevelNumber();
            currentLevelGhostSchedules_.at(J.name()).at(lvl)->fillData(fillTime);
            currentPatchGhostSchedules_.at(J.name()).at(lvl)->fillData(fillTime);
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

        void synchronize(SAMRAI::hier::PatchLevel& /*level*/) final {}

        // reflux: coarsen Hybrid flux sums into MHD timeFluxes + ghost refill.
        void reflux(int const coarserLevelNumber, int const fineLevelNumber,
                    double const syncTime) override
        { refluxComms_.reflux(fineLevelNumber, coarserLevelNumber, syncTime); }

        void postSynchronize(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                             double const /*time*/) override
        {
            // No-op: conservation handled by the reflux mechanism.
        }

    private:
        enum class ParticleSet { Domain, Old, New };

        std::shared_ptr<RMType> resourcesManager_;
        int const firstLevel_;

        // Same-type MHD refine ops for refluxComms_ ghost fills
        std::shared_ptr<MHDERefineOp> mhdERefineOp_{std::make_shared<MHDERefineOp>()};
        std::shared_ptr<MHDMagRefineOp> mhdMagRefineOp_{std::make_shared<MHDMagRefineOp>()};
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
        // One MagneticRefinePatchStrategy per B field (each registers its own id for div-correction)
        std::map<std::string, std::shared_ptr<MagStratT>> magStratPerField_;
        RefineAlgoMap ePatchGhostAlgos_;
        RefineAlgoMap eLevelGhostAlgos_;
        RefineSchedMap ePatchGhostSchedules_;
        RefineSchedMap eLevelGhostSchedules_;
        RefineAlgoMap currentPatchGhostAlgos_;
        RefineAlgoMap currentLevelGhostAlgos_;
        RefineSchedMap currentPatchGhostSchedules_;
        RefineSchedMap currentLevelGhostSchedules_;

        // Coarse→fine particle refine ops + pools (hybrid-hybrid mirror)
        RefOp_ptr interiorParticleRefineOp_{std::make_shared<InteriorParticleRefineOp>()};
        RefOp_ptr levelGhostParticlesOldOp_{std::make_shared<CoarseToFineRefineOpOld>()};
        RefOp_ptr levelGhostParticlesNewOp_{std::make_shared<CoarseToFineRefineOpNew>()};
        InitDomPartRefinerPool domainParticlesRefiners_{resourcesManager_};
        LvlGhostPartRefinerPool lvlGhostPartOldRefiners_{resourcesManager_};
        LvlGhostPartRefinerPool lvlGhostPartNewRefiners_{resourcesManager_};

        // Strategy-owned ParticlesPack views — registered with resourcesManager_ so they
        // get unique SAMRAI IDs in the same database as the hybrid dest IDs (required by Refiner).
        // Source side of coarse→fine refinement; destinations are the hybrid model's particle IDs.
        std::vector<core::ParticlesPack<ParticleArrayT>> strategyCoarseDomainPacks_;
        std::vector<core::ParticlesPack<ParticleArrayT>> strategyCoarseOldPacks_;
        std::vector<core::ParticlesPack<ParticleArrayT>> strategyCoarseNewPacks_;
        std::vector<std::string> coarseDomainPartNames_;
        std::vector<std::string> coarseGhostPartOldNames_;
        std::vector<std::string> coarseGhostPartNewNames_;
        std::vector<int> coarseDomainPartIds_;
        std::vector<int> coarseGhostPartOldIds_;
        std::vector<int> coarseGhostPartNewIds_;

        // MHD conservative field IDs + heat capacity ratio (consumed by spawnCoarseParticles_)
        int mhdRhoId_  = -1;
        int mhdRhoVId_ = -1;
        int mhdBId_    = -1;
        int mhdEtotId_ = -1;
        double gamma_  = 5.0 / 3.0;

        // Hierarchy reference for spawnCoarseParticles_
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

        ParticleSpawnStrategy spawnStratDomain_, spawnStratOld_, spawnStratNew_;

        int primRhoId_ = -1, primVId_ = -1, primPId_ = -1;
        int mhdVId_ = -1, mhdPId_ = -1;
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
            for (auto const& key : hybridInfo->ghostMagnetic)
            {
                auto id = resourcesManager_->getID(key);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing B ghost field ID for " + key);
                auto strat = std::make_shared<MagStratT>(*resourcesManager_);
                strat->registerIDs(*id);
                magStratPerField_[key] = strat;
                magPatchGhostAlgos_[key].registerRefine(*id, *id, *id, nullptr,
                                                        nonOverwriteInteriorTFfillPattern_);
                magLevelGhostAlgos_[key].registerRefine(*id, *mhd_b_id, *id, mhdMagRefineOp_,
                                                        nonOverwriteInteriorTFfillPattern_);
            }
            for (auto const& key : hybridInfo->ghostElectric)
            {
                auto id = resourcesManager_->getID(key);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing E ghost field ID for " + key);
                ePatchGhostAlgos_[key].registerRefine(*id, *id, *id, nullptr,
                                                      nonOverwriteInteriorTFfillPattern_);
                eLevelGhostAlgos_[key].registerRefine(*id, *mhd_e_id, *id, mhdERefineOp_,
                                                      nonOverwriteInteriorTFfillPattern_);
            }
            for (auto const& key : hybridInfo->ghostCurrent)
            {
                auto id = resourcesManager_->getID(key);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing J ghost field ID for " + key);
                currentPatchGhostAlgos_[key].registerRefine(*id, *id, *id, nullptr,
                                                            nonOverwriteInteriorTFfillPattern_);
                currentLevelGhostAlgos_[key].registerRefine(*id, *mhd_j_id, *id, mhdERefineOp_,
                                                            nonOverwriteInteriorTFfillPattern_);
            }

            // Strategy-owned coarse particle buffers: registered with resourcesManager_
            // (same DB as the hybrid dest IDs), allocated on coarse MHD patches, written by
            // spawnCoarseParticles_, refined coarse→fine into hybrid level-ghost arrays.
            registerStrategyOwnedParticleBuffers_(
                "MHDHybMess_coarseGhostPartOld_pop", hybridInfo->levelGhostParticlesOld.size(),
                strategyCoarseOldPacks_, coarseGhostPartOldNames_, coarseGhostPartOldIds_);
            registerStrategyOwnedParticleBuffers_(
                "MHDHybMess_coarseGhostPartNew_pop", hybridInfo->levelGhostParticlesNew.size(),
                strategyCoarseNewPacks_, coarseGhostPartNewNames_, coarseGhostPartNewIds_);

            lvlGhostPartOldRefiners_.addStaticRefiners(hybridInfo->levelGhostParticlesOld,
                                                       coarseGhostPartOldNames_,
                                                       levelGhostParticlesOldOp_,
                                                       hybridInfo->levelGhostParticlesOld);
            lvlGhostPartNewRefiners_.addStaticRefiners(hybridInfo->levelGhostParticlesNew,
                                                       coarseGhostPartNewNames_,
                                                       levelGhostParticlesNewOp_,
                                                       hybridInfo->levelGhostParticlesNew);

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

            // B init fill: scratch==dst (same pattern as ghost fill, SAMRAI-legal).
            auto hyb_b_id = resourcesManager_->getID(hybridInfo->modelMagnetic);
            if (!hyb_b_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing Hybrid B ID in registerInitComms_");
            magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            magComms_.BalgoInit.registerRefine(*hyb_b_id, *mhd_B_id, *hyb_b_id,
                                               mhdMagRefineOp_,
                                               overwriteInteriorTFfillPattern_);

            // B regrid, step 2 (old-level copy): same-id, same-resolution → pure copy
            // transactions, no refine op needed. Step 1 (coarse MHD fill) reuses BalgoInit.
            // A single 3-level regrid schedule is impossible here: registerRefine has one src
            // slot but the schedule reads it on BOTH the old hybrid level (copy leg) and the
            // coarse MHD level (interp leg) — no ID is allocated on both. Verified empirically
            // 2026-06-10: src=mhd_B → null PatchData in RefineCopyTransaction::copyLocalData.
            magComms_.BregridAlgo.registerRefine(*hyb_b_id, *hyb_b_id, *hyb_b_id,
                                                 std::shared_ptr<SAMRAI::hier::RefineOperator>{},
                                                 overwriteInteriorTFfillPattern_);

            // E init fills: scratch==dst
            for (auto const& eName : hybridInfo->initElectric)
            {
                auto hyb_e_id = resourcesManager_->getID(eName);
                if (!hyb_e_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing Hybrid E ID for " + eName);
                eInitComms_.algo.registerRefine(*hyb_e_id, *mhd_e_id, *hyb_e_id,
                                                mhdERefineOp_, nullptr);
            }

            // Cache MHD conservative IDs + EOS gamma for spawnCoarseParticles_ / consToPrim_
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

            // Cache strategy-owned fine primitive field IDs
            auto prim_rho_id = resourcesManager_->getID(primRho_.name());
            auto prim_V_id   = resourcesManager_->getID(primV_.name());
            auto prim_P_id   = resourcesManager_->getID(primP_.name());
            if (!prim_rho_id or !prim_V_id or !prim_P_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing prim field IDs in registerInitComms_");
            primRhoId_ = *prim_rho_id;
            primVId_   = *prim_V_id;
            primPId_   = *prim_P_id;
            gamma_     = getHeatCapacityRatio_();

            // Wire spawn strategy field IDs (same prim fields for all three sets)
            spawnStratDomain_.setFieldIds(primRhoId_, primVId_, primPId_);
            spawnStratOld_.setFieldIds(primRhoId_, primVId_, primPId_);
            spawnStratNew_.setFieldIds(primRhoId_, primVId_, primPId_);

            // Register coarse MHD (rho, V, P) → fine strategy-owned prim fields in all three algos
            for (auto* algo : {&primAlgoDomain_, &primAlgoOld_, &primAlgoNew_})
            {
                algo->registerRefine(primRhoId_, mhdRhoId_, primRhoId_, mhdScalarPrimRefineOp_);
                algo->registerRefine(primVId_,   mhdVId_,   primVId_,   mhdVecPrimRefineOp_);
                algo->registerRefine(primPId_,   mhdPId_,   primPId_,   mhdScalarPrimRefineOp_);
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

            // Strategy-owned coarse interior particle buffers
            registerStrategyOwnedParticleBuffers_(
                "MHDHybMess_coarseDomainPart_pop", hybridInfo->interiorParticles.size(),
                strategyCoarseDomainPacks_, coarseDomainPartNames_, coarseDomainPartIds_);

            domainParticlesRefiners_.addStaticRefiners(hybridInfo->interiorParticles,
                                                       coarseDomainPartNames_,
                                                       interiorParticleRefineOp_,
                                                       hybridInfo->interiorParticles);

            std::cout << "[DIAG registerInitComms_] mhdRhoId_=" << mhdRhoId_
                      << " mhdRhoVId_=" << mhdRhoVId_ << " mhdBId_=" << mhdBId_
                      << " mhdEtotId_=" << mhdEtotId_
                      << " mhdVId_=" << mhdVId_ << " mhdPId_=" << mhdPId_
                      << " primRhoId_=" << primRhoId_
                      << " primVId_=" << primVId_ << " primPId_=" << primPId_ << std::endl;
        }

        void registerStrategyOwnedParticleBuffers_(
            std::string const& namePrefix, std::size_t count,
            std::vector<core::ParticlesPack<ParticleArrayT>>& packs,
            std::vector<std::string>& names, std::vector<int>& ids)
        {
            packs.clear();
            names.clear();
            ids.clear();
            packs.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                std::string const name = namePrefix + std::to_string(i);
                packs.emplace_back();
                packs.back()._name = name;
                resourcesManager_->registerResources(packs.back());
                auto id = resourcesManager_->getID(name);
                if (!id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: failed to register strategy-owned "
                        + name);
                names.push_back(name);
                ids.push_back(*id);
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

        void populateSpawnStrategies_(IonsT const& ions)
        {
            std::vector<typename ParticleSpawnStrategy::PopParams>
                domainParams, oldParams, newParams;
            std::size_t popIdx = 0;
            for (auto const& pop : ions)
            {
                auto const& info    = pop.particleInitializerInfo();
                double const charge = info["charge"].template to<double>();
                auto const nbrPPC   = static_cast<std::uint32_t>(
                    info["nbr_part_per_cell"].template to<int>());
                std::optional<std::size_t> seed;
                if (info.contains("init") && info["init"].contains("seed"))
                    seed = info["init"]["seed"].template to<std::optional<std::size_t>>();

                domainParams.push_back({domainPartFineIds_[popIdx], charge, nbrPPC, seed,
                                        ParticleSpawnStrategy::ParticleBucket::Domain});
                oldParams.push_back({lvlGhostOldFineIds_[popIdx], charge, nbrPPC, seed,
                                     ParticleSpawnStrategy::ParticleBucket::GhostOld});
                newParams.push_back({lvlGhostNewFineIds_[popIdx], charge, nbrPPC, seed,
                                     ParticleSpawnStrategy::ParticleBucket::GhostNew});
                ++popIdx;
            }
            spawnStratDomain_.setPopulations(std::move(domainParams));
            spawnStratOld_.setPopulations(std::move(oldParams));
            spawnStratNew_.setPopulations(std::move(newParams));
        }

        double timeInterpCoef_(double const afterPushTime, std::size_t levelNumber)
        {
            return (afterPushTime - beforePushCoarseTime_[levelNumber])
                   / (afterPushCoarseTime_[levelNumber] - beforePushCoarseTime_[levelNumber]);
        }

        std::vector<int>& particleSetIds_(ParticleSet set)
        {
            switch (set)
            {
                case ParticleSet::Domain: return coarseDomainPartIds_;
                case ParticleSet::Old:    return coarseGhostPartOldIds_;
                case ParticleSet::New:    return coarseGhostPartNewIds_;
            }
            throw std::runtime_error("MHDHybridMessengerStrategy: bad ParticleSet");
        }

        void spawnCoarseParticles_(int coarseLevelNumber, ParticleSet set, IonsT const& ions)
        {
            auto& ids = particleSetIds_(set);
            if (ids.empty())
                return;

            auto hierarchy = hierarchy_.lock();
            if (!hierarchy)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy::spawnCoarseParticles_: hierarchy expired");

            auto const coarseLevel = hierarchy->getPatchLevel(coarseLevelNumber);

            std::size_t popIdx = 0;
            for (auto const& pop : ions)
            {
                auto const& info = pop.particleInitializerInfo();
                double const charge = info["charge"].template to<double>();
                auto const nbrPPC = static_cast<std::uint32_t>(info["nbr_part_per_cell"].template to<int>());
                std::optional<std::size_t> userSeed;
                if (info.contains("init") && info["init"].contains("seed"))
                    userSeed = info["init"]["seed"].template to<std::optional<std::size_t>>();
                int const particleDataId = ids[popIdx];

                for (auto const& patch : *coarseLevel)
                {
                    std::cout << "[DIAG spawnCoarseParticles_] coarseLvl="
                              << coarseLevelNumber << " set=" << static_cast<int>(set)
                              << " patchId=" << patch->getLocalId().getValue()
                              << " mhdRhoId_=" << mhdRhoId_ << " alloc="
                              << patch->checkAllocated(mhdRhoId_)
                              << " mhdRhoVId_=" << mhdRhoVId_ << " alloc="
                              << patch->checkAllocated(mhdRhoVId_)
                              << " mhdBId_=" << mhdBId_ << " alloc="
                              << patch->checkAllocated(mhdBId_)
                              << " mhdEtotId_=" << mhdEtotId_ << " alloc="
                              << patch->checkAllocated(mhdEtotId_) << std::endl;

                    auto& rho       = MHDFieldDataT::getField(*patch, mhdRhoId_);
                    auto& rhoVcomps = MHDVecFieldDataT::getFields(*patch, mhdRhoVId_);
                    auto& Bcomps    = MHDVecFieldDataT::getFields(*patch, mhdBId_);
                    auto& Etot      = MHDFieldDataT::getField(*patch, mhdEtotId_);

                    auto layout    = layoutFromPatch<MHDGridLayoutT>(*patch);
                    auto const& dx = layout.meshSize();

                    auto localIdx = [&](double x, [[maybe_unused]] double y,
                                        [[maybe_unused]] double z) {
                        int const ix = static_cast<int>(x / dx[0]);
                        if constexpr (dimension == 1)
                            return layout.AMRToLocal(core::Point<int, 1>{ix});
                        else if constexpr (dimension == 2)
                        {
                            int const iy = static_cast<int>(y / dx[1]);
                            return layout.AMRToLocal(core::Point<int, 2>{ix, iy});
                        }
                        else
                        {
                            int const iy = static_cast<int>(y / dx[1]);
                            int const iz = static_cast<int>(z / dx[2]);
                            return layout.AMRToLocal(core::Point<int, 3>{ix, iy, iz});
                        }
                    };

                    std::optional<std::size_t> seed = userSeed;
                    if (!seed)
                        seed = static_cast<std::size_t>(coarseLevelNumber) + popIdx
                               + static_cast<std::size_t>(patch->getBox().lower(0));

                    auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                        patch->getPatchData(particleDataId));
                    partData.domainParticles.clear();
                    spawnMaxwellianFromMHD(layout, rho, rhoVcomps, Bcomps, Etot, localIdx, gamma_,
                                           charge, nbrPPC, seed, partData.domainParticles);
                }
                ++popIdx;
            }
        }
    };

} // namespace amr
} // namespace PHARE

#endif
