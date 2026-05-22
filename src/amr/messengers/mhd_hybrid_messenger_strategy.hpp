#ifndef PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP
#define PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP

#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "amr/messengers/spawn_maxwellian_from_mhd.hpp"
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

#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefineSchedule.h>

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
        using ParticlesDataT   = ParticlesData<ParticleArrayT>;

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
            resourcesManager_->registerResources(bGhostScratch_);
            resourcesManager_->registerResources(eGhostScratch_);
            resourcesManager_->registerResources(jGhostScratch_);
        }

        void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override
        {
            resourcesManager_->allocate(sumVec_, patch, allocateTime);
            resourcesManager_->allocate(sumField_, patch, allocateTime);
            resourcesManager_->allocate(bGhostScratch_, patch, allocateTime);
            resourcesManager_->allocate(eGhostScratch_, patch, allocateTime);
            resourcesManager_->allocate(jGhostScratch_, patch, allocateTime);
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

                magGhostSchedules_[levelNumber]
                    = magGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy,
                                                   &magComms_.magneticRefinePatchStrategy_);
                eGhostSchedules_[levelNumber]
                    = eGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                currentGhostSchedules_[levelNumber]
                    = currentGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);

                // Allocate coarse-source particle arrays (domain + level-ghost old/new) on the
                // MHD level below us. These are hybrid-model SAMRAI IDs reused on coarse patches.
                // Use the existing MHD patch-data time so SAMRAI's RefineSchedule sees matching
                // src/dst times during ghost fills.
                // Hybrid-typed B/E/J ghost-fill scratch IDs are also allocated coarse-side so
                // SAMRAI's d_coarse_interp_level can stage src→scratch→refine→dst.
                auto const coarseLevel = hierarchy->getPatchLevel(levelNumber - 1);
                auto bScratchId = resourcesManager_->getID(bGhostScratch_.name());
                auto eScratchId = resourcesManager_->getID(eGhostScratch_.name());
                auto jScratchId = resourcesManager_->getID(jGhostScratch_.name());
                if (!bScratchId or !eScratchId or !jScratchId)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy::registerLevel: ghost scratch IDs missing");
                for (auto const& patch : *coarseLevel)
                {
                    double const allocTime
                        = patch->getPatchData(mhdRhoId_)->getTime();
                    for (int id : coarseDomainPartIds_)
                        if (!patch->checkAllocated(id))
                            patch->allocatePatchData(id, allocTime);
                    for (int id : coarseGhostPartOldIds_)
                        if (!patch->checkAllocated(id))
                            patch->allocatePatchData(id, allocTime);
                    for (int id : coarseGhostPartNewIds_)
                        if (!patch->checkAllocated(id))
                            patch->allocatePatchData(id, allocTime);
                    for (int id : {*bScratchId, *eScratchId, *jScratchId})
                        if (!patch->checkAllocated(id))
                            patch->allocatePatchData(id, allocTime);
                }

                domainParticlesRefiners_.registerLevel(hierarchy, level);
                lvlGhostPartOldRefiners_.registerLevel(hierarchy, level);
                lvlGhostPartNewRefiners_.registerLevel(hierarchy, level);
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

            spawnCoarseParticles_(lvl - 1, ParticleSet::New, ions);
            lvlGhostPartNewRefiners_.fill(lvl, currentTime);

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
                spawnCoarseParticles_(lvl - 1, ParticleSet::Domain, ions);
                domainParticlesRefiners_.fill(lvl, initDataTime);

                spawnCoarseParticles_(lvl - 1, ParticleSet::Old, ions);
                lvlGhostPartOldRefiners_.fill(lvl, initDataTime);
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

            magComms_.magneticRegriding_(hierarchy, level, oldLevel, initDataTime);
            eInitComms_.fill(levelNumber, initDataTime);

            spawnCoarseParticles_(levelNumber - 1, ParticleSet::Domain, ions);
            domainParticlesRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);

            if (levelNumber != rootLevelNumber)
            {
                spawnCoarseParticles_(levelNumber - 1, ParticleSet::Old, ions);
                lvlGhostPartOldRefiners_.fill(levelNumber, initDataTime);
                copyLevelGhostOldToPushable_(*level, model);
            }
        }

        void fillMagneticGhosts(VecFieldT& B, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(B, level, *resourcesManager_);
            magGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillElectricGhosts(VecFieldT& E, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(E, level, *resourcesManager_);
            eGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillCurrentGhosts(VecFieldT& J, SAMRAI::hier::PatchLevel const& level,
                               double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(J, level, *resourcesManager_);
            currentGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
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

        // Ghost fill algos + schedule maps (B/E/J ghost fills)
        SAMRAI::xfer::RefineAlgorithm magGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> magGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm eGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> eGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm currentGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> currentGhostSchedules_;

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

        // Hybrid-typed staging VecFields used as the scratch arg of MHD→Hybrid ghost-fill
        // RefineAlgorithms. Allocated on both hybrid fine patches (via allocate()) AND on
        // MHD coarse patches (via registerLevel) so SAMRAI's d_coarse_interp_level can build
        // the canonical src→scratch→refine→dst pipeline (scratch != src, scratch != dst).
        VecFieldT bGhostScratch_{"MHDHybrid_bGhostScratch", core::PhysicalQuantity::Vector::B};
        VecFieldT eGhostScratch_{"MHDHybrid_eGhostScratch", core::PhysicalQuantity::Vector::E};
        VecFieldT jGhostScratch_{"MHDHybrid_jGhostScratch", core::PhysicalQuantity::Vector::J};

        // Particle-to-moment interpolator
        core::Interpolator<dimension, interpOrder> interpolate_;

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
            auto bScratchId  = resourcesManager_->getID(bGhostScratch_.name());
            auto eScratchId  = resourcesManager_->getID(eGhostScratch_.name());
            auto jScratchId  = resourcesManager_->getID(jGhostScratch_.name());
            if (!mhd_b_id or !mhd_e_id or !mhd_j_id or !mhd_rho_id or !mhd_rhoV_id
                or !mhd_Etot_id or !hyb_b_id or !hyb_e_id or !hyb_j_id
                or !bScratchId or !eScratchId or !jScratchId)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing field IDs in registerGhostComms_");

            // B/E/J ghost fills (MHD→Hybrid, same-type after PhysicalQuantity merge).
            // Use hybrid-typed staging IDs as scratch so SAMRAI's canonical
            // src→scratch→refine→dst pipeline works across the model boundary.
            magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            magGhostAlgo_.registerRefine(*hyb_b_id, *mhd_b_id, *bScratchId, mhdMagRefineOp_,
                                         nonOverwriteInteriorTFfillPattern_);
            eGhostAlgo_.registerRefine(*hyb_e_id, *mhd_e_id, *eScratchId, mhdERefineOp_,
                                        nonOverwriteInteriorTFfillPattern_);
            currentGhostAlgo_.registerRefine(*hyb_j_id, *mhd_j_id, *jScratchId, mhdERefineOp_,
                                              nonOverwriteInteriorTFfillPattern_);

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

            // B init fill (MHD→Hybrid, same-type after PhysicalQuantity merge).
            // Path B: use hybrid-typed staging IDs as scratch (allocated coarse+fine
            // in registerLevel/allocate), matching the runtime ghost-fill pattern.
            auto hyb_b_id    = resourcesManager_->getID(hybridInfo->modelMagnetic);
            auto bScratchId  = resourcesManager_->getID(bGhostScratch_.name());
            auto eScratchId  = resourcesManager_->getID(eGhostScratch_.name());
            if (!hyb_b_id or !bScratchId or !eScratchId)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing Hybrid B/scratch IDs in "
                    "registerInitComms_");
            magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            magComms_.BalgoInit.registerRefine(*hyb_b_id, *mhd_B_id, *bScratchId,
                                               mhdMagRefineOp_,
                                               overwriteInteriorTFfillPattern_);

            // E init fills (MHD→Hybrid, same-type after PhysicalQuantity merge)
            for (auto const& eName : hybridInfo->initElectric)
            {
                auto hyb_e_id = resourcesManager_->getID(eName);
                if (!hyb_e_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing Hybrid E ID for " + eName);
                eInitComms_.algo.registerRefine(*hyb_e_id, *mhd_e_id, *eScratchId,
                                                mhdERefineOp_, nullptr);
            }

            // Cache MHD conservative IDs + EOS gamma for spawnCoarseParticles_
            mhdRhoId_  = *mhd_rho_id;
            mhdRhoVId_ = *mhd_rhoV_id;
            mhdBId_    = *mhd_B_id;
            mhdEtotId_ = *mhd_Etot_id;
            gamma_     = getHeatCapacityRatio_();

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
                      << " mhdEtotId_=" << mhdEtotId_ << std::endl;
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
