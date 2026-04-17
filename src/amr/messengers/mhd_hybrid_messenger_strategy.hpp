#ifndef PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP
#define PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP

#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "amr/messengers/mhd_hybrid_particle_injection_patch_strategy.hpp"
#include "amr/messengers/mhd_hybrid/mhd_hybrid_reflux_comms.hpp"
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

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel, typename HybridModel>
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

        using HybridRMType = typename HybridModel::resources_manager_type;
        using MHDRMType    = typename MHDModel::resources_manager_type;

        // VecFieldData type for Hybrid B (needed by BfieldComms / MagneticRefinePatchStrategy)
        using HybVectorFieldDataT
            = TensorFieldData<1, HybridGridLayoutT, HybridGridT, core::HybridQuantity>;

        // Same-type MHD refine ops for initLevelAlgo_ (particle injection context fields)
        // Must be same-type to avoid null PatchData crash via d_coarse_interp_level.
        template<typename Policy>
        using MHDVecFieldRefineOp = VecFieldRefineOperator<MHDGridLayoutT, MHDGridT, Policy>;
        using MHDHydroRefineOp = FieldRefineOperator<MHDGridLayoutT, typename MHDModel::field_type,
                                                     MHDFieldRefiner<dimension>>;
        using MHDVecHydroRefineOp = MHDVecFieldRefineOp<MHDFieldRefiner<dimension>>;

        // Same-type MHD refine ops passed to refluxComms_ for ghost fills after coarsen
        using MHDERefineOp    = MHDVecFieldRefineOp<ElectricFieldRefiner<dimension>>;
        using MHDMagRefineOp  = MHDVecFieldRefineOp<MagneticFieldRefiner<dimension>>;
        using MHDFluxRefineOp = FieldRefineOperator<MHDGridLayoutT, typename MHDModel::field_type,
                                                    MHDFluxRefiner<dimension>>;
        using MHDVecFluxRefineOp = MHDVecFieldRefineOp<MHDFluxRefiner<dimension>>;

        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension>;

        using ParticleInjectionStrategy
            = MHDHybridParticleInjectionPatchStrategy<MHDModel, HybridModel>;

    public:
        static inline std::string const stratName = "MHDModel-HybridModel";

        MHDHybridMessengerStrategy(std::shared_ptr<MHDRMType> const& mhdRm,
                                   std::shared_ptr<HybridRMType> const& hybridRm,
                                   int const firstLevel)
            : HybridMessengerStrategy<HybridModel>{stratName}
            , mhdResourcesManager_{mhdRm}
            , hybridResourcesManager_{hybridRm}
            , firstLevel_{firstLevel}
            , magComms_{*hybridRm}
        {
        }

        void allocate(SAMRAI::hier::Patch& /*patch*/, double const /*allocateTime*/) const override
        {
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
            refluxComms_.registerQuantities(*mhdInfo, *hybridInfo, *mhdResourcesManager_,
                                            *hybridResourcesManager_, mhdERefineOp_,
                                            mhdFluxRefineOp_, mhdVecFluxRefineOp_,
                                            nonOverwriteInteriorTFfillPattern_);
        }

        void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                           int const levelNumber) override
        {
            refluxComms_.registerLevel(levelNumber, hierarchy);

            if (levelNumber != rootLevelNumber)
            {
                auto const level = hierarchy->getPatchLevel(levelNumber);

                magComms_.magInitRefineSchedules_[levelNumber]
                    = magComms_.BalgoInit.createSchedule(level, nullptr, levelNumber - 1, hierarchy,
                                                         &magComms_.magneticRefinePatchStrategy_);

                eInitComms_.createSchedule(levelNumber, level, levelNumber - 1, hierarchy);

                initLevelSchedules_[levelNumber] = initLevelAlgo_.createSchedule(
                    level, nullptr, levelNumber - 1, hierarchy, &particleInjectionStrategy_);

                magGhostSchedules_[levelNumber]
                    = magGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy,
                                                   &magComms_.magneticRefinePatchStrategy_);
                eGhostSchedules_[levelNumber]
                    = eGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                currentGhostSchedules_[levelNumber]
                    = currentGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                ionMomentGhostSchedules_[levelNumber]
                    = ionMomentGhostAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                densityBorderSchedules_[levelNumber]
                    = densityBorderAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                fluxBorderSchedules_[levelNumber]
                    = fluxBorderAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                ionBorderSchedules_[levelNumber]
                    = ionBorderAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                ghostParticleSchedules_[levelNumber]
                    = ghostParticleAlgo_.createSchedule(level, levelNumber - 1, hierarchy,
                                                        &ghostParticleInjectionStrategy_);
            }
        }

        std::unique_ptr<IMessengerInfo> emptyInfoFromCoarser() override
        { return std::make_unique<MHDMessengerInfo>(); }

        std::unique_ptr<IMessengerInfo> emptyInfoFromFiner() override
        { return std::make_unique<HybridMessengerInfo>(); }

        std::string fineModelName() const override { return HybridModel::model_name; }
        std::string coarseModelName() const override { return MHDModel::model_name; }

        virtual ~MHDHybridMessengerStrategy() = default;

        void firstStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& /*hierarchy*/,
                       double const /*currentTime*/, double const /*prevCoarserTime*/,
                       double const /*newCoarserTime*/) override
        {
        }

        void lastStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/) override {}

        void prepareStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                         double /*currentTime*/) final
        {
        }

        void fillRootGhosts(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                            double const /*initDataTime*/) final
        {
            // Root level is MHD-only; no cross-model root ghost fill needed.
        }

        // initLevel: fill B and E from MHD, then inject particles into new Hybrid patches.
        void initLevel(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       double const initDataTime) override
        {
            int const lvl     = level.getLevelNumber();
            auto& hybridModel = dynamic_cast<HybridModel&>(model);
            magComms_.magInitRefineSchedules_.at(lvl)->fillData(initDataTime);
            eInitComms_.fill(lvl, initDataTime);
            setupParticleInjection_(hybridModel, lvl, particleInjectionStrategy_,
                                    initLevelSchedules_, initDataTime);
        }

        // regrid: SAMRAI restricts postprocessRefine to genuinely new boxes only.
        // Use magneticRegriding_ to preserve fine data in old-level overlap (not overwrite).
        void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                    int const levelNumber,
                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                    IPhysicalModel& model, double const initDataTime) override
        {
            auto const level  = hierarchy->getPatchLevel(levelNumber);
            auto& hybridModel = dynamic_cast<HybridModel&>(model);
            magComms_.magneticRegriding_(hierarchy, level, oldLevel, initDataTime);
            eInitComms_.fill(levelNumber, initDataTime);
            setupParticleInjection_(hybridModel, levelNumber, particleInjectionStrategy_,
                                    initLevelSchedules_, initDataTime);
        }

        void fillMagneticGhosts(VecFieldT& B, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(B, level, *hybridResourcesManager_);
            magGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillElectricGhosts(VecFieldT& E, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(E, level, *hybridResourcesManager_);
            eGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillCurrentGhosts(VecFieldT& J, SAMRAI::hier::PatchLevel const& level,
                               double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(J, level, *hybridResourcesManager_);
            currentGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillIonGhostParticles(IonsT& ions, SAMRAI::hier::PatchLevel& level,
                                   double const fillTime) override
        {
            int const lvl = level.getLevelNumber();
            setPopulationPhysicsFromIons_(ions, ghostParticleInjectionStrategy_);
            ghostParticleInjectionStrategy_.setLevelNumber(lvl);
            ghostParticleSchedules_.at(lvl)->fillData(fillTime);
        }

        void fillIonPopMomentGhosts(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& level,
                                    double const fillTime) override
        {
            ionMomentGhostSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillFluxBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& level,
                             double const fillTime) override
        {
            fluxBorderSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillDensityBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& level,
                                double const fillTime) override
        {
            densityBorderSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

        void fillIonBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& level,
                            double const fillTime) override
        {
            ionBorderSchedules_.at(level.getLevelNumber())->fillData(fillTime);
        }

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
        std::shared_ptr<MHDRMType> mhdResourcesManager_;
        std::shared_ptr<HybridRMType> hybridResourcesManager_;
        int const firstLevel_;

        // Same-type MHD refine ops for initLevelAlgo_
        std::shared_ptr<MHDHydroRefineOp> mhdHydroRefineOp_{std::make_shared<MHDHydroRefineOp>()};
        std::shared_ptr<MHDVecHydroRefineOp> mhdVecHydroRefineOp_{
            std::make_shared<MHDVecHydroRefineOp>()};
        std::shared_ptr<MHDMagRefineOp> mhdMagRefineOp_{std::make_shared<MHDMagRefineOp>()};

        // Same-type MHD refine ops for refluxComms_ ghost fills
        std::shared_ptr<MHDERefineOp> mhdERefineOp_{std::make_shared<MHDERefineOp>()};
        std::shared_ptr<MHDFluxRefineOp> mhdFluxRefineOp_{std::make_shared<MHDFluxRefineOp>()};
        std::shared_ptr<MHDVecFluxRefineOp> mhdVecFluxRefineOp_{
            std::make_shared<MHDVecFluxRefineOp>()};

        // Fill patterns
        std::shared_ptr<TensorFieldFillPattern_t> nonOverwriteInteriorTFfillPattern_{
            std::make_shared<TensorFieldFillPattern_t>()};
        std::shared_ptr<TensorFieldFillPattern_t> overwriteInteriorTFfillPattern_{
            std::make_shared<TensorFieldFillPattern_t>(true)};

        // Particle injection strategy
        ParticleInjectionStrategy particleInjectionStrategy_;

        // B fills: MagneticRefinePatchStrategy provides div-correction after interpolation.
        BfieldComms<HybridRMType, HybVectorFieldDataT> magComms_;

        // E fills
        EfieldComms eInitComms_;

        // Particle injection: separate from B/E algos so postprocessRefine fires only in
        // initLevel/regrid, not during ghost fills
        SAMRAI::xfer::RefineAlgorithm initLevelAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> initLevelSchedules_;

        // Reflux: 4 channels (E, HydroX, HydroY, HydroZ), cross-type coarsen + MHD ghost refill
        MHDHybridRefluxComms<MHDModel, HybridModel> refluxComms_;

        // Ghost particle injection strategy (for levelGhostParticlesOld)
        ParticleInjectionStrategy ghostParticleInjectionStrategy_;

        // Ghost fill algos + schedule maps (one per fill concern)
        SAMRAI::xfer::RefineAlgorithm magGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> magGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm eGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> eGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm currentGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> currentGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm ionMomentGhostAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> ionMomentGhostSchedules_;
        SAMRAI::xfer::RefineAlgorithm densityBorderAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> densityBorderSchedules_;
        SAMRAI::xfer::RefineAlgorithm fluxBorderAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> fluxBorderSchedules_;
        SAMRAI::xfer::RefineAlgorithm ionBorderAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> ionBorderSchedules_;
        SAMRAI::xfer::RefineAlgorithm ghostParticleAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> ghostParticleSchedules_;

        void registerGhostComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo,
                                  std::unique_ptr<HybridMessengerInfo> const& hybridInfo)
        {
            // TODO(Phase3): B ghost fill (MHD→Hybrid) — restore as same-type after PhysicalQuantity merge.
            // magGhostAlgo_.registerRefine(*hyb_b_id, *mhd_b_id, *mhd_b_id, <BRefineOp>, nonOverwrite)

            // TODO(Phase3): E ghost fill (MHD→Hybrid) — restore as same-type after PhysicalQuantity merge.
            // eGhostAlgo_.registerRefine(*hyb_e_id, *mhd_e_id, *mhd_e_id, <ERefineOp>, nonOverwrite)

            // TODO(Phase3): J ghost fill (MHD→Hybrid) — restore as same-type after PhysicalQuantity merge.
            // currentGhostAlgo_.registerRefine(*hyb_j_id, *mhd_j_id, *mhd_j_id, <ERefineOp>, nonOverwrite)

            // TODO(Phase3): ion moment ghost fills (MHD hydro ddd → Hybrid moments ppp) — restore after PhysicalQuantity merge.
            // ionMomentGhostAlgo_.registerRefine(*hyb_ni_id, *mhd_rho_id, ..., <HydroScalarRefineOp>, nonOverwrite)
            // ionMomentGhostAlgo_.registerRefine(*hyb_vi_id, *mhd_v_id,   ..., <HydroVecRefineOp>,    nonOverwrite)

            // TODO(Phase3): density/flux/ion border fills (MHD hydro → Hybrid population borders) — restore after PhysicalQuantity merge.
            // densityBorderAlgo_.registerRefine(*hyb_den_id, *mhd_rho_id, ...) for each sumBorderFields
            // fluxBorderAlgo_.registerRefine(*hyb_flux_id, *mhd_v_id, ...)     for each ghostFlux
            // ionBorderAlgo_.registerRefine(*hyb_ion_id, *mhd_rho_id/v_id, ...) for each maxBorderFields/maxBorderVecFields

            // Ghost particle injection: same-type MHD conservatives + ghostParticleInjectionStrategy_
            // (levelGhostParticlesOld IDs — same mechanism as domain injection, different data)
            // Uses conservatives: ghost particle schedules run during regrid before primitive updates.
            auto mhd_b_id    = mhdResourcesManager_->getID(mhdInfo->modelMagnetic);
            auto mhd_rho_id  = mhdResourcesManager_->getID(mhdInfo->modelDensity);
            auto mhd_rhoV_id = mhdResourcesManager_->getID(mhdInfo->modelMomentum);
            auto mhd_Etot_id = mhdResourcesManager_->getID(mhdInfo->modelTotalEnergy);
            if (!mhd_b_id or !mhd_rho_id or !mhd_rhoV_id or !mhd_Etot_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing MHD conservative IDs in registerGhostComms_");
            ghostParticleAlgo_.registerRefine(*mhd_rho_id,  *mhd_rho_id,  *mhd_rho_id,  mhdHydroRefineOp_);
            ghostParticleAlgo_.registerRefine(*mhd_rhoV_id, *mhd_rhoV_id, *mhd_rhoV_id, mhdVecHydroRefineOp_);
            ghostParticleAlgo_.registerRefine(*mhd_Etot_id, *mhd_Etot_id, *mhd_Etot_id, mhdHydroRefineOp_);

            ghostParticleInjectionStrategy_.registerMHDConsIds(*mhd_rho_id, *mhd_rhoV_id,
                                                               *mhd_b_id, *mhd_Etot_id,
                                                               getHeatCapacityRatio_());
            for (std::size_t i = 0; i < hybridInfo->levelGhostParticlesOld.size(); ++i)
            {
                auto part_id
                    = hybridResourcesManager_->getID(hybridInfo->levelGhostParticlesOld[i]);
                if (!part_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing ghost particle ID for population "
                        + hybridInfo->levelGhostParticlesOld[i]);
                ghostParticleInjectionStrategy_.addPopulation(*part_id, i);
            }
        }

        void registerInitComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo,
                                 std::unique_ptr<HybridMessengerInfo> const& hybridInfo)
        {
            // TODO(Phase3): B init fill (MHD→Hybrid) — restore as same-type after PhysicalQuantity merge.
            // magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            // magComms_.BalgoInit.registerRefine(*hyb_b_id, *mhd_b_id, *hyb_b_id, <BRefineOp>, overwrite)

            // TODO(Phase3): E init fill (MHD→Hybrid) — restore as same-type after PhysicalQuantity merge.
            // eInitComms_.algo.registerRefine(*hyb_e_id, *mhd_e_id, *hyb_e_id, <ERefineOp>, overwrite)

            // Particle injection: same-type MHD conservatives in initLevelAlgo_
            // Same-type ensures SAMRAI can allocate fields on d_coarse_interp_level so
            // postprocessRefine can read them. Conservatives not primitives: regrid runs
            // before the conservative-to-primitive conversion, so V/P may be stale.
            auto mhd_rho_id  = mhdResourcesManager_->getID(mhdInfo->modelDensity);
            auto mhd_rhoV_id = mhdResourcesManager_->getID(mhdInfo->modelMomentum);
            auto mhd_B_id    = mhdResourcesManager_->getID(mhdInfo->modelMagnetic);
            auto mhd_Etot_id = mhdResourcesManager_->getID(mhdInfo->modelTotalEnergy);
            if (!mhd_rho_id or !mhd_rhoV_id or !mhd_B_id or !mhd_Etot_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing MHD conservative IDs in registerInitComms_");
            initLevelAlgo_.registerRefine(*mhd_rho_id,  *mhd_rho_id,  *mhd_rho_id,  mhdHydroRefineOp_);
            initLevelAlgo_.registerRefine(*mhd_rhoV_id, *mhd_rhoV_id, *mhd_rhoV_id, mhdVecHydroRefineOp_);
            initLevelAlgo_.registerRefine(*mhd_Etot_id, *mhd_Etot_id, *mhd_Etot_id, mhdHydroRefineOp_);

            particleInjectionStrategy_.registerMHDConsIds(*mhd_rho_id, *mhd_rhoV_id, *mhd_B_id,
                                                          *mhd_Etot_id, getHeatCapacityRatio_());
            for (std::size_t i = 0; i < hybridInfo->interiorParticles.size(); ++i)
            {
                auto part_id = hybridResourcesManager_->getID(hybridInfo->interiorParticles[i]);
                if (!part_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing particle ID for population "
                        + hybridInfo->interiorParticles[i]);
                particleInjectionStrategy_.addPopulation(*part_id, i);
            }
        }

        void setupParticleInjection_(HybridModel& hybridModel, int levelNumber,
                                      ParticleInjectionStrategy& strategy,
                                      std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>>& schedules,
                                      double time)
        {
            setPopulationPhysicsFromIons_(hybridModel.state.ions, strategy);
            strategy.setLevelNumber(levelNumber);
            schedules.at(levelNumber)->fillData(time);
        }

        template<typename IonRange>
        static void setPopulationPhysicsFromIons_(IonRange const& ions,
                                                   ParticleInjectionStrategy& strategy)
        {
            std::size_t popIdx = 0;
            for (auto const& pop : ions)
            {
                auto const& info    = pop.particleInitializerInfo();
                double const charge = info["charge"].template to<double>();
                auto const nbrPPC
                    = static_cast<std::uint32_t>(info["nbr_part_per_cell"].template to<int>());
                strategy.setPopulationPhysics(popIdx, charge, nbrPPC);
                ++popIdx;
            }
        }

        static double getHeatCapacityRatio_()
        {
            auto const& simAlgo
                = PHARE::initializer::PHAREDictHandler::INSTANCE().dict()["simulation"]["algo"];
            return simAlgo["heat_capacity_ratio"].template to<double>();
        }
    };

} // namespace amr
} // namespace PHARE
#endif
