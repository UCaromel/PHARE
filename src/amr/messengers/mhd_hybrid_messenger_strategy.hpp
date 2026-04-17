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
#include "amr/data/field/refine/hydro_moments_refiner.hpp"
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

        // Cross-type refine ops: MHD B/E → Hybrid B/E
        using CrossBRefineOp
            = CrossTypeVecFieldRefineOperator<MHDGridLayoutT, MHDGridT, core::MHDQuantity,
                                              HybridGridLayoutT, HybridGridT, core::HybridQuantity,
                                              MagneticFieldRefiner<dimension>>;
        using CrossERefineOp
            = CrossTypeVecFieldRefineOperator<MHDGridLayoutT, MHDGridT, core::MHDQuantity,
                                              HybridGridLayoutT, HybridGridT, core::HybridQuantity,
                                              ElectricFieldRefiner<dimension>>;

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
        using FieldFillPattern_t       = FieldFillPattern<dimension>;

        // Cross-type ops for Hybrid ghost fills (MHD hydro → Hybrid)
        using CrossJRefineOp = CrossERefineOp; // J has same edge centering as E
        using CrossHydroScalarRefineOp
            = CrossTypeScalarFieldRefineOperator<MHDGridLayoutT, typename MHDModel::field_type,
                                                 HybridGridLayoutT, typename HybridModel::field_type,
                                                 HydroMomentsRefiner<dimension>>;
        using CrossHydroVecRefineOp
            = CrossTypeVecFieldRefineOperator<MHDGridLayoutT, MHDGridT, core::MHDQuantity,
                                              HybridGridLayoutT, HybridGridT, core::HybridQuantity,
                                              HydroMomentsRefiner<dimension>>;

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
                    = magGhostAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy,
                                                   &magComms_.magneticRefinePatchStrategy_);
                eGhostSchedules_[levelNumber]
                    = eGhostAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy);
                currentGhostSchedules_[levelNumber]
                    = currentGhostAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy);
                ionMomentGhostSchedules_[levelNumber]
                    = ionMomentGhostAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy);
                densityBorderSchedules_[levelNumber]
                    = densityBorderAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy);
                fluxBorderSchedules_[levelNumber]
                    = fluxBorderAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy);
                ionBorderSchedules_[levelNumber]
                    = ionBorderAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy);
                ghostParticleSchedules_[levelNumber]
                    = ghostParticleAlgo_.createSchedule(level, nullptr, levelNumber - 1, hierarchy,
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

        // Cross-type refine ops (MHD→Hybrid)
        std::shared_ptr<CrossBRefineOp> crossBRefineOp_{std::make_shared<CrossBRefineOp>()};
        std::shared_ptr<CrossERefineOp> crossERefineOp_{std::make_shared<CrossERefineOp>()};

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

        // Cross-type hydro refine ops (MHD scalar/vec → Hybrid scalar/vec)
        std::shared_ptr<CrossHydroScalarRefineOp> crossHydroScalarRefineOp_{
            std::make_shared<CrossHydroScalarRefineOp>()};
        std::shared_ptr<CrossHydroVecRefineOp> crossHydroVecRefineOp_{
            std::make_shared<CrossHydroVecRefineOp>()};

        // Scalar fill pattern (for FieldData ghost fills)
        std::shared_ptr<FieldFillPattern_t> nonOverwriteInteriorFieldFillPattern_{
            std::make_shared<FieldFillPattern_t>()};

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
            // B ghost: nonOverwrite, reuses magneticRefinePatchStrategy_ (IDs registered in registerInitComms_)
            auto mhd_b_id = mhdResourcesManager_->getID(mhdInfo->modelMagnetic);
            auto hyb_b_id = hybridResourcesManager_->getID(hybridInfo->modelMagnetic);
            if (!mhd_b_id or !hyb_b_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing B IDs in registerGhostComms_");
            magGhostAlgo_.registerRefine(*hyb_b_id, *mhd_b_id, *mhd_b_id, crossBRefineOp_,
                                         nonOverwriteInteriorTFfillPattern_);

            // E ghost
            auto mhd_e_id = mhdResourcesManager_->getID(mhdInfo->modelElectric);
            auto hyb_e_id = hybridResourcesManager_->getID(hybridInfo->modelElectric);
            if (!mhd_e_id or !hyb_e_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing E IDs in registerGhostComms_");
            eGhostAlgo_.registerRefine(*hyb_e_id, *mhd_e_id, *mhd_e_id, crossERefineOp_,
                                       nonOverwriteInteriorTFfillPattern_);

            // J ghost (same edge centering as E)
            auto mhd_j_id = mhdResourcesManager_->getID(mhdInfo->modelCurrent);
            auto hyb_j_id = hybridResourcesManager_->getID(hybridInfo->modelCurrent);
            if (!mhd_j_id or !hyb_j_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing J IDs in registerGhostComms_");
            currentGhostAlgo_.registerRefine(*hyb_j_id, *mhd_j_id, *mhd_j_id, crossERefineOp_,
                                             nonOverwriteInteriorTFfillPattern_);

            // Ion moment ghost: MHD density → Hybrid ni (scalar); MHD velocity → Hybrid Vi (vec)
            auto mhd_rho_id = mhdResourcesManager_->getID(mhdInfo->modelDensity);
            auto hyb_ni_id  = hybridResourcesManager_->getID(hybridInfo->modelIonDensity);
            auto mhd_v_id   = mhdResourcesManager_->getID(mhdInfo->modelVelocity);
            auto hyb_vi_id  = hybridResourcesManager_->getID(hybridInfo->modelIonBulkVelocity);
            if (!mhd_rho_id or !hyb_ni_id or !mhd_v_id or !hyb_vi_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing hydro IDs in registerGhostComms_");
            ionMomentGhostAlgo_.registerRefine(*hyb_ni_id, *mhd_rho_id, *mhd_rho_id,
                                               crossHydroScalarRefineOp_,
                                               nonOverwriteInteriorFieldFillPattern_);
            ionMomentGhostAlgo_.registerRefine(*hyb_vi_id, *mhd_v_id, *mhd_v_id,
                                               crossHydroVecRefineOp_,
                                               nonOverwriteInteriorTFfillPattern_);

            // Density border: MHD density → each Hybrid population density border field
            for (auto const& name : hybridInfo->sumBorderFields)
            {
                auto hyb_den_id = hybridResourcesManager_->getID(name);
                if (!hyb_den_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing density border ID: " + name);
                densityBorderAlgo_.registerRefine(*hyb_den_id, *mhd_rho_id, *mhd_rho_id,
                                                  crossHydroScalarRefineOp_,
                                                  nonOverwriteInteriorFieldFillPattern_);
            }

            // Flux border: MHD velocity → each Hybrid population flux border field
            for (auto const& name : hybridInfo->ghostFlux)
            {
                auto hyb_flux_id = hybridResourcesManager_->getID(name);
                if (!hyb_flux_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing flux border ID: " + name);
                fluxBorderAlgo_.registerRefine(*hyb_flux_id, *mhd_v_id, *mhd_v_id,
                                              crossHydroVecRefineOp_,
                                              nonOverwriteInteriorTFfillPattern_);
            }

            // Ion border: MHD density → maxBorderFields (scalar); MHD velocity → maxBorderVecFields
            for (auto const& name : hybridInfo->maxBorderFields)
            {
                auto hyb_ion_id = hybridResourcesManager_->getID(name);
                if (!hyb_ion_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing ion border scalar ID: " + name);
                ionBorderAlgo_.registerRefine(*hyb_ion_id, *mhd_rho_id, *mhd_rho_id,
                                             crossHydroScalarRefineOp_,
                                             nonOverwriteInteriorFieldFillPattern_);
            }
            for (auto const& name : hybridInfo->maxBorderVecFields)
            {
                auto hyb_ion_id = hybridResourcesManager_->getID(name);
                if (!hyb_ion_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing ion border vec ID: " + name);
                ionBorderAlgo_.registerRefine(*hyb_ion_id, *mhd_v_id, *mhd_v_id,
                                             crossHydroVecRefineOp_,
                                             nonOverwriteInteriorTFfillPattern_);
            }

            // Ghost particle injection: same-type MHD conservatives + ghostParticleInjectionStrategy_
            // (levelGhostParticlesOld IDs — same mechanism as domain injection, different data)
            // Uses conservatives: ghost particle schedules run during regrid before primitive updates.
            auto mhd_rhoV_id = mhdResourcesManager_->getID(mhdInfo->modelMomentum);
            auto mhd_Etot_id = mhdResourcesManager_->getID(mhdInfo->modelTotalEnergy);
            if (!mhd_rhoV_id or !mhd_Etot_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing MHD rhoV/Etot IDs in registerGhostComms_");
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
            // B init fills: cross-type MHD→Hybrid, overwrite interior
            auto mhd_b_id = mhdResourcesManager_->getID(mhdInfo->modelMagnetic);
            auto hyb_b_id = hybridResourcesManager_->getID(hybridInfo->modelMagnetic);
            if (!mhd_b_id or !hyb_b_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing B IDs in registerInitComms_");
            magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            magComms_.BalgoInit.registerRefine(*hyb_b_id, *mhd_b_id, *hyb_b_id, crossBRefineOp_,
                                               overwriteInteriorTFfillPattern_);

            // E init fills: cross-type MHD→Hybrid, overwrite interior
            auto mhd_e_id = mhdResourcesManager_->getID(mhdInfo->modelElectric);
            auto hyb_e_id = hybridResourcesManager_->getID(hybridInfo->modelElectric);
            if (!mhd_e_id or !hyb_e_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing E IDs in registerInitComms_");
            eInitComms_.algo.registerRefine(*hyb_e_id, *mhd_e_id, *hyb_e_id, crossERefineOp_,
                                            overwriteInteriorTFfillPattern_);

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
