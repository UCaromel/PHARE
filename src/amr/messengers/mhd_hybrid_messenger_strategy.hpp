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

            // --- B init fills: cross-type MHD→Hybrid, overwrite interior ---
            // BfieldComms provides div-correction via MagneticRefinePatchStrategy.
            // Same schedules re-used for firstStep/prepareStep/fillMagneticGhosts.
            auto mhd_b_id = mhdResourcesManager_->getID(mhdInfo->modelMagnetic);
            auto hyb_b_id = hybridResourcesManager_->getID(hybridInfo->modelMagnetic);
            if (!mhd_b_id or !hyb_b_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing B IDs in registerQuantities");

            magComms_.magneticRefinePatchStrategy_.registerIDs(*hyb_b_id);
            magComms_.BalgoInit.registerRefine(*hyb_b_id, *mhd_b_id, *hyb_b_id, crossBRefineOp_,
                                               overwriteInteriorTFfillPattern_);

            // --- E init fills: cross-type MHD→Hybrid, overwrite interior ---
            auto mhd_e_id = mhdResourcesManager_->getID(mhdInfo->modelElectric);
            auto hyb_e_id = hybridResourcesManager_->getID(hybridInfo->modelElectric);
            if (!mhd_e_id or !hyb_e_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing E IDs in registerQuantities");

            eInitComms_.algo.registerRefine(*hyb_e_id, *mhd_e_id, *hyb_e_id, crossERefineOp_,
                                            overwriteInteriorTFfillPattern_);

            // --- Particle injection: same-type MHD hydro in initLevelAlgo_ ---
            // Same-type ensures SAMRAI can allocate rho/V/P on d_coarse_interp_level so
            // postprocessRefine can read them. B/E already filled above before this fires.
            auto mhd_rho_id = mhdResourcesManager_->getID(mhdInfo->modelDensity);
            auto mhd_v_id   = mhdResourcesManager_->getID(mhdInfo->modelVelocity);
            auto mhd_p_id   = mhdResourcesManager_->getID(mhdInfo->modelPressure);
            if (!mhd_rho_id or !mhd_v_id or !mhd_p_id)
                throw std::runtime_error(
                    "MHDHybridMessengerStrategy: missing MHD hydro IDs in registerQuantities");

            initLevelAlgo_.registerRefine(*mhd_rho_id, *mhd_rho_id, *mhd_rho_id, mhdHydroRefineOp_);
            initLevelAlgo_.registerRefine(*mhd_v_id, *mhd_v_id, *mhd_v_id, mhdVecHydroRefineOp_);
            initLevelAlgo_.registerRefine(*mhd_p_id, *mhd_p_id, *mhd_p_id, mhdHydroRefineOp_);

            particleInjectionStrategy_.registerMHDPrimIds(*mhd_rho_id, *mhd_v_id, *mhd_p_id);
            for (std::size_t i = 0; i < hybridInfo->interiorParticles.size(); ++i)
            {
                auto part_id = hybridResourcesManager_->getID(hybridInfo->interiorParticles[i]);
                if (!part_id)
                    throw std::runtime_error(
                        "MHDHybridMessengerStrategy: missing particle ID for population "
                        + hybridInfo->interiorParticles[i]);
                particleInjectionStrategy_.addPopulation(*part_id, i);
            }

            // --- Reflux: Hybrid flux sums → MHD timeFluxes (via refluxComms_) ---
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
            }
        }

        std::unique_ptr<IMessengerInfo> emptyInfoFromCoarser() override
        { return std::make_unique<MHDMessengerInfo>(); }

        std::unique_ptr<IMessengerInfo> emptyInfoFromFiner() override
        { return std::make_unique<HybridMessengerInfo>(); }

        std::string fineModelName() const override { return HybridModel::model_name; }
        std::string coarseModelName() const override { return MHDModel::model_name; }

        virtual ~MHDHybridMessengerStrategy() = default;

        void firstStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& level,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& /*hierarchy*/,
                       double const currentTime, double const /*prevCoarserTime*/,
                       double const /*newCoarserTime*/) override
        {
            int const lvl = level.getLevelNumber();
            if (magComms_.magInitRefineSchedules_.count(lvl))
                magComms_.magInitRefineSchedules_.at(lvl)->fillData(currentTime);
            eInitComms_.fill(lvl, currentTime);
        }

        void lastStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/) override {}

        void prepareStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& level,
                         double currentTime) final
        {
            int const lvl = level.getLevelNumber();
            if (magComms_.magInitRefineSchedules_.count(lvl))
                magComms_.magInitRefineSchedules_.at(lvl)->fillData(currentTime);
            eInitComms_.fill(lvl, currentTime);
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
            int const lvl = level.getLevelNumber();
            if (magComms_.magInitRefineSchedules_.count(lvl))
                magComms_.magInitRefineSchedules_.at(lvl)->fillData(initDataTime);
            eInitComms_.fill(lvl, initDataTime);

            if (initLevelSchedules_.count(lvl))
            {
                auto& hybridModel  = dynamic_cast<HybridModel&>(model);
                std::size_t popIdx = 0;
                for (auto const& pop : hybridModel.state.ions)
                {
                    auto const& info    = pop.particleInitializerInfo();
                    double const charge = info["charge"].template to<double>();
                    auto const nbrPPC
                        = static_cast<std::uint32_t>(info["nbr_part_per_cell"].template to<int>());
                    particleInjectionStrategy_.setPopulationPhysics(popIdx, charge, nbrPPC);
                    ++popIdx;
                }
                particleInjectionStrategy_.setLevelNumber(lvl);
                initLevelSchedules_.at(lvl)->fillData(initDataTime);
            }
        }

        // regrid: SAMRAI restricts postprocessRefine to genuinely new boxes only.
        void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& /*hierarchy*/,
                    int const levelNumber,
                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& /*oldLevel*/,
                    IPhysicalModel& model, double const initDataTime) override
        {
            if (magComms_.magInitRefineSchedules_.count(levelNumber))
                magComms_.magInitRefineSchedules_.at(levelNumber)->fillData(initDataTime);
            eInitComms_.fill(levelNumber, initDataTime);

            if (initLevelSchedules_.count(levelNumber))
            {
                auto& hybridModel  = dynamic_cast<HybridModel&>(model);
                std::size_t popIdx = 0;
                for (auto const& pop : hybridModel.state.ions)
                {
                    auto const& info    = pop.particleInitializerInfo();
                    double const charge = info["charge"].template to<double>();
                    auto const nbrPPC
                        = static_cast<std::uint32_t>(info["nbr_part_per_cell"].template to<int>());
                    particleInjectionStrategy_.setPopulationPhysics(popIdx, charge, nbrPPC);
                    ++popIdx;
                }
                particleInjectionStrategy_.setLevelNumber(levelNumber);
                initLevelSchedules_.at(levelNumber)->fillData(initDataTime);
            }
        }

        void fillMagneticGhosts(VecFieldT& B, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(B, level, *hybridResourcesManager_);
            int const lvl = level.getLevelNumber();
            if (magComms_.magInitRefineSchedules_.count(lvl))
                magComms_.magInitRefineSchedules_.at(lvl)->fillData(fillTime);
        }

        void fillElectricGhosts(VecFieldT& E, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            setNaNsOnVecfieldGhosts<HybridGridLayoutT>(E, level, *hybridResourcesManager_);
            eInitComms_.fill(level.getLevelNumber(), fillTime);
        }

        void fillCurrentGhosts(VecFieldT& /*J*/, SAMRAI::hier::PatchLevel const& /*level*/,
                               double const /*fillTime*/) override
        {
        }

        void fillIonGhostParticles(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                                   double const /*fillTime*/) override
        {
        }
        void fillIonPopMomentGhosts(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                                    double const /*fillTime*/) override
        {
        }
        void fillFluxBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                             double const /*fillTime*/) override
        {
        }
        void fillDensityBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                                double const /*fillTime*/) override
        {
        }
        void fillIonBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                            double const /*fillTime*/) override
        {
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
    };

} // namespace amr
} // namespace PHARE
#endif
