#ifndef PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP
#define PHARE_MHD_HYBRID_MESSENGER_STRATEGY_HPP

#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/mhd_hybrid_particle_injection_patch_strategy.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/mhd_flux_coarsener.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/magnetic_field_refiner.hpp"
#include "amr/data/field/refine/electric_field_refiner.hpp"

#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefineSchedule.h>
#include <SAMRAI/xfer/CoarsenAlgorithm.h>
#include <SAMRAI/xfer/CoarsenSchedule.h>

#include <string>
#include <memory>
#include <unordered_map>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel, typename HybridModel>
    class MHDHybridMessengerStrategy : public HybridMessengerStrategy<HybridModel>
    {
        static constexpr std::size_t dimension = HybridModel::dimension;

        using IonsT          = decltype(std::declval<HybridModel>().state.ions);
        using VecFieldT      = decltype(std::declval<HybridModel>().state.electromag.E);
        using IPhysicalModel = typename HybridModel::Interface;

        using HybridGridLayoutT = HybridModel::gridlayout_type;
        using HybridGridT       = HybridModel::grid_type;
        using MHDGridLayoutT    = MHDModel::gridlayout_type;
        using MHDGridT          = MHDModel::grid_type;

        using HybridRMType = HybridModel::resources_manager_type;
        using MHDRMType    = MHDModel::resources_manager_type;

        // Cross-type coarsen op: Hybrid flux sum VecFields → MHD receiver VecFields.
        // Types differ across the model boundary; centering is per-direction (VecFlux_x/y/z).
        using HybridFluxCoarsenOp
            = CrossTypeVecFieldCoarsenOperator<HybridGridLayoutT, HybridGridT, core::HybridQuantity,
                                               MHDGridLayoutT, MHDGridT, core::MHDQuantity,
                                               MHDFluxCoarsener<dimension>>;

        // Cross-type coarsen op: Hybrid scalar flux Fields → MHD scalar receiver Fields.
        // Used for rho and Etot (one Field per face direction, ScalarFlux_x/y/z centering).
        using HybridScalarFluxCoarsenOp
            = CrossTypeScalarFieldCoarsenOperator<HybridGridLayoutT, typename HybridModel::field_type,
                                                  MHDGridLayoutT, typename MHDModel::field_type,
                                                  MHDFluxCoarsener<dimension>>;

        // Refine ops (MHD→Hybrid): B and E staggering identical between models
        template<typename Policy>
        using VecFieldRefineOp = VecFieldRefineOperator<MHDGridLayoutT, MHDGridT, Policy>;
        using BRefineOp = VecFieldRefineOp<MagneticFieldRefiner<dimension>>;
        using ERefineOp = VecFieldRefineOp<ElectricFieldRefiner<dimension>>;

        using ParticleInjectionStrategy
            = MHDHybridParticleInjectionPatchStrategy<MHDModel, HybridModel>;

    public:
        static inline std::string const stratName = "MHDModel-HybridModel";

        MHDHybridMessengerStrategy(std::shared_ptr<MHDRMType> const& mhdResourcesManager,
                                   std::shared_ptr<HybridRMType> const& hybridResourcesManager,
                                   int const firstLevel)
            : HybridMessengerStrategy<HybridModel>{stratName}
            , mhdResourcesManager_{mhdResourcesManager}
            , hybridResourcesManager_{hybridResourcesManager}
            , firstLevel_{firstLevel}
        {
            hybridResourcesManager_->registerResources(EM_old_);
        }

        void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override
        {
            hybridResourcesManager_->allocate(EM_old_, patch, allocateTime);
        }

        /**
         * @brief Set up SAMRAI refine/coarsen algorithms for all cross-model communications.
         *
         * Refine algorithms (MHD coarser → Hybrid finer, static fill):
         *   - B: MagneticFieldRefineOp (face staggering identical between models)
         *   - E: ElectricFieldRefineOp (edge staggering identical)
         *   - rho, V: cell-centered linear (for particle injection context)
         *
         * Coarsen algorithms (Hybrid finer → MHD coarser):
         *   - fluxSumRho, fluxSumRhoV, fluxSumEtot, fluxSumE: HybridFluxCoarsenOp
         */
        void registerQuantities(
            std::unique_ptr<IMessengerInfo> fromCoarserInfo,
            std::unique_ptr<IMessengerInfo> fromFinerInfo) override
        {
            std::unique_ptr<MHDMessengerInfo> mhdInfo{
                dynamic_cast<MHDMessengerInfo*>(fromCoarserInfo.release())};
            std::unique_ptr<HybridMessengerInfo> hybridInfo{
                dynamic_cast<HybridMessengerInfo*>(fromFinerInfo.release())};

            // --- Refine: MHD B → Hybrid B (static, for ghost fills and init) ---
            auto&& [mhd_b_id]    = mhdResourcesManager_->getIDsList(mhdInfo->modelMagnetic);
            auto&& [hyb_b_id]    = hybridResourcesManager_->getIDsList(hybridInfo->modelMagnetic);
            auto&& [em_old_b_id] = hybridResourcesManager_->getIDsList(EM_old_.B.name());

            BInitAlgo_.registerRefine(hyb_b_id, mhd_b_id, hyb_b_id, BRefineOp_);
            BEMOldAlgo_.registerRefine(em_old_b_id, mhd_b_id, em_old_b_id, BRefineOp_);

            // --- Refine: MHD E → Hybrid E (static) ---
            auto&& [mhd_e_id]    = mhdResourcesManager_->getIDsList(mhdInfo->modelElectric);
            auto&& [hyb_e_id]    = hybridResourcesManager_->getIDsList(hybridInfo->modelElectric);
            auto&& [em_old_e_id] = hybridResourcesManager_->getIDsList(EM_old_.E.name());

            EInitAlgo_.registerRefine(hyb_e_id, mhd_e_id, hyb_e_id, ERefineOp_);
            EEMOldAlgo_.registerRefine(em_old_e_id, mhd_e_id, em_old_e_id, ERefineOp_);

            // --- Refine: MHD rho, V → Hybrid (cell-centered, particle injection context) ---
            // TODO: register rho and V refine algorithms once the Hybrid side exposes the
            // corresponding data IDs in HybridMessengerInfo for the MHD-Hybrid context.

            // --- Coarsen: Hybrid flux sums → MHD receivers ---
            // Per-direction registrations (9 total: 6 scalar + 3 VecField), with dimension guards.
            // Scalar (rho, Etot): CrossTypeScalarFieldCoarsenOperator
            // Vector (rhoV): CrossTypeVecFieldCoarsenOperator (all 3 momentum components at same face)

            // x-face (always)
            {
                auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumRho_fx);
                auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumRho_fx);
                FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, scalarFluxCoarsenOp_);
            }
            {
                auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumRhoV_fx);
                auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumRhoV_fx);
                FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, fluxCoarsenOp_);
            }
            {
                auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumEtot_fx);
                auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumEtot_fx);
                FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, scalarFluxCoarsenOp_);
            }

            if constexpr (dimension >= 2)
            {
                {
                    auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumRho_fy);
                    auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumRho_fy);
                    FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, scalarFluxCoarsenOp_);
                }
                {
                    auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumRhoV_fy);
                    auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumRhoV_fy);
                    FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, fluxCoarsenOp_);
                }
                {
                    auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumEtot_fy);
                    auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumEtot_fy);
                    FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, scalarFluxCoarsenOp_);
                }

                if constexpr (dimension == 3)
                {
                    {
                        auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumRho_fz);
                        auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumRho_fz);
                        FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, scalarFluxCoarsenOp_);
                    }
                    {
                        auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumRhoV_fz);
                        auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumRhoV_fz);
                        FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, fluxCoarsenOp_);
                    }
                    {
                        auto&& [hyb_id] = hybridResourcesManager_->getIDsList(hybridInfo->fluxSumEtot_fz);
                        auto&& [mhd_id] = mhdResourcesManager_->getIDsList(mhdInfo->hybridFluxSumEtot_fz);
                        FluxCoarsenAlgo_.registerCoarsen(mhd_id, hyb_id, scalarFluxCoarsenOp_);
                    }
                }
            }

            // Store IDs needed by firstStep/prepareStep
            mhdBId_  = mhd_b_id;
            mhdEId_  = mhd_e_id;
            hybBId_  = hyb_b_id;
            hybEId_  = hyb_e_id;
        }

        void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                           int const levelNumber) override
        {
            auto const level = hierarchy->getPatchLevel(levelNumber);

            if (levelNumber > 0)
            {
                BInitSchedules_[levelNumber]
                    = BInitAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                EInitSchedules_[levelNumber]
                    = EInitAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                BEMOldSchedules_[levelNumber]
                    = BEMOldAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
                EEMOldSchedules_[levelNumber]
                    = EEMOldAlgo_.createSchedule(level, levelNumber - 1, hierarchy);
            }

            FluxCoarsenSchedules_[levelNumber]
                = FluxCoarsenAlgo_.createSchedule(hierarchy->getPatchLevel(levelNumber - 1), level);
        }

        std::unique_ptr<IMessengerInfo> emptyInfoFromCoarser() override
        {
            return std::make_unique<MHDMessengerInfo>();
        }

        std::unique_ptr<IMessengerInfo> emptyInfoFromFiner() override
        {
            return std::make_unique<HybridMessengerInfo>();
        }

        std::string fineModelName() const override { return HybridModel::model_name; }
        std::string coarseModelName() const override { return MHDModel::model_name; }

        virtual ~MHDHybridMessengerStrategy() = default;

        /**
         * @brief Snapshot MHD EM into EM_old_ and do static ghost fill from MHD t^n state.
         * E ghost cells are frozen at t^n for the entire Hybrid subcycle.
         * B ghost cells are refilled statically in prepareStep() from current MHD B.
         */
        void firstStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& level,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& /*hierarchy*/,
                       double const currentTime, double const /*prevCoarserTime*/,
                       double const /*newCoarserTime*/) override
        {
            int const lvl = level.getLevelNumber();

            // Snapshot MHD B and E into EM_old_ (t^n)
            if (BEMOldSchedules_.count(lvl))
                BEMOldSchedules_.at(lvl)->fillData(currentTime);
            if (EEMOldSchedules_.count(lvl))
                EEMOldSchedules_.at(lvl)->fillData(currentTime);

            // Fill Hybrid B ghost cells from MHD (static, t^n)
            if (BInitSchedules_.count(lvl))
                BInitSchedules_.at(lvl)->fillData(currentTime);

            // Fill Hybrid E ghost cells from MHD (static, frozen at t^n for entire subcycle)
            if (EInitSchedules_.count(lvl))
                EInitSchedules_.at(lvl)->fillData(currentTime);
        }

        void lastStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/) override {}

        /**
         * @brief Re-fill B ghost cells from current MHD B (static, matches MHD/Hybrid convention).
         * E ghost cells remain at t^n snapshot set in firstStep() — not updated here.
         */
        void prepareStep(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& level,
                         double currentTime) final
        {
            int const lvl = level.getLevelNumber();
            // B is static refined from MHD current state (no time interpolation)
            if (BInitSchedules_.count(lvl))
                BInitSchedules_.at(lvl)->fillData(currentTime);
            // E ghost cells remain at frozen t^n — no fill here
        }

        void fillRootGhosts(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                            double const /*initDataTime*/) final
        {
            // Root level is MHD-only; no cross-model root ghost fill needed
        }

        /**
         * @brief Fill B and E on new Hybrid patches from MHD, then inject particles.
         */
        void initLevel(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& level,
                       double const initDataTime) override
        {
            int const lvl = level.getLevelNumber();
            if (BInitSchedules_.count(lvl))
                BInitSchedules_.at(lvl)->fillData(initDataTime);
            if (EInitSchedules_.count(lvl))
                EInitSchedules_.at(lvl)->fillData(initDataTime);

            // TODO: execute rho/V refine schedules (particle injection context)
            // TODO: invoke particleInjectionStrategy_ for each patch
        }

        void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& /*hierarchy*/,
                    int const /*levelNumber*/,
                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& /*oldLevel*/,
                    IPhysicalModel& model, double const initDataTime) override
        {
            // SAMRAI restricts postprocessRefine to truly new boxes automatically
            // TODO: same as initLevel once rho/V refine + particle injection are implemented
        }

        void fillMagneticGhosts(VecFieldT& /*B*/, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            int const lvl = level.getLevelNumber();
            if (BInitSchedules_.count(lvl))
                BInitSchedules_.at(lvl)->fillData(fillTime);
        }

        void fillElectricGhosts(VecFieldT& /*E*/, SAMRAI::hier::PatchLevel const& level,
                                double const fillTime) override
        {
            int const lvl = level.getLevelNumber();
            if (EInitSchedules_.count(lvl))
                EInitSchedules_.at(lvl)->fillData(fillTime);
        }

        void fillCurrentGhosts(VecFieldT& /*J*/, SAMRAI::hier::PatchLevel const& /*level*/,
                               double const /*fillTime*/) override
        {
            // MHD does not provide J to Hybrid
        }

        void fillIonGhostParticles(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                                   double const /*fillTime*/) override {}
        void fillIonPopMomentGhosts(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                                    double const /*fillTime*/) override {}
        void fillFluxBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                             double const /*fillTime*/) override {}
        void fillDensityBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                                double const /*fillTime*/) override {}
        void fillIonBorders(IonsT& /*ions*/, SAMRAI::hier::PatchLevel& /*level*/,
                            double const /*fillTime*/) override {}

        /**
         * @brief No moment coarsening needed — global reflux pipeline handles conservation.
         */
        void synchronize(SAMRAI::hier::PatchLevel& /*level*/) final {}

        /**
         * @brief Coarsen all Hybrid flux sums globally to MHD, then delegate to SolverMHD::reflux.
         */
        void reflux(int const /*coarserLevelNumber*/, int const fineLevelNumber,
                    double const syncTime) override
        {
            // 1. Coarsen Hybrid flux sum VecFields → MHD receiver VecFields (keyed by fine level)
            if (FluxCoarsenSchedules_.count(fineLevelNumber))
                FluxCoarsenSchedules_.at(fineLevelNumber)->coarsenData();

            // 2. TODO: copy hybridFluxSumRho/RhoV/Etot per-patch into MHD fluxSum_ scalars,
            //    then delegate to SolverMHD::reflux() to retake MHD step with corrected fluxes.
            (void)syncTime;
        }

        void postSynchronize(IPhysicalModel& /*model*/, SAMRAI::hier::PatchLevel& /*level*/,
                             double const /*time*/) override
        {
            // TODO: refill MHD patch ghosts from corrected state via MHD refine schedules
        }

    private:
        using Electromag = decltype(std::declval<HybridModel>().state.electromag);

        std::shared_ptr<MHDRMType> mhdResourcesManager_;
        std::shared_ptr<HybridRMType> hybridResourcesManager_;
        int const firstLevel_;

        Electromag EM_old_{stratName + "_EM_old"};

        // Refine operators (MHD→Hybrid, static fill)
        std::shared_ptr<BRefineOp> BRefineOp_ = std::make_shared<BRefineOp>();
        std::shared_ptr<ERefineOp> ERefineOp_ = std::make_shared<ERefineOp>();

        // Coarsen operators (Hybrid flux sums→MHD)
        std::shared_ptr<HybridFluxCoarsenOp>       fluxCoarsenOp_
            = std::make_shared<HybridFluxCoarsenOp>();
        std::shared_ptr<HybridScalarFluxCoarsenOp> scalarFluxCoarsenOp_
            = std::make_shared<HybridScalarFluxCoarsenOp>();

        // Particle injection strategy (passed to SAMRAI refine algorithm for initLevel/regrid)
        ParticleInjectionStrategy particleInjectionStrategy_;

        // SAMRAI algorithms
        SAMRAI::xfer::RefineAlgorithm BInitAlgo_;
        SAMRAI::xfer::RefineAlgorithm EInitAlgo_;
        SAMRAI::xfer::RefineAlgorithm BEMOldAlgo_;
        SAMRAI::xfer::RefineAlgorithm EEMOldAlgo_;
        SAMRAI::xfer::CoarsenAlgorithm FluxCoarsenAlgo_{SAMRAI::tbox::Dimension{dimension}};

        // SAMRAI schedules (per level number)
        std::unordered_map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> BInitSchedules_;
        std::unordered_map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> EInitSchedules_;
        std::unordered_map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> BEMOldSchedules_;
        std::unordered_map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> EEMOldSchedules_;
        std::unordered_map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> FluxCoarsenSchedules_;

        // Cached patch data IDs (set during registerQuantities)
        int mhdBId_ = -1;
        int mhdEId_ = -1;
        int hybBId_ = -1;
        int hybEId_ = -1;
    };


} // namespace amr
} // namespace PHARE
#endif
