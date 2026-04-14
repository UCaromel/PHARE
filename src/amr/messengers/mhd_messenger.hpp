#ifndef PHARE_MHD_MESSENGER_HPP
#define PHARE_MHD_MESSENGER_HPP

#include "amr/data/field/coarsening/electric_field_coarsener.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
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
#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "amr/messengers/mhd_mhd/mhd_mhd_reflux_comms.hpp"

#include "core/mhd/mhd_quantities.hpp"

#include "SAMRAI/hier/CoarsenOperator.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/RefineOperator.h"

#include <memory>
#include <string>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel>
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

    public:
        static constexpr std::size_t rootLevelNumber = 0;
        static inline std::string const stratName    = "MHDModel-MHDModel";

        MHDMessenger(std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager,
                     int const firstLevel)
            : resourcesManager_{std::move(resourcesManager)}
            , firstLevel_{firstLevel}
        {
            // moment ghosts are primitive quantities
            resourcesManager_->registerResources(rhoOld_);
            resourcesManager_->registerResources(Vold_);
            resourcesManager_->registerResources(Pold_);

            resourcesManager_->registerResources(rhoVold_);
            resourcesManager_->registerResources(EtotOld_);

            resourcesManager_->registerResources(Jold_); // conditionally register

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

            resourcesManager_->allocate(Jold_, patch, allocateTime);
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

            magComms_.magneticRefinePatchStrategy_.registerIDs(*b_id);

            magComms_.BalgoInit.registerRefine(*b_id, *b_id, *b_id, BfieldRegridOp_,
                                     overwriteInteriorTFfillPattern);

            magComms_.BregridAlgo.registerRefine(*b_id, *b_id, *b_id, BfieldRegridOp_,
                                       overwriteInteriorTFfillPattern);

            auto e_id = resourcesManager_->getID(mhdInfo->modelElectric);

            if (!e_id)
            {
                throw std::runtime_error(
                    "MHDMessengerStrategy: missing electric field variable IDs");
            }

            mhdRefluxComms_.registerQuantities(*mhdInfo, *resourcesManager_, EfieldRefineOp_,
                                              electricFieldCoarseningOp_, mhdFluxRefineOp_,
                                              mhdVecFluxRefineOp_, nonOverwriteInteriorTFfillPattern);

            registerGhostComms_(mhdInfo);
            registerInitComms_(mhdInfo);
        }



        void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                           int const levelNumber) override
        {
            auto const level = hierarchy->getPatchLevel(levelNumber);

            mhdRefluxComms_.registerLevel(levelNumber, hierarchy);

            elecGhostsRefiners_.registerLevel(hierarchy, level);
            currentGhostsRefiners_.registerLevel(hierarchy, level);

            rhoGhostsRefiners_.registerLevel(hierarchy, level);

            momentumGhostsRefiners_.registerLevel(hierarchy, level);
            totalEnergyGhostsRefiners_.registerLevel(hierarchy, level);

            magFluxesXGhostRefiners_.registerLevel(hierarchy, level);
            magFluxesYGhostRefiners_.registerLevel(hierarchy, level);
            magFluxesZGhostRefiners_.registerLevel(hierarchy, level);

            magGhostsRefiners_.registerLevel(hierarchy, level);
            magMaxRefiners_.registerLevel(hierarchy, level);
            magMaxModelRefiners_.registerLevel(hierarchy, level);

            if (levelNumber != rootLevelNumber)
            {
                // refinement
                magComms_.magInitRefineSchedules_[levelNumber] = magComms_.BalgoInit.createSchedule(
                    level, nullptr, levelNumber - 1, hierarchy, &magComms_.magneticRefinePatchStrategy_);

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

            magComms_.magneticRegriding_(hierarchy, level, oldLevel, initDataTime);
            magMaxModelRefiners_.fill(mhdModel.state.B, level->getLevelNumber(), initDataTime);

            densityInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            momentumInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            totalEnergyInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);

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

            magComms_.magInitRefineSchedules_[levelNumber]->fillData(initDataTime);
            densityInitRefiners_.fill(levelNumber, initDataTime);
            momentumInitRefiners_.fill(levelNumber, initDataTime);
            totalEnergyInitRefiners_.fill(levelNumber, initDataTime);
        }

        void firstStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                       double const currentTime, double const prevCoarserTIme,
                       double const newCoarserTime) final
        {
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
                    mhdModel.state.rhoV, mhdModel.state.Etot, mhdModel.state.J, rhoOld_, Vold_,
                    Pold_, rhoVold_, EtotOld_, Jold_);

                resourcesManager_->setTime(rhoOld_, *patch, currentTime);
                resourcesManager_->setTime(Vold_, *patch, currentTime);
                resourcesManager_->setTime(Pold_, *patch, currentTime);
                resourcesManager_->setTime(rhoVold_, *patch, currentTime);
                resourcesManager_->setTime(EtotOld_, *patch, currentTime);
                resourcesManager_->setTime(Jold_, *patch, currentTime);

                rhoOld_.copyData(mhdModel.state.rho);
                Vold_.copyData(mhdModel.state.V);
                Pold_.copyData(mhdModel.state.P);
                rhoVold_.copyData(mhdModel.state.rhoV);
                EtotOld_.copyData(mhdModel.state.Etot);
                Jold_.copyData(mhdModel.state.J);
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
            mhdRefluxComms_.reflux(fineLevelNumber, coarserLevelNumber, syncTime);
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
            PHARE::amr::setNaNsOnFieldGhosts<GridLayoutT>(state.rho, level, *resourcesManager_);
            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(state.rhoV, level, *resourcesManager_);
            PHARE::amr::setNaNsOnFieldGhosts<GridLayoutT>(state.Etot, level, *resourcesManager_);
            rhoGhostsRefiners_.fill(state.rho, level.getLevelNumber(), fillTime);
            momentumGhostsRefiners_.fill(state.rhoV, level.getLevelNumber(), fillTime);
            totalEnergyGhostsRefiners_.fill(state.Etot, level.getLevelNumber(), fillTime);
        }

        void fillMagneticFluxesXGhosts(VecFieldT& Fx_B, level_t const& level, double const fillTime)
        {
            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(Fx_B, level, *resourcesManager_);
            magFluxesXGhostRefiners_.fill(Fx_B, level.getLevelNumber(), fillTime);
        }

        void fillMagneticFluxesYGhosts(VecFieldT& Fy_B, level_t const& level, double const fillTime)
        {
            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(Fy_B, level, *resourcesManager_);
            magFluxesYGhostRefiners_.fill(Fy_B, level.getLevelNumber(), fillTime);
        }

        void fillMagneticFluxesZGhosts(VecFieldT& Fz_B, level_t const& level, double const fillTime)
        {
            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(Fz_B, level, *resourcesManager_);
            magFluxesZGhostRefiners_.fill(Fz_B, level.getLevelNumber(), fillTime);
        }

        void fillElectricGhosts(VecFieldT& E, level_t const& level, double const fillTime)
        {
            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(E, level, *resourcesManager_);
            elecGhostsRefiners_.fill(E, level.getLevelNumber(), fillTime);
        }

        void fillMagneticGhosts(VecFieldT& B, level_t const& level, double const fillTime)
        {
            PHARE_LOG_SCOPE(3, "MHDMessenger::fillMagneticGhosts");

            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(B, level, *resourcesManager_);
            magGhostsRefiners_.fill(B, level.getLevelNumber(), fillTime);
            magMaxRefiners_.fill(B, level.getLevelNumber(), fillTime);
        }

        void fillCurrentGhosts(VecFieldT& J, level_t const& level, double const fillTime)
        {
            PHARE::amr::setNaNsOnVecfieldGhosts<GridLayoutT>(J, level, *resourcesManager_);
            currentGhostsRefiners_.fill(J, level.getLevelNumber(), fillTime);
        }

        std::string name() override { return stratName; }



    private:
        void registerGhostComms_(std::unique_ptr<MHDMessengerInfo> const& info)
        {
            // static refinement for J and E because in MHD they are temporaries, so keeping there
            // state updated after each regrid is not a priority. However if we do not correctly
            // refine on regrid, the post regrid state is not up to date (in our case it will be nan
            // since we nan-initialise) and thus is is better to rely on static refinement, which
            // uses the state after computation of ampere or CT.
            elecGhostsRefiners_.addStaticRefiners(info->ghostElectric, EfieldRefineOp_,
                                                  info->ghostElectric,
                                                  nonOverwriteInteriorTFfillPattern);

            currentGhostsRefiners_.addStaticRefiners(info->ghostCurrent, EfieldRefineOp_,
                                                     info->ghostCurrent,
                                                     nonOverwriteInteriorTFfillPattern);


            rhoGhostsRefiners_.addTimeRefiners(info->ghostDensity, info->modelDensity,
                                               rhoOld_.name(), mhdFieldRefineOp_, fieldTimeOp_,
                                               nonOverwriteFieldFillPattern);


            momentumGhostsRefiners_.addTimeRefiners(
                info->ghostMomentum, info->modelMomentum, rhoVold_.name(), mhdVecFieldRefineOp_,
                vecFieldTimeOp_, nonOverwriteInteriorTFfillPattern);

            totalEnergyGhostsRefiners_.addTimeRefiners(
                info->ghostTotalEnergy, info->modelTotalEnergy, EtotOld_.name(), mhdFieldRefineOp_,
                fieldTimeOp_, nonOverwriteFieldFillPattern);

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
            magComms_.magneticPatchStratPerGhostRefiner_ = [&]() {
                std::vector<std::shared_ptr<
                    MagneticRefinePatchStrategy<ResourcesManagerT, VectorFieldDataT>>>
                    result;

                result.reserve(info->ghostMagnetic.size());

                for (auto const& key : info->ghostMagnetic)
                {
                    auto&& [id] = resourcesManager_->getIDsList(key);

                    auto patch_strat = std::make_shared<
                        MagneticRefinePatchStrategy<ResourcesManagerT, VectorFieldDataT>>(
                        *resourcesManager_);

                    patch_strat->registerIDs(id);

                    result.push_back(patch_strat);
                }
                return result;
            }();

            for (size_t i = 0; i < info->ghostMagnetic.size(); ++i)
            {
                magGhostsRefiners_.addStaticRefiner(
                    info->ghostMagnetic[i], BfieldRegridOp_, info->ghostMagnetic[i],
                    nonOverwriteInteriorTFfillPattern, magComms_.magneticPatchStratPerGhostRefiner_[i]);

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


        // --- saved state ---
        FieldT rhoOld_{stratName + "rhoOld", core::MHDQuantity::Scalar::rho};
        VecFieldT Vold_{stratName + "Vold", core::MHDQuantity::Vector::V};
        FieldT Pold_{stratName + "Pold", core::MHDQuantity::Scalar::P};

        VecFieldT rhoVold_{stratName + "rhoVold", core::MHDQuantity::Vector::rhoV};
        FieldT EtotOld_{stratName + "EtotOld", core::MHDQuantity::Scalar::Etot};

        VecFieldT Jold_{stratName + "Jold", core::MHDQuantity::Vector::J};


        // --- resources ---
        using rm_t = typename MHDModel::resources_manager_type;
        std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager_;
        int const firstLevel_;

        using InitRefinerPool             = RefinerPool<rm_t, RefinerType::InitField>;
        using GhostRefinerPool            = RefinerPool<rm_t, RefinerType::GhostField>;
        using InitDomPartRefinerPool      = RefinerPool<rm_t, RefinerType::InitInteriorPart>;
        using VecFieldGhostMaxRefinerPool = RefinerPool<rm_t, RefinerType::PatchVecFieldBorderMax>;

        // --- B-field comms ---
        BfieldComms<ResourcesManagerT, VectorFieldDataT> magComms_{*resourcesManager_};

        // --- reflux comms ---
        MHDMHDRefluxComms<MHDModel> mhdRefluxComms_;

        // --- refiner pools ---
        GhostRefinerPool elecGhostsRefiners_{resourcesManager_};
        GhostRefinerPool currentGhostsRefiners_{resourcesManager_};
        GhostRefinerPool rhoGhostsRefiners_{resourcesManager_};
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

        // --- operators ---
        using RefOp_ptr     = std::shared_ptr<SAMRAI::hier::RefineOperator>;
        using CoarsenOp_ptr = std::shared_ptr<SAMRAI::hier::CoarsenOperator>;
        using TimeOp_ptr    = std::shared_ptr<SAMRAI::hier::TimeInterpolateOperator>;

        template<typename Policy>
        using FieldRefineOp = FieldRefineOperator<GridLayoutT, GridT, Policy>;

        template<typename Policy>
        using VecFieldRefineOp = VecFieldRefineOperator<GridLayoutT, GridT, Policy>;

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

        using ElectricFieldCoarsenOp = VecFieldCoarsenOp<ElectricFieldCoarsener<dimension>>;

        SynchronizerPool<rm_t> electroSynchronizers_{resourcesManager_};

        RefOp_ptr mhdFluxRefineOp_{std::make_shared<MHDFluxRefineOp>()};
        RefOp_ptr mhdVecFluxRefineOp_{std::make_shared<MHDVecFluxRefineOp>()};
        RefOp_ptr mhdFieldRefineOp_{std::make_shared<MHDFieldRefineOp>()};
        RefOp_ptr mhdVecFieldRefineOp_{std::make_shared<MHDVecFieldRefineOp>()};
        RefOp_ptr EfieldRefineOp_{std::make_shared<ElectricFieldRefineOp>()};
        RefOp_ptr BfieldRefineOp_{std::make_shared<MagneticFieldRefineOp>()};
        RefOp_ptr BfieldRegridOp_{std::make_shared<MagneticFieldRegridOp>()};

        TimeOp_ptr fieldTimeOp_{std::make_shared<FieldTimeInterp>()};
        TimeOp_ptr vecFieldTimeOp_{std::make_shared<VecFieldTimeInterp>()};

        // --- fill patterns ---
        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension /*, rank=1*/>;
        using FieldFillPattern_t       = FieldFillPattern<dimension>;

        std::shared_ptr<FieldFillPattern_t> nonOverwriteFieldFillPattern
            = std::make_shared<FieldFillPattern<dimension>>(); // stateless (mostly)

        std::shared_ptr<TensorFieldFillPattern_t> nonOverwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>();

        std::shared_ptr<TensorFieldFillPattern_t> overwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>(
                /*overwrite_interior=*/true);

        CoarsenOp_ptr electricFieldCoarseningOp_{std::make_shared<ElectricFieldCoarsenOp>()};
    };

} // namespace amr
} // namespace PHARE
#endif
