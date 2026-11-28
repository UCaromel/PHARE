#ifndef PHARE_MHD_MESSENGER_HPP
#define PHARE_MHD_MESSENGER_HPP

#include "phare_mpi.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include "amr/data/field/coarsening/electric_field_coarsener.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/mhd_flux_coarsener.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/composite_field_refiner.hpp"
#include "amr/data/field/refine/limited_o4_refiner.hpp"
#include "amr/data/field/refine/magnetic_composite_refiner.hpp"
#include "amr/data/field/refine/parent_cell_field_refiner.hpp"
#include "amr/messengers/refinement_config.hpp"
#include "amr/data/field/time_interpolate/field_linear_time_interpolate.hpp"
#include "amr/messengers/refiner.hpp"
#include "amr/messengers/composite_refine_patch_strategy.hpp"
#include "amr/messengers/refiner_pool.hpp"
#include "amr/messengers/synchronizer_pool.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/messenger_info.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/data/field/refine/field_refine_patch_strategy.hpp"
#include "amr/messengers/mhd_messenger_fwd.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/data/field/refine/magnetic_refine_patch_strategy.hpp"
#include "core/numerics/rk_stage_context.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"


#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/RefineOperator.h"
#include "SAMRAI/hier/CoarsenOperator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace PHARE
{
namespace amr
{
    template<typename MHDModel, typename TemporalTransfer>
    class MHDMessenger : public IMessenger<typename MHDModel::Interface>
    {
        using amr_types     = MHDModel::amr_types;
        using level_t       = amr_types::level_t;
        using patch_t       = amr_types::patch_t;
        using hierarchy_t   = amr_types::hierarchy_t;
        using level_ptr     = amr_types::level_ptr;
        using hierarchy_ptr = amr_types::hierarchy_ptr;

        using IPhysicalModel    = MHDModel::Interface;
        using FieldT            = MHDModel::field_type;
        using VecFieldT         = MHDModel::vecfield_type;
        using MHDStateT         = MHDModel::state_type;
        using GridLayoutT       = MHDModel::gridlayout_type;
        using GridT             = MHDModel::grid_type;
        using ResourcesManagerT = MHDModel::resources_manager_type;
        using BoundaryManagerT  = MHDModel::boundary_manager_type;
        using FieldDataT        = FieldData<GridLayoutT, GridT, core::MHDQuantity::Scalar>;
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
                     std::shared_ptr<BoundaryManagerT> boundaryManager, int const firstLevel,
                     RefinementConfig const& refinementConfig = {})
            : temporalTransfer_{resourcesManager}
            , resourcesManager_{std::move(resourcesManager)}
            , boundaryManager_{std::move(boundaryManager)}
            , firstLevel_{firstLevel}
        {
            makeRefineOperators_(refinementConfig);

            resourcesManager_->registerResources(temporalTransfer_);
        }

        virtual ~MHDMessenger() = default;

        void allocate(patch_t& patch, double const allocateTime) const override
        {
            resourcesManager_->allocate(temporalTransfer_, patch, allocateTime);
        }


        void registerQuantities(std::unique_ptr<IMessengerInfo> /*fromCoarserInfo*/,
                                std::unique_ptr<IMessengerInfo> fromFinerInfo) override
        {
            std::unique_ptr<MHDMessengerInfo> mhdInfo{
                dynamic_cast<MHDMessengerInfo*>(fromFinerInfo.release())};
            temporalTransfer_.registerQuantities(*mhdInfo);

            // without a coarse-fine path there is nothing to reflux or to initialise from a
            // coarser level: only the ghost fills are registered, same-level ones
            if (temporalTransfer_.hasCoarseFine())
                registerCoarseFineComms_(mhdInfo);
            else
                registerGhostComms_(mhdInfo);
        }

        void registerCoarseFineComms_(std::unique_ptr<MHDMessengerInfo> const& mhdInfo)
        {
            auto b_id = resourcesManager_->getID(mhdInfo->modelMagnetic);

            if (!b_id)
            {
                throw std::runtime_error(
                    "MHDMessengerStrategy: missing magnetic field variable IDs");
            }

            magneticRefinePatchStrategy_->registerIDs(*b_id);

            auto e_id = resourcesManager_->getID(mhdInfo->modelElectric);

            if (!e_id)
            {
                throw std::runtime_error(
                    "MHDMessengerStrategy: missing electric field variable IDs");
            }

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


            // fluxes, flux sums and E of all directions share the refluxing algorithms: their
            // factory types differ by centering
            registerFluxReflux_(*rho_fx_reflux_id, *rhoV_fx_reflux_id, *Etot_fx_reflux_id,
                                *rho_fx_fluxsum_id, *rhoV_fx_fluxsum_id, *Etot_fx_fluxsum_id);

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

                registerFluxReflux_(*rho_fy_reflux_id, *rhoV_fy_reflux_id, *Etot_fy_reflux_id,
                                    *rho_fy_fluxsum_id, *rhoV_fy_fluxsum_id, *Etot_fy_fluxsum_id);

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

                    registerFluxReflux_(*rho_fz_reflux_id, *rhoV_fz_reflux_id, *Etot_fz_reflux_id,
                                        *rho_fz_fluxsum_id, *rhoV_fz_fluxsum_id,
                                        *Etot_fz_fluxsum_id);
                }
            }

            auto e_reflux_id = resourcesManager_->getID(mhdInfo->refluxElectric);

            auto e_fluxsum_id = resourcesManager_->getID(mhdInfo->fluxSumElectric);

            if (!e_reflux_id or !e_fluxsum_id)
            {
                throw std::runtime_error(
                    "MHDMessenger: missing electric refluxing field variable IDs");
            }

            refluxAlgo_.registerCoarsen(*e_reflux_id, *e_fluxsum_id, electricFieldCoarseningOp_);

            patchGhostRefluxedAlgo_.registerRefine(*e_reflux_id, *e_reflux_id, *e_reflux_id,
                                                   EfieldRefineOp_,
                                                   nonOverwriteInteriorTFfillPattern);

            registerGhostComms_(mhdInfo);
            registerInitComms_(mhdInfo);
        }



        void registerLevel(hierarchy_ptr const& hierarchy, int const levelNumber) override
        {
            hierarchy_ = hierarchy;

            if (levelNumber != rootLevelNumber && !temporalTransfer_.hasCoarseFine())
                throw std::runtime_error(
                    "MHDMessenger: this temporal transfer has no coarse-fine path (fourth-order "
                    "MHD on refined levels needs the SSPRK4_5 time integrator)");

            // the level is new: drop the schedules of the level it replaces, rebuild same-level
            sameLevelBuiltOn_.erase(levelNumber);
            interLevelBuiltOn_.erase(levelNumber);
            removeInterLevelSchedules_(levelNumber);
            buildSameLevelSchedules_(hierarchy, levelNumber);
            sameLevelBuiltOn_[levelNumber] = levelsOf_(hierarchy, levelNumber, levelNumber);

            // inter-level schedules read the coarser levels, which may still be replaced before
            // the hierarchy settles (e.g. a level-0 rebalance), so onHierarchyChanged builds them
        }



        // Rebuilds the schedules of levelNumber that were built from a level since replaced.
        // Building a schedule is MPI-collective: all ranks take the same branches, because
        // SAMRAI replaces a level on all ranks at once.
        void onHierarchyChanged(hierarchy_ptr const& hierarchy, int const levelNumber) override
        {
            if (!builtOnCurrentLevels_(sameLevelBuiltOn_, hierarchy, levelNumber, levelNumber))
            {
                buildSameLevelSchedules_(hierarchy, levelNumber);
                sameLevelBuiltOn_[levelNumber] = levelsOf_(hierarchy, levelNumber, levelNumber);
            }

            if (levelNumber != rootLevelNumber
                and !builtOnCurrentLevels_(interLevelBuiltOn_, hierarchy, rootLevelNumber,
                                           levelNumber))
            {
                interLevelBuiltOn_.erase(levelNumber);
                removeInterLevelSchedules_(levelNumber);
                buildInterLevelSchedules_(hierarchy, levelNumber);
                interLevelBuiltOn_[levelNumber]
                    = levelsOf_(hierarchy, rootLevelNumber, levelNumber);
            }

            refillGhostsAtInitTime_(hierarchy, levelNumber);
        }


        void regrid(hierarchy_ptr const& hierarchy, int const levelNumber,
                    level_ptr const& oldLevel, IPhysicalModel& model,
                    double const initDataTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            auto& mhdModel = static_cast<MHDModel&>(model);
            auto level     = hierarchy->getPatchLevel(levelNumber);

            stateInitRefiners_.regrid(hierarchy, levelNumber, oldLevel, initDataTime);
            probeState_(mhdModel.state, *level, "regrid", initDataTime);

            fillGhostsAtInitTime_(mhdModel, *level, initDataTime);
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

        void initLevel(hierarchy_ptr const& hierarchy, IPhysicalModel& model, level_t& level,
                       double const initDataTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            auto& mhdModel      = static_cast<MHDModel&>(model);
            auto const levelPtr = hierarchy->getPatchLevel(level.getLevelNumber());

            // init-only schedules are used once, so they are built at fill time, not stored
            stateInitRefiners_.fillTransient(hierarchy, levelPtr, initDataTime);
            probeState_(mhdModel.state, level, "initLevel", initDataTime);

            fillGhostsAtInitTime_(mhdModel, level, initDataTime);
        }

        void firstStep(IPhysicalModel& model, level_t& level, double const /*currentTime*/,
                       double const prevCoarserTIme, double const newCoarserTime) final
        {
            temporalTransfer_.firstStep(level, hierarchy_, prevCoarserTIme, newCoarserTime);
        }


        void lastStep(IPhysicalModel& model, level_t& level) final {}


        // the old state used for time interpolation is snapshotted by SolverMHD::prepareStep
        void prepareStep(IPhysicalModel& /*model*/, level_t& /*level*/,
                         double /*currentTime*/) final
        {
        }

        void fillRootGhosts(IPhysicalModel& model, level_t& level, double const initDataTime) final
        {
            auto& mhdModel = static_cast<MHDModel&>(model);
            temporalTransfer_.levelMade(level, hierarchy_, initDataTime);
            fillConservativeGhosts(mhdModel.state, level, initDataTime,
                                   core::RKStageContext::endOfStep(initDataTime));

            // A face on a periodic boundary is initialised twice, at x=0 and at x=L, and the two
            // values can differ in the last bits: make every patch holding it agree once, before
            // anything reads B. This must follow the ghost fill: the max is taken over whole ghost
            // overlaps, so ghosts still holding initializer values would leak into the interior.
            magMaxRefiners_.fill(mhdModel.state.B, level.getLevelNumber(), initDataTime);
        }

        void synchronize(level_t& level) final {}

        void reflux(int const coarserLevelNumber, int const fineLevelNumber,
                    double const syncTime) override
        {
            if (!temporalTransfer_.hasCoarseFine())
                return;

            refluxSchedules_.at(fineLevelNumber)->coarsenData();
            patchGhostRefluxedSchedules_.at(coarserLevelNumber)->fillData(syncTime);
        }

        void postSynchronize(IPhysicalModel& model, level_t& level, double const time) override
        {
            // The ghosts for B are obtained in the solver's reflux_euler. For B, this is because
            // refluxing is done through faraday which is computed on the ghost box for the other
            // quantities, the ghosts are filled in the end of the euler step anyways.
        }

        // Fills the ghosts of the conserved state (B, rho, rhoV, Etot). The context places the
        // fill in the Runge-Kutta step, for the temporal transfer to prepare what is refined.
        void fillConservativeGhosts(MHDStateT& state, level_t const& level, double const fillTime,
                                    core::RKStageContext const& context)
        {
            PHARE_LOG_SCOPE(3, "MHDMessenger::fillConservativeGhosts");

            setNaNsOnConservativeGhosts_(state, level);
            temporalTransfer_.prepareTarget(state, level, context);
            fillConservativeGhosts_(state, level, fillTime);
            probeState_(state, level, "ghostfill", fillTime);
        }

        // EXP PROBE (exp/o4-refine-limit, PHARE_PROBE=1): min rho / min p on the interior
        // (p from cell averages, and from Laplacian point values of the hydro rows) and on the
        // ghost cells (averages only) of a level. "initLevel"/"regrid": every call printed;
        // "ghostfill": only calls with a non-positive or NaN cell, capped. PHARE_PROBE=2: all.
        void probeState_(MHDStateT& st, level_t const& level, char const* tag, double const time)
        {
            static int const on = [] {
                auto const* e = std::getenv("PHARE_PROBE");
                return e ? std::atoi(e) : 0;
            }();
            static int printed = 0;
            if (!on)
                return;
            bool const isGhostFill = tag[0] == 'g';
            if (isGhostFill and on < 2 and printed >= 300)
                return;
            constexpr double g1 = 5. / 3. - 1.;
            double iRho = 1e300, iP = 1e300, iRhoPV = 1e300, iPPV = 1e300, gRho = 1e300, gP = 1e300;
            long iBad = 0, iBadPV = 0, iNaN = 0, gBad = 0, gNaN = 0, nI = 0, nG = 0;
            using Idx = core::MeshIndex<dimension>;
            int iWorstPatch = -1, gWorstPatch = -1, ip = 0;
            std::array<int, dimension> iWorstAMR{}, gWorstAMR{};
            for (auto const& patch : level)
            {
                auto dataOnPatch
                    = resourcesManager_->setOnPatch(*patch, st.rho, st.rhoV, st.Etot, st.B);
                auto layout = layoutFromPatch<GridLayoutT>(*patch);
                std::array<std::uint32_t, dimension> lo{}, hi{};
                for (std::size_t d = 0; d < dimension; ++d)
                {
                    lo[d] = layout.physicalStartIndex(st.rho, static_cast<core::Direction>(d));
                    hi[d] = layout.physicalEndIndex(st.rho, static_cast<core::Direction>(d));
                }
                auto const amrOf = [&](Idx const& idx) {
                    std::array<int, dimension> a{};
                    for (std::size_t d = 0; d < dimension; ++d)
                        a[d] = static_cast<int>(idx[d]) - static_cast<int>(lo[d])
                               + layout.AMRBox().lower[d];
                    return a;
                };
                auto const b2 = [&](Idx const& idx) {
                    double s = 0;
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        auto const& B = st.B(static_cast<core::Component>(c));
                        double b      = B(idx);
                        if (c < dimension)
                        {
                            Idx up = idx;
                            ++up[c];
                            b = 0.5 * (b + B(up));
                        }
                        s += b * b;
                    }
                    return s;
                };
                layout.evalOnGhostBox(st.rho, [&](auto&... args) {
                    Idx idx{args...};
                    bool interior = true, edge = false;
                    for (std::size_t d = 0; d < dimension; ++d)
                    {
                        interior = interior and idx[d] >= lo[d] and idx[d] <= hi[d];
                        edge     = edge or idx[d] + 1 >= st.rho.shape()[d];
                    }
                    if (edge)
                        return; // no upper face for the B average
                    double const rho = st.rho(idx);
                    double m2 = 0;
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        double const m = st.rhoV(static_cast<core::Component>(c))(idx);
                        m2 += m * m;
                    }
                    double const bb = b2(idx);
                    double const p  = g1 * (st.Etot(idx) - 0.5 * m2 / rho - 0.5 * bb);
                    if (!interior)
                    {
                        ++nG;
                        if (std::isnan(rho) or std::isnan(p))
                        {
                            ++gNaN;
                            return;
                        }
                        gRho = std::min(gRho, rho);
                        if (p < gP)
                        {
                            gP          = p;
                            gWorstAMR   = amrOf(idx);
                            gWorstPatch = ip;
                        }
                        if (rho <= 0 or p <= 0)
                            ++gBad;
                        return;
                    }
                    ++nI;
                    if (std::isnan(rho) or std::isnan(p))
                    {
                        ++iNaN;
                        return;
                    }
                    iRho = std::min(iRho, rho);
                    if (p < iP)
                    {
                        iP          = p;
                        iWorstAMR   = amrOf(idx);
                        iWorstPatch = ip;
                    }
                    if (rho <= 0 or p <= 0)
                        ++iBad;
                    double const rhoPV = rho - layout.lapl(st.rho, idx) / 24.;
                    double mPV2        = 0;
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        auto const& mf = st.rhoV(static_cast<core::Component>(c));
                        double const m = mf(idx) - layout.lapl(mf, idx) / 24.;
                        mPV2 += m * m;
                    }
                    double const pPV = g1
                                       * (st.Etot(idx) - layout.lapl(st.Etot, idx) / 24.
                                          - 0.5 * mPV2 / rhoPV - 0.5 * bb);
                    if (!std::isnan(pPV))
                    {
                        iRhoPV = std::min(iRhoPV, rhoPV);
                        iPPV   = std::min(iPPV, pPV);
                        if (rhoPV <= 0 or pPV <= 0)
                            ++iBadPV;
                    }
                });
                ++ip;
            }
            if (isGhostFill and on < 2 and iBad + iBadPV + iNaN + gBad + gNaN == 0)
                return;
            if (isGhostFill)
                ++printed; // the cap is on ghost-fill lines only
            std::cerr.precision(10);
            std::cerr << "PROBE " << tag << " t=" << time << " L" << level.getLevelNumber()
                      << " nI=" << nI << " iRho=" << iRho << " iP=" << iP << " iRhoPV=" << iRhoPV
                      << " iPPV=" << iPPV << " iBad=" << iBad << " iBadPV=" << iBadPV
                      << " iNaN=" << iNaN << " nG=" << nG << " gRho=" << gRho << " gP=" << gP
                      << " gBad=" << gBad << " gNaN=" << gNaN << " iAt=" << iWorstAMR[0] << ","
                      << (dimension > 1 ? iWorstAMR[1] : 0) << " gAt=" << gWorstAMR[0] << ","
                      << (dimension > 1 ? gWorstAMR[1] : 0) << std::endl;
            if (not isGhostFill and refine_exp::rule() != 0)
                std::cerr << "EROW t=" << time << " limited=" << refine_exp::nLimited
                          << " applied=" << refine_exp::eApplied
                          << " fallback=" << refine_exp::eFallback << std::endl;
        }

        void fillElectricGhosts(VecFieldT& E, level_t const& level, double const fillTime)
        {
            setNaNsOnVecfieldGhosts(E, level);
            elecGhostsRefiners_.fill(E, level.getLevelNumber(), fillTime);
        }

        // Flags of the ghost cells, from the patches that own the cells. A ghost cell of a
        // coarse-fine boundary takes the flag its coarse cell had at the last stage of the coarser
        // level: the flags are not interpolated in time.
        void fillTroubledCellGhosts(FieldT& troubled, level_t const& level, double const fillTime)
        {
            setNaNsOnFieldGhosts(troubled, level);
            troubledCellGhostsRefiners_.fill(troubled, level.getLevelNumber(), fillTime);
        }

        std::string name() override { return stratName; }



    private:
        void setNaNsOnConservativeGhosts_(MHDStateT& state, level_t const& level)
        {
            setNaNsOnVecfieldGhosts(state.B, level);
            setNaNsOnFieldGhosts(state.rho, level);
            setNaNsOnVecfieldGhosts(state.rhoV, level);
            setNaNsOnFieldGhosts(state.Etot, level);
        }

        void fillConservativeGhosts_(MHDStateT& state, level_t const& level, double const fillTime)
        {
            stateGhostsRefiners_.fillByKey(state.B.name(), level.getLevelNumber(), fillTime);
        }

        void buildSameLevelSchedules_(hierarchy_ptr const& hierarchy, int const levelNumber)
        {
            auto const level = hierarchy->getPatchLevel(levelNumber);

            // nothing is refluxed without a coarse-fine path
            if (temporalTransfer_.hasCoarseFine())
            {
                auto& patchGhostRefluxed = patchGhostRefluxedSchedules_[levelNumber];
                patchGhostRefluxed       = patchGhostRefluxedAlgo_.createSchedule(level);
            }

            // on the root level there is no coarser source, so these are same-level schedules
            if (levelNumber == rootLevelNumber)
            {
                forEachGhostFieldPool_([&](auto& pool) { pool.registerLevel(hierarchy, level); });
            }
            if (levelNumber == rootLevelNumber)
                magMaxRefiners_.registerLevel(hierarchy, level);
        }



        void buildInterLevelSchedules_(hierarchy_ptr const& hierarchy, int const levelNumber)
        {
            auto const level = hierarchy->getPatchLevel(levelNumber);

            forEachGhostFieldPool_([&](auto& pool) { pool.registerLevel(hierarchy, level); });

            // refluxing
            auto const& coarseLevel       = hierarchy->getPatchLevel(levelNumber - 1);
            refluxSchedules_[levelNumber] = refluxAlgo_.createSchedule(coarseLevel, level);
        }



        // the levels lo..hi of the hierarchy, recorded against the schedules built from them
        static auto levelsOf_(hierarchy_ptr const& hierarchy, int const lo, int const hi)
        {
            std::vector<std::weak_ptr<level_t>> levels;
            for (auto ilvl = lo; ilvl <= hi; ++ilvl)
                levels.emplace_back(hierarchy->getPatchLevel(ilvl));
            return levels;
        }

        // whether the schedules of level hi were built against the current levels lo..hi
        static bool builtOnCurrentLevels_(auto const& builtOn, hierarchy_ptr const& hierarchy,
                                          int const lo, int const hi)
        {
            auto const it = builtOn.find(hi);
            if (it == builtOn.end())
                return false;
            for (auto ilvl = lo; ilvl <= hi; ++ilvl)
                if (it->second[ilvl - lo].lock() != hierarchy->getPatchLevel(ilvl))
                    return false;
            return true;
        }

        template<typename Fn>
        void forEachGhostFieldPool_(Fn&& fn)
        {
            fn(elecGhostsRefiners_);
            fn(stateGhostsRefiners_);
            fn(troubledCellGhostsRefiners_);
        }

        void removeInterLevelSchedules_(int const levelNumber)
        {
            forEachGhostFieldPool_([&](auto& pool) { pool.removeLevel(levelNumber); });
            refluxSchedules_.erase(levelNumber);
        }

        // The init and regrid schedules interpolate in space only and read the coarser levels at
        // their latest time. A level made partway through a coarser step thus gets ghosts from
        // coarser data ahead of its own time. Its ghosts are refilled with time interpolation, as
        // in a step, by onHierarchyChanged once the schedules for it are built. The refill reads
        // the coarser levels' old data, which their solver sets when they are made and again
        // after their own refill (MultiPhysicsIntegrator).
        void fillGhostsAtInitTime_(MHDModel& model, level_t& level, double const initDataTime)
        {
            bool const refill = temporalTransfer_.levelMade(level, hierarchy_, initDataTime);
            if (level.getLevelNumber() == rootLevelNumber || !refill)
                return;
            pendingGhostRefills_[level.getLevelNumber()] = {&model, initDataTime};
        }

        void refillGhostsAtInitTime_(hierarchy_ptr const& hierarchy, int const levelNumber)
        {
            auto const pending = pendingGhostRefills_.find(levelNumber);
            if (pending == pendingGhostRefills_.end())
                return;

            auto const [model, initDataTime] = pending->second;
            pendingGhostRefills_.erase(pending);

            auto& level = *hierarchy->getPatchLevel(levelNumber);
            fillConservativeGhosts(model->state, level, initDataTime,
                                   core::RKStageContext::endOfStep(initDataTime));
        }

        // EXP: the operators of the density, momentum and total-energy rows of one state.
        // `limited` false or no rule: the base operators.
        struct HydroOps_
        {
            std::shared_ptr<SAMRAI::hier::RefineOperator> rho, rhoV, etot;
        };
        HydroOps_ hydroOps_(bool const limited, std::string const& rhoName,
                            std::string const& rhoVName)
        {
            if (!limited or !limFieldRefineOp_)
                return {mhdFieldRefineOp_, mhdVecFieldRefineOp_, mhdFieldRefineOp_};
            std::shared_ptr<SAMRAI::hier::RefineOperator> etot = limFieldRefineOp_;
            if (refine_exp::energy() != 0)
            {
                auto const rhoId  = resourcesManager_->getID(rhoName);
                auto const rhoVId = resourcesManager_->getID(rhoVName);
                if (!rhoId or !rhoVId)
                    throw std::runtime_error("energy row: no id for " + rhoName + " / " + rhoVName);
                etot = std::make_shared<
                    EnergyRowO4RefineOperator<GridLayoutT, GridT, VectorFieldDataT>>(*rhoId,
                                                                                    *rhoVId);
            }
            return {limFieldRefineOp_, limVecFieldRefineOp_, etot};
        }

        // Select the field-refinement operators once at construction. The composite runtime
        // kernels are built from the configured order.
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

            // EXP: limited order-4 operators of the hydro rows (2D, env knobs)
            if (dimension == 2 and config.order == FieldRefinementOrder::Cubic
                and refine_exp::rule() != 0)
            {
                limFieldRefineOp_
                    = std::make_shared<KernelFieldRefineOperator<GridLayoutT, GridT>>(
                        std::make_unique<LimitedO4FieldRefiner<GridLayoutT, GridT>>());
                limVecFieldRefineOp_
                    = std::make_shared<KernelVecFieldRefineOperator<VectorFieldDataT>>(
                        std::make_unique<LimitedO4FieldRefiner<GridLayoutT, GridT>>());
            }

            mhdFluxRefineOp_     = fieldKernel();
            mhdVecFluxRefineOp_  = vecKernel();
            mhdFieldRefineOp_    = fieldKernel();
            troubledCellRefineOp_ = std::make_shared<KernelFieldRefineOperator<GridLayoutT, GridT>>(
                std::make_unique<ParentCellFieldRefiner<GridLayoutT, GridT>>());
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
            // The refiners for the electric field only serve for filling ghosts at physical
            // boundaries.
            registerGhostRefinePatchStrategies_(elecPatchStrats, info->ghostElectric);
            for (std::size_t i = 0; i < info->ghostElectric.size(); ++i)
                elecGhostsRefiners_.addStaticRefiner(info->ghostElectric[i], nullptr,
                                                     info->ghostElectric[i], noOverlapFillPattern,
                                                     elecPatchStrats[i]);

            // each quantity keeps its own patch strategy, so that the physical-boundary ghosts
            // are filled by its boundary condition; a state's strategies are combined below
            registerGhostRefinePatchStrategies_(rhoPatchStrats, info->ghostDensity);
            registerGhostRefinePatchStrategies_(momentumPatchStrats, info->ghostMomentum);
            registerGhostRefinePatchStrategies_(totalEnergyPatchStrats, info->ghostTotalEnergy);

            // the strategy is per quantity, not per refiner: each one registers the ids of its
            // quantity, and the strategies of a state are combined into its refiner below
            registerGhostRefinePatchStrategies_(magPatchStrats, info->ghostMagnetic);

            // the i-th entry of each ghost list belongs to the same state
            auto const nStates = info->ghostMagnetic.size();
            if (info->ghostDensity.size() != nStates or info->ghostMomentum.size() != nStates
                or info->ghostTotalEnergy.size() != nStates)
                throw std::runtime_error("MHDMessenger: ghost state lists differ in size");

            // Where the ghosts of a quantity come from depends on the temporal transfer:
            //  - no coarse-fine path: the same level and its periodic images only
            //  - linear in time: the coarser old and model states, interpolated in time
            //  - otherwise: the state the transfer assembles on the coarser level for each fill
            bool const coarseFine = temporalTransfer_.hasCoarseFine();
            auto const assembled  = [&]() -> core::MHDStateNames {
                if constexpr (linearInTime)
                    return {};
                else
                    return temporalTransfer_.assembledStateNames();
            }();
            auto const item = [&](std::string const& ghost, std::string const& model,
                                  std::string const& old, std::string const& assembledName,
                                  RefOp_ptr const& refineOp, TimeOp_ptr const& timeOp,
                                  std::shared_ptr<SAMRAI::xfer::VariableFillPattern> fillPattern) {
                if (!coarseFine)
                    return RefineItem{.dst         = ghost,
                                      .src         = ghost,
                                      .refineOp    = nullptr,
                                      .fillPattern = fillPattern};
                if constexpr (linearInTime)
                    return RefineItem{.dst         = ghost,
                                      .src         = ghost,
                                      .told        = old,
                                      .tnew        = model,
                                      .refineOp    = refineOp,
                                      .timeOp      = timeOp,
                                      .fillPattern = fillPattern};
                else
                    return RefineItem{.dst         = ghost,
                                      .src         = assembledName,
                                      .refineOp    = refineOp,
                                      .fillPattern = fillPattern};
            };

            for (std::size_t i = 0; i < nStates; ++i)
            {
                auto const ops = hydroOps_(coarseFine and refine_exp::limitGhosts(),
                                           info->ghostDensity[i], info->ghostMomentum[i]);
                stateGhostsRefiners_.addRefiner(
                    info->ghostMagnetic[i],
                    {item(info->ghostMagnetic[i], info->modelMagnetic, info->oldState.B(),
                          assembled.B(), BfieldRegridOp_, vecFieldTimeOp_,
                          nonOverwriteInteriorTFfillPattern),
                     item(info->ghostDensity[i], info->modelDensity, info->oldState.rho(),
                          assembled.rho(), ops.rho, fieldTimeOp_,
                          nonOverwriteFieldFillPattern),
                     item(info->ghostMomentum[i], info->modelMomentum, info->oldState.rhoV(),
                          assembled.rhoV(), ops.rhoV, vecFieldTimeOp_,
                          nonOverwriteInteriorTFfillPattern),
                     item(info->ghostTotalEnergy[i], info->modelTotalEnergy, info->oldState.Etot(),
                          assembled.Etot(), ops.etot, fieldTimeOp_,
                          nonOverwriteFieldFillPattern)},
                    std::make_shared<CompositeRefinePatchStrategy>(
                        std::vector<CompositeRefinePatchStrategy::strategy_ptr>{
                            magPatchStrats[i], rhoPatchStrats[i], momentumPatchStrats[i],
                            totalEnergyPatchStrats[i]}));
            }

            if (!info->troubledCells.empty())
            {
                registerGhostRefinePatchStrategies_(troubledCellPatchStrats, {info->troubledCells});
                troubledCellGhostsRefiners_.addStaticRefiner(
                    info->troubledCells, coarseFine ? troubledCellRefineOp_ : nullptr,
                    info->troubledCells, nonOverwriteFieldFillPattern,
                    troubledCellPatchStrats.front());
            }

            magMaxRefiners_.addStaticRefiner(
                info->modelMagnetic, info->modelMagnetic, nullptr, info->modelMagnetic,
                std::make_shared<
                    TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());
        }




        // registers the reflux of the fluxes of one direction, and the refill of their ghosts so
        // that they agree with the newly refluxed cells (both run in reflux)
        void registerFluxReflux_(int const rho, int const rhoV, int const Etot, int const rhoSum,
                                 int const rhoVSum, int const EtotSum)
        {
            refluxAlgo_.registerCoarsen(rho, rhoSum, mhdFluxCoarseningOp_);
            refluxAlgo_.registerCoarsen(rhoV, rhoVSum, mhdVecFluxCoarseningOp_);
            refluxAlgo_.registerCoarsen(Etot, EtotSum, mhdFluxCoarseningOp_);

            patchGhostRefluxedAlgo_.registerRefine(rho, rho, rho, mhdFluxRefineOp_,
                                                   nonOverwriteInteriorTFfillPattern);
            patchGhostRefluxedAlgo_.registerRefine(rhoV, rhoV, rhoV, mhdVecFluxRefineOp_,
                                                   nonOverwriteInteriorTFfillPattern);
            patchGhostRefluxedAlgo_.registerRefine(Etot, Etot, Etot, mhdFluxRefineOp_,
                                                   nonOverwriteInteriorTFfillPattern);
        }




        /**
         * @brief Register a list of refine patch strategy pointers corresponding to a list of
         * keys.
         *
         * @tparam RefinePatchStrategyT type inheriting from SAMRAI's `RefinePatchStrategy`
         * @param patchStrategies the list of refine patch strategy pointers.
         * @param keys the list of keys.
         */
        template<typename RefinePatchStrategyT>
        void registerGhostRefinePatchStrategies_(
            std::vector<std::shared_ptr<RefinePatchStrategyT>>& patchStrategies,
            std::vector<std::string> const& keys)
        {
            patchStrategies.clear(); // registerQuantities must be idempotent
            patchStrategies.reserve(keys.size());
            for (auto const& key : keys)
            {
                auto&& [id]     = resourcesManager_->getIDsList(key);
                auto patchStrat = std::make_shared<RefinePatchStrategyT>(*boundaryManager_);
                patchStrat->registerIDs(id);
                patchStrategies.push_back(patchStrat);
            }
        }


        // should this use conservative quantities ? When should we do the initial conversion ?
        // Maybe mhd_init
        void registerInitComms_(std::unique_ptr<MHDMessengerInfo> const& info)
        {
            // The model state's moment strategies (index 0 of each ghost-strategy list:
            // ghostX[0] == modelX == initX) fill the physical-boundary ghosts of a new or
            // regridded level, so that it carries valid moment ghosts before its first flux.
            // Moments keep the default fill pattern, which overwrites the interior: a new or
            // regridded level must have its interior filled.
            std::vector<CompositeRefinePatchStrategy::strategy_ptr> strategies{
                magneticRefinePatchStrategy_};
            if (!rhoPatchStrats.empty())
                strategies.push_back(rhoPatchStrats.front());
            if (!momentumPatchStrats.empty())
                strategies.push_back(momentumPatchStrats.front());
            if (!totalEnergyPatchStrats.empty())
                strategies.push_back(totalEnergyPatchStrats.front());

            std::vector<RefineItem> items{{.dst         = info->modelMagnetic,
                                           .src         = info->modelMagnetic,
                                           .refineOp    = BfieldRegridOp_,
                                           .fillPattern = overwriteInteriorTFfillPattern}};
            if (limFieldRefineOp_
                and (info->initDensity.size() != info->initTotalEnergy.size()
                     or info->initMomentum.size() != info->initTotalEnergy.size()))
                throw std::runtime_error("limited refinement: init rows do not pair up");
            for (auto const& rho : info->initDensity)
                items.push_back({.dst      = rho,
                                 .src      = rho,
                                 .refineOp = limFieldRefineOp_ ? limFieldRefineOp_
                                                               : mhdFieldRefineOp_});
            for (auto const& rhoV : info->initMomentum)
                items.push_back({.dst      = rhoV,
                                 .src      = rhoV,
                                 .refineOp = limVecFieldRefineOp_ ? limVecFieldRefineOp_
                                                                  : mhdVecFieldRefineOp_});
            for (std::size_t k = 0; k < info->initTotalEnergy.size(); ++k)
            {
                auto const& Etot = info->initTotalEnergy[k];
                auto const etotOp
                    = limFieldRefineOp_
                          ? hydroOps_(true, info->initDensity[k], info->initMomentum[k]).etot
                          : mhdFieldRefineOp_;
                items.push_back({.dst = Etot, .src = Etot, .refineOp = etotOp});
            }

            stateInitRefiners_.addRefiner(
                info->modelMagnetic, items,
                std::make_shared<CompositeRefinePatchStrategy>(std::move(strategies)));
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
        void setNaNsOnFieldGhosts(FieldT& field)
        {
            auto const nGhosts = GridLayoutT::nDNbrGhosts(core::QtyCentering::primal)[0];
            field[core::NdArrayMask{0, nGhosts - 1}]
                = std::numeric_limits<typename FieldT::type>::quiet_NaN();
        }

        void setNaNsOnFieldGhosts(FieldT& field, level_t const& level)
        {
            for ([[maybe_unused]] auto& patch : resourcesManager_->enumerate(level, field))
                setNaNsOnFieldGhosts(field);
        }

        void setNaNsOnVecfieldGhosts(VecFieldT& vf, level_t const& level)
        {
            for ([[maybe_unused]] auto& patch : resourcesManager_->enumerate(level, vf))
                for (auto& component : vf)
                    setNaNsOnFieldGhosts(component);
        }


        TemporalTransfer temporalTransfer_;

        using rm_t = MHDModel::resources_manager_type;
        std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager_;
        std::shared_ptr<BoundaryManagerT> boundaryManager_;
        hierarchy_ptr hierarchy_; // the temporal transfer brackets a level with its coarser one
        int const firstLevel_;

        using InitRefinerPool             = RefinerPool<rm_t, RefinerType::InitField>;
        using GhostRefinerPool            = RefinerPool<rm_t, RefinerType::GhostField>;
        using PatchGhostRefinerPool       = RefinerPool<rm_t, RefinerType::PatchGhostField>;
        using InitDomPartRefinerPool      = RefinerPool<rm_t, RefinerType::InitInteriorPart>;
        using VecFieldGhostMaxRefinerPool = RefinerPool<rm_t, RefinerType::PatchVecFieldBorderMax>;


        // coarsens the flux sums and E onto the coarser level, then refills their patch ghosts
        SAMRAI::xfer::CoarsenAlgorithm refluxAlgo_{SAMRAI::tbox::Dimension{dimension}};
        SAMRAI::xfer::RefineAlgorithm patchGhostRefluxedAlgo_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> refluxSchedules_;
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> patchGhostRefluxedSchedules_;

        // Level objects each level's schedules were built against: {L} for same-level schedules,
        // [root, L] for inter-level ones (coarse interpolation can read below L-1). SAMRAI
        // replaces a reinitialised level object and never changes a kept one in place.
        std::map<int, std::vector<std::weak_ptr<level_t>>> sameLevelBuiltOn_;
        std::map<int, std::vector<std::weak_ptr<level_t>>> interLevelBuiltOn_;

        // levels made in the current regrid pass, whose ghosts await refillGhostsAtInitTime_
        struct PendingGhostRefill
        {
            MHDModel* model;
            double initDataTime;
        };
        std::map<int, PendingGhostRefill> pendingGhostRefills_;

        PatchGhostRefinerPool elecGhostsRefiners_{resourcesManager_};
        // B, rho, rhoV and Etot of each state, keyed by the state's B name
        GhostRefinerPool stateGhostsRefiners_{resourcesManager_};
        // flags of the conversion limiter, registered only when the solver publishes them
        GhostRefinerPool troubledCellGhostsRefiners_{resourcesManager_};

        // model B, root level only, for the one-time equalization in fillRootGhosts
        VecFieldGhostMaxRefinerPool magMaxRefiners_{resourcesManager_};

        // B, rho, rhoV and Etot of the model state on new and regridded levels
        InitRefinerPool stateInitRefiners_{resourcesManager_};

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
        RefOp_ptr limFieldRefineOp_;    // EXP: null unless PHARE_REFINE_SLOPE is set
        RefOp_ptr limVecFieldRefineOp_; // EXP
        RefOp_ptr EfieldRefineOp_;
        RefOp_ptr BfieldRegridOp_;
        RefOp_ptr troubledCellRefineOp_;

        TimeOp_ptr fieldTimeOp_{std::make_shared<FieldTimeInterp>()};
        TimeOp_ptr vecFieldTimeOp_{std::make_shared<VecFieldTimeInterp>()};

        using TensorFieldFillPattern_t = TensorFieldFillPattern<dimension /*, rank=1*/>;
        using FieldFillPattern_t       = FieldFillPattern<dimension>;

        std::shared_ptr<FieldFillPattern_t> nonOverwriteFieldFillPattern
            = std::make_shared<FieldFillPattern<dimension>>(); // stateless (mostly)

        std::shared_ptr<TensorFieldFillPattern_t> nonOverwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension /*, rank=1*/>>();

        std::shared_ptr<NoOverlapFillPattern<dimension>> noOverlapFillPattern
            = std::make_shared<NoOverlapFillPattern<dimension>>();

        std::shared_ptr<TensorFieldFillPattern<dimension, 1, /*overwrite_interior=*/true>>
            overwriteInteriorTFfillPattern
            = std::make_shared<TensorFieldFillPattern<dimension, 1, /*overwrite_interior=*/true>>();

        CoarsenOp_ptr mhdFluxCoarseningOp_{std::make_shared<MHDFluxCoarsenOp>()};
        CoarsenOp_ptr mhdVecFluxCoarseningOp_{std::make_shared<MHDVecFluxCoarsenOp>()};
        CoarsenOp_ptr electricFieldCoarseningOp_{std::make_shared<ElectricFieldCoarsenOp>()};

        using FieldRefinePatchStrategyT = FieldRefinePatchStrategy<FieldDataT, BoundaryManagerT>;
        using VectorFieldRefinePatchStrategyT
            = FieldRefinePatchStrategy<VectorFieldDataT, BoundaryManagerT>;
        using MagneticRefinePatchStrategyT
            = MagneticRefinePatchStrategy<VectorFieldDataT, BoundaryManagerT>;
        using FieldRefinePatchStrategyList
            = std::vector<std::shared_ptr<FieldRefinePatchStrategyT>>;
        using VectorFieldRefinePatchStrategyList
            = std::vector<std::shared_ptr<VectorFieldRefinePatchStrategyT>>;
        using MagneticRefinePatchStrategyList
            = std::vector<std::shared_ptr<MagneticRefinePatchStrategyT>>;

        std::shared_ptr<MagneticRefinePatchStrategyT> magneticRefinePatchStrategy_
            = std::make_shared<MagneticRefinePatchStrategyT>(*boundaryManager_);

        FieldRefinePatchStrategyList rhoPatchStrats;
        FieldRefinePatchStrategyList troubledCellPatchStrats;
        FieldRefinePatchStrategyList totalEnergyPatchStrats;
        VectorFieldRefinePatchStrategyList momentumPatchStrats;
        VectorFieldRefinePatchStrategyList elecPatchStrats;
        MagneticRefinePatchStrategyList magPatchStrats;
    };

} // namespace amr
} // namespace PHARE
#endif
