#ifndef PHARE_MESSENGER_UTILS_HPP
#define PHARE_MESSENGER_UTILS_HPP

#include "amr/data/field/field_geometry.hpp"
#include "amr/data/field/refine/magnetic_refine_patch_strategy.hpp"
#include "amr/resources_manager/amr_utils.hpp"

#include <SAMRAI/hier/BoxContainer.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/hier/PatchLevel.h>
#include <SAMRAI/xfer/CoarsenAlgorithm.h>
#include <SAMRAI/xfer/CoarsenSchedule.h>
#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefineSchedule.h>
#include <SAMRAI/xfer/RefineTransactionFactory.h>

#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace PHARE::amr
{

// BfieldComms holds all SAMRAI state for B-field AMR communication.
// Shared between MHDMessenger and HybridHybridMessengerStrategy.
// Uses raw SAMRAI algorithms directly: pools lack named-slot support for the 2 distinct
// B-field registration phases (init / regrid). First candidate for
// pool-consistency pass once the file structure is stable.
template<typename ResourcesManagerT, typename VectorFieldDataT>
struct BfieldComms
{
    SAMRAI::xfer::RefineAlgorithm BalgoInit;
    SAMRAI::xfer::RefineAlgorithm BregridAlgo;

    std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> magInitRefineSchedules_;

    MagneticRefinePatchStrategy<ResourcesManagerT, VectorFieldDataT> magneticRefinePatchStrategy_;
    std::vector<std::shared_ptr<MagneticRefinePatchStrategy<ResourcesManagerT, VectorFieldDataT>>>
        magneticPatchStratPerGhostRefiner_;

    explicit BfieldComms(ResourcesManagerT& rm)
        : magneticRefinePatchStrategy_{rm}
    {
    }

    void magneticRegriding_(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                            std::shared_ptr<SAMRAI::hier::PatchLevel> const& level,
                            std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                            double const initDataTime)
    {
        auto magSchedule = BregridAlgo.createSchedule(level, oldLevel,
                                                      level->getNextCoarserHierarchyLevelNumber(),
                                                      hierarchy, &magneticRefinePatchStrategy_);
        magSchedule->fillData(initDataTime);
    }
};


// RefluxChannel holds the SAMRAI state for one coarsen→ghost-refill channel.
// MHDMessenger holds four channels (E, HydroX, HydroY, HydroZ); HybridHybridMessengerStrategy holds
// one.
struct RefluxChannel
{
    SAMRAI::xfer::CoarsenAlgorithm coarsenAlgo;
    SAMRAI::xfer::RefineAlgorithm refineAlgo;
    std::map<int, std::shared_ptr<SAMRAI::xfer::CoarsenSchedule>> coarsenSchedules;
    std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> refineSchedules;

    explicit RefluxChannel(int dim)
        : coarsenAlgo{SAMRAI::tbox::Dimension{static_cast<unsigned short>(dim)}}
    {
    }

    void registerLevel(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                       std::shared_ptr<SAMRAI::hier::PatchLevel> const& level, int levelNumber,
                       int rootLevelNumber)
    {
        refineSchedules[levelNumber] = refineAlgo.createSchedule(level);
        if (levelNumber != rootLevelNumber)
        {
            auto const coarseLevel        = hierarchy->getPatchLevel(levelNumber - 1);
            coarsenSchedules[levelNumber] = coarsenAlgo.createSchedule(coarseLevel, level);
        }
    }

    void reflux(int fineLevelNumber, int coarserLevelNumber, double syncTime)
    {
        coarsenSchedules[fineLevelNumber]->coarsenData();
        refineSchedules[coarserLevelNumber]->fillData(syncTime);
    }
};


// EfieldComms holds all SAMRAI state for E-field AMR init/refill communication.
// Parallel to BfieldComms: raw RefineAlgorithm + schedule map, no patch strategy.
// Accepts any refine op — same-type or cross-type.
struct EfieldComms
{
    SAMRAI::xfer::RefineAlgorithm algo;
    std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> schedules;

    void createSchedule(int levelNumber,
                        std::shared_ptr<SAMRAI::hier::PatchLevel> const& level,
                        int coarserLevelNumber,
                        std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy)
    {
        schedules[levelNumber]
            = algo.createSchedule(level, nullptr, coarserLevelNumber, hierarchy);
    }

    void fill(int levelNumber, double fillTime) { schedules[levelNumber]->fillData(fillTime); }
};


// HydroChannelSpec packs all parameters needed to register one hydro RefluxChannel:
// coarsen (src→dst) and ghost-refill refine for rho, rhoV, Etot.
struct HydroChannelSpec
{
    int rho_dst, rhoV_dst, Etot_dst; // coarsen destination IDs (MHD timeFluxes)
    int rho_src, rhoV_src, Etot_src; // coarsen source IDs (MHD or Hybrid flux sums)
    std::shared_ptr<SAMRAI::hier::CoarsenOperator> scalarCoarsenOp; // for rho, Etot
    std::shared_ptr<SAMRAI::hier::CoarsenOperator> vecCoarsenOp;    // for rhoV
    std::shared_ptr<SAMRAI::hier::RefineOperator> scalarRefineOp;   // ghost refill rho, Etot
    std::shared_ptr<SAMRAI::hier::RefineOperator> vecRefineOp;      // ghost refill rhoV
    std::shared_ptr<SAMRAI::xfer::RefineTransactionFactory> fillPattern;
};


// registerHydroChannel registers coarsen (src→dst) + ghost-refill refine for
// rho/rhoV/Etot on a single RefluxChannel. Extracted from MHDMHDRefluxComms.
inline void registerHydroChannel(RefluxChannel& channel, HydroChannelSpec const& spec)
{
    channel.coarsenAlgo.registerCoarsen(spec.rho_dst, spec.rho_src, spec.scalarCoarsenOp);
    channel.coarsenAlgo.registerCoarsen(spec.rhoV_dst, spec.rhoV_src, spec.vecCoarsenOp);
    channel.coarsenAlgo.registerCoarsen(spec.Etot_dst, spec.Etot_src, spec.scalarCoarsenOp);
    channel.refineAlgo.registerRefine(spec.rho_dst, spec.rho_dst, spec.rho_dst,
                                      spec.scalarRefineOp, spec.fillPattern);
    channel.refineAlgo.registerRefine(spec.rhoV_dst, spec.rhoV_dst, spec.rhoV_dst,
                                      spec.vecRefineOp, spec.fillPattern);
    channel.refineAlgo.registerRefine(spec.Etot_dst, spec.Etot_dst, spec.Etot_dst,
                                      spec.scalarRefineOp, spec.fillPattern);
}


// setNaNsOnFieldGhosts / setNaNsOnVecfieldGhosts
//
// Sets NaNs on ghost nodes so refinement operators can identify nodes not yet touched
// by a schedule copy. Required when FieldVariable::fineBoundaryRepresentsVariable=false,
// which causes the copy to run before refinement, leaving level ghost nodes at NaN.
//
// Implementation: patch-local — for each patch, compute the ghost layer by removing
// the interior field box from the full ghost field box. This is equivalent to the
// collection-based approach (pre-gathering all level boxes with level.getBoxes() and
// calling removeIntersections on the collection), since both produce the same ghost region.
template<typename GridLayoutT, typename FieldT>
void setNaNsOnFieldGhosts(FieldT& field, SAMRAI::hier::Patch const& patch)
{
    static constexpr auto dimension = GridLayoutT::dimension;

    auto const qty         = field.physicalQuantity();
    using qty_t            = std::decay_t<decltype(qty)>;
    using field_geometry_t = FieldGeometry<GridLayoutT, qty_t>;

    auto const box    = patch.getBox();
    auto const layout = layoutFromPatch<GridLayoutT>(patch);

    auto const gbox  = layout.AMRGhostBoxFor(field.physicalQuantity());
    auto const sgbox = samrai_box_from(gbox);
    auto const fbox  = field_geometry_t::toFieldBox(box, qty, layout);

    SAMRAI::hier::BoxContainer ghostLayerBoxes{};
    ghostLayerBoxes.removeIntersections(sgbox, fbox);

    for (auto const& gb : ghostLayerBoxes)
        for (auto const& index : layout.AMRToLocal(phare_box_from<dimension>(gb)))
            field(index) = std::numeric_limits<typename FieldT::value_type>::quiet_NaN();
}

template<typename GridLayoutT, typename FieldT, typename RM>
void setNaNsOnFieldGhosts(FieldT& field, SAMRAI::hier::PatchLevel const& level, RM& rm)
{
    for (auto& patch : rm.enumerate(level, field))
        setNaNsOnFieldGhosts<GridLayoutT>(field, *patch);
}

template<typename GridLayoutT, typename VecFieldT, typename RM>
void setNaNsOnVecfieldGhosts(VecFieldT& vf, SAMRAI::hier::PatchLevel const& level, RM& rm)
{
    for (auto& patch : rm.enumerate(level, vf))
        for (auto& component : vf)
            setNaNsOnFieldGhosts<GridLayoutT>(component, *patch);
}

} // namespace PHARE::amr
#endif
