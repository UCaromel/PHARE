#ifndef PHARE_AMR_MESSENGERS_CROSS_MODEL_FILL_CONTEXT_HPP
#define PHARE_AMR_MESSENGERS_CROSS_MODEL_FILL_CONTEXT_HPP

#include "core/def.hpp"

#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefinePatchStrategy.h>

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace PHARE::amr
{
enum class ModelKind { MHD, Hybrid };
enum class PairKind { MHD_MHD, MHD_Hyb, Hyb_Hyb };


/**
 * Shared state for recursive ghost fills that cross the single MHD<->Hybrid coupling
 * boundary (N MHD levels below, M hybrid levels above). SAMRAI RefineSchedules inherit
 * one patch strategy and one item set into every coarse-interp recursion frame, so a
 * schedule built for a hybrid level pair can recurse onto MHD level pairs (and its copy
 * transactions can touch ids on patches of the other model). This context provides:
 *
 *  - the level -> model map (from the single firstHybridLevel integer, populated by
 *    MultiPhysicsIntegrator, sole owner of model identity) and the PairKind of any
 *    coarse/fine level pair — valid at every recursion depth since SAMRAI interp temp
 *    levels carry real level numbers;
 *  - symmetric presence hooks: allocate hybrid-only ids (empty) on MHD levels and prim
 *    ids on hybrid levels above the boundary, so every id a crossing transaction touches
 *    is allocated;
 *  - the MHD electromag mirror hook: copy model B->Bpred, E->Eavg on MHD patches so
 *    crossing GhostField schedules read defined values;
 *  - per-population nested-frame spawn fragments (Domain-bucket, no interior clip) used
 *    by DispatchingRefinePatchStrategy at MHD_Hyb boundary frames;
 *  - the crossing-prim registrar: co-registers prim rho/V/P items into the hybrid
 *    particle RefineAlgorithms so prim content rides the same schedules.
 *
 * Constructed by MessengerFactory only when two models are present; hooks and fragments
 * are written solely by MHDHybridMessengerStrategy; presence and mirror are applied by
 * MultiPhysicsIntegrator. Absent context => all consumers take their pre-existing paths.
 */
class CrossModelFillContext
{
public:
    CrossModelFillContext() = default;

    // ------------------------------------------------------------------ level -> model map

    void setFirstHybridLevel(int firstHybridLevel) { firstHybridLevel_ = firstHybridLevel; }

    NO_DISCARD bool hasFirstHybridLevel() const { return firstHybridLevel_ >= 0; }

    NO_DISCARD int firstHybridLevel() const
    {
        assert(hasFirstHybridLevel());
        return firstHybridLevel_;
    }

    NO_DISCARD ModelKind kindOf(int levelNumber) const
    {
        assert(hasFirstHybridLevel());
        return levelNumber < firstHybridLevel_ ? ModelKind::MHD : ModelKind::Hybrid;
    }

    NO_DISCARD PairKind pairKind(int coarseLn, int fineLn) const
    {
        assert(coarseLn < fineLn);
        if (kindOf(coarseLn) == ModelKind::MHD)
            return kindOf(fineLn) == ModelKind::MHD ? PairKind::MHD_MHD : PairKind::MHD_Hyb;
        return PairKind::Hyb_Hyb; // single boundary: no MHD level above a hybrid one
    }

    // ------------------------------------------------------- symmetric component presence

    using PatchHook = std::function<void(SAMRAI::hier::Patch&, double)>;

    void addMHDLevelPresence(PatchHook hook) { mhdPresence_.push_back(std::move(hook)); }
    void addHybridLevelPresence(PatchHook hook) { hybridPresence_.push_back(std::move(hook)); }

    void applyPresence(SAMRAI::hier::Patch& patch, int levelNumber, double allocTime) const
    {
        if (kindOf(levelNumber) == ModelKind::MHD)
            for (auto const& hook : mhdPresence_)
                hook(patch, allocTime);
        else if (levelNumber > firstHybridLevel_)
            for (auto const& hook : hybridPresence_)
                hook(patch, allocTime);
        // levelNumber == firstHybridLevel_: prims already allocated by the coupled messenger
    }

    // --------------------------------------------- EM value mirror (B->Bpred, E->Eavg)

    void setMHDElectromagMirror(PatchHook hook) { electromagMirror_ = std::move(hook); }

    void applyMHDElectromagMirror(SAMRAI::hier::Patch& patch, int levelNumber, double time) const
    {
        if (electromagMirror_ and kindOf(levelNumber) == ModelKind::MHD)
            electromagMirror_(patch, time);
    }

    // ----------------------------------- nested-frame Domain-bucket spawn fragments

    void setNestedSpawnFragment(std::string const& popName,
                                std::shared_ptr<SAMRAI::xfer::RefinePatchStrategy> fragment)
    {
        nestedSpawnFragments_[popName] = std::move(fragment);
    }

    NO_DISCARD SAMRAI::xfer::RefinePatchStrategy*
    nestedSpawnFragment(std::string const& popName) const
    {
        auto const it = nestedSpawnFragments_.find(popName);
        return it == nestedSpawnFragments_.end() ? nullptr : it->second.get();
    }

    // ------------------------------------------------- crossing prim item registrar

    using AlgoRegistrar = std::function<void(SAMRAI::xfer::RefineAlgorithm&)>;

    void setCrossingPrimRegistrar(AlgoRegistrar registrar) { primRegistrar_ = std::move(registrar); }

    void applyCrossingPrimItems(SAMRAI::xfer::RefineAlgorithm& algorithm) const
    {
        if (primRegistrar_)
            primRegistrar_(algorithm);
    }


private:
    int firstHybridLevel_ = -1;

    std::vector<PatchHook> mhdPresence_;
    std::vector<PatchHook> hybridPresence_;
    PatchHook electromagMirror_;

    std::map<std::string, std::shared_ptr<SAMRAI::xfer::RefinePatchStrategy>>
        nestedSpawnFragments_;

    AlgoRegistrar primRegistrar_;
};

} // namespace PHARE::amr

#endif
