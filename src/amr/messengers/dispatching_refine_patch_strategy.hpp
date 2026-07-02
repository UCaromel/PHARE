#ifndef PHARE_AMR_MESSENGERS_DISPATCHING_REFINE_PATCH_STRATEGY_HPP
#define PHARE_AMR_MESSENGERS_DISPATCHING_REFINE_PATCH_STRATEGY_HPP

#include "amr/messengers/cross_model_fill_context.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/BoxContainer.h>
#include <SAMRAI/hier/IntVector.h>
#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/tbox/Dimension.h>
#include <SAMRAI/xfer/RefinePatchStrategy.h>

#include <functional>
#include <memory>
#include <utility>

namespace PHARE::amr
{
/**
 * RefinePatchStrategy facade routing each callback to the physically correct fragment for
 * the (coarse, fine) level pair of the current recursion frame. SAMRAI RefineSchedules bind
 * ONE strategy at the top schedule and inherit it verbatim into every coarse-interp
 * recursion frame; when recursion crosses the MHD<->Hybrid boundary the inherited strategy
 * is wrong-physics machinery for the frame it runs on. Interp temp levels carry real level
 * numbers, so PairKind is computable at every depth from the callback args alone.
 *
 * A missing fragment (fragments_(kind) == nullptr) means no-op for that pair — e.g. the
 * coupled spawn only acts at MHD_Hyb frames; Hyb_Hyb frames are handled by the pool's
 * split operator and MHD_MHD frames by field refine operators.
 */
class DispatchingRefinePatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
public:
    using FragmentFn = std::function<SAMRAI::xfer::RefinePatchStrategy*(PairKind)>;

    DispatchingRefinePatchStrategy(std::shared_ptr<CrossModelFillContext const> ctx,
                                   FragmentFn fragments, SAMRAI::hier::IntVector stencilWidth)
        : ctx_{std::move(ctx)}
        , fragments_{std::move(fragments)}
        , stencilWidth_{std::move(stencilWidth)}
    {
    }

    // matches the nullptr-strategy behavior of the schedules being wrapped
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch&, double,
                                       SAMRAI::hier::IntVector const&) override
    {
    }

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& /*dim*/) const override
    {
        return stencilWidth_;
    }

    void preprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                          SAMRAI::hier::Box const& fine_box,
                          SAMRAI::hier::IntVector const& ratio) override
    {
        if (auto* fragment = resolve_(fine, coarse))
            fragment->preprocessRefine(fine, coarse, fine_box, ratio);
    }

    void postprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                           SAMRAI::hier::Box const& fine_box,
                           SAMRAI::hier::IntVector const& ratio) override
    {
        if (auto* fragment = resolve_(fine, coarse))
            fragment->postprocessRefine(fine, coarse, fine_box, ratio);
    }

    void preprocessRefineBoxes(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                               SAMRAI::hier::BoxContainer const& fine_boxes,
                               SAMRAI::hier::IntVector const& ratio) override
    {
        if (auto* fragment = resolve_(fine, coarse))
            fragment->preprocessRefineBoxes(fine, coarse, fine_boxes, ratio);
    }

    void postprocessRefineBoxes(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                                SAMRAI::hier::BoxContainer const& fine_boxes,
                                SAMRAI::hier::IntVector const& ratio) override
    {
        if (auto* fragment = resolve_(fine, coarse))
            fragment->postprocessRefineBoxes(fine, coarse, fine_boxes, ratio);
    }


private:
    SAMRAI::xfer::RefinePatchStrategy* resolve_(SAMRAI::hier::Patch const& fine,
                                                SAMRAI::hier::Patch const& coarse) const
    {
        return fragments_(
            ctx_->pairKind(coarse.getPatchLevelNumber(), fine.getPatchLevelNumber()));
    }

    std::shared_ptr<CrossModelFillContext const> ctx_;
    FragmentFn fragments_;
    SAMRAI::hier::IntVector stencilWidth_;
};

} // namespace PHARE::amr

#endif
