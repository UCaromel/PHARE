#ifndef PHARE_MAGNETIC_COMPOSITE_REFINER_HPP
#define PHARE_MAGNETIC_COMPOSITE_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/types.hpp"

#include "coarse_cell_round_out.hpp"
#include "composite_field_refiner.hpp"
#include "field_refiner_kernel.hpp"
#include "limited_field_refiner.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>


namespace PHARE::amr
{

/**
 * @brief Stage 1 of the Balsara divB-free B prolongation: fill every fine face of a B
 * component from its own coarse faces, with whichever field kernel it decorates.
 *
 * The decorated kernel is master's composite table or a limited kernel (makeRefineKernel): the
 * primal-even direction (collocated with coarse) an exact copy, primal-odd (a new fine face) a
 * half-point interpolation, the dual directions the child ladder, slope-limited when a rule is
 * selected. This stage makes no divB claim; stage 2, the cross-component touch-up
 * (magnetic_refine_patch_strategy.hpp), is what establishes it.
 *
 * What is magnetic here, and the whole reason for a decorator, is the region gathered: stage 2
 * reads the shared faces of a whole coarse cell, so stage 1 has to have written them.
 */
template<typename GridLayoutT, typename FieldT>
class MagneticRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using Kernel = IFieldRefineKernel<GridLayoutT, FieldT>;

public:
    explicit MagneticRefiner(std::unique_ptr<Kernel> stage1)
        : stage1_{std::move(stage1)}
    {
    }

    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        stage1_->refineBox(sourceField, destinationField,
                           gatherRegion_(intersectionBox, centering, destFieldBox), centering,
                           destFieldBox, sourceFieldBox, ratio);
    }

    int coarseStencilWidth() const override { return stage1_->coarseStencilWidth(); }

private:
    /**
     * @brief The fine region to gather for an overlap: whole coarse cells, so the stage-2 touch-up
     * finds every face it reads already written.
     *
     * A B component is primal in at most one direction, its face normal. Without one — the
     * out-of-plane components, By and Bz in 1D, Bz in 2D — the component has no interior face,
     * hence no divB neighbourhood to protect and nothing to round out.
     *
     * With one, round out to whole coarse cells (coarse_cell_round_out.hpp), then re-clip: the
     * overlap arrives already clipped to the allocation, so rounding can push it back out. Widening
     * the gather is safe because the fill writes only still-NaN fine indices, and the fine indices
     * rounded in share the coarse anchors, and the stencil reach, of those already gathered. Faces
     * a rounded B copy already wrote (copiesWholeCoarseCells) are kept, so the touch-up reads
     * same-level data wherever it exists.
     */
    static SAMRAI::hier::Box
    gatherRegion_(SAMRAI::hier::Box const& overlap,
                  std::array<core::QtyCentering, dimension> const& centering,
                  SAMRAI::hier::Box const& destFieldBox)
    {
        auto const primalCount = core::sum_from(
            centering, [](auto const c) { return c == core::QtyCentering::primal ? 1 : 0; });

        if (primalCount > 1)
            throw std::runtime_error(
                "magnetic refiner expects at most one primal (normal) direction");

        if (primalCount == 0)
            return overlap;

        return roundFieldBoxOutToCoarseCells<dimension>(overlap, centering) * destFieldBox;
    }

    std::unique_ptr<Kernel> stage1_;
};


// master's table when the settings are master's (bitwise), the limited kernel otherwise.
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(FieldRefinementOrder const order, LimiterSettings const& settings,
                         std::shared_ptr<RefineProbe> probe)
{
    return std::make_unique<MagneticRefiner<GridLayoutT, FieldT>>(
        makeRefineKernel<GridLayoutT, FieldT>(order, settings, std::move(probe)));
}


} // namespace PHARE::amr


#endif
