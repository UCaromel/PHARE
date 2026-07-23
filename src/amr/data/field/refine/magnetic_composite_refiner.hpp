#ifndef PHARE_MAGNETIC_COMPOSITE_REFINER_HPP
#define PHARE_MAGNETIC_COMPOSITE_REFINER_HPP


#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "field_refiner_kernel.hpp"
#include "composite_field_refiner.hpp"

#include <memory>
#include <string>


namespace PHARE::amr
{

/**
 * @brief Stage 1 of the Balsara ADPT divB-free B prolongation.
 *
 * Fills ALL fine faces of a B component from its coarse faces, per component (primal-even
 * direction: exact copy; primal-odd direction: directionalInterp half-point; dual directions:
 * directionalProlongation ±¼ ladder). This is exactly CompositeFieldRefiner run unrestricted —
 * B carries no special stage-1 gating.
 *
 * Its stage-2 partner is ADPTMagneticRefinePatchStrategy::postprocessRefine: an order-independent
 * divB touch-up that equalizes the 2^d subzone divergences of each coarse zone via a closed-form
 * min-norm correction, making the composite result divB-free exactly regardless of the stage-1
 * order/limiter.
 *
 * `makeMagneticRefineKernel` is kept as a named forward (rather than call-site aliasing to
 * `makeRefineKernel` directly) as the hook for a future B-specific stage-1 kernel (e.g. a
 * WENO-AO limiter set tuned for divB preservation) without touching call sites.
 */
template<typename GridLayoutT, typename FieldT, Representation R>
std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(int order, std::string const& limiter)
{
    return makeRefineKernel<GridLayoutT, FieldT, R>(order, limiter);
}


} // namespace PHARE::amr


#endif
