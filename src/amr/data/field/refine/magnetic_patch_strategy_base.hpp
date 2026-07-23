#ifndef PHARE_AMR_MAGNETIC_PATCH_STRATEGY_BASE_HPP
#define PHARE_AMR_MAGNETIC_PATCH_STRATEGY_BASE_HPP

#include "SAMRAI/xfer/RefinePatchStrategy.h"

namespace PHARE::amr
{
/**
 * @brief Non-templated polymorphic base for the magnetic RefinePatchStrategy family.
 *
 * Lets the messengers hold either the legacy Tóth-Roe strategy
 * (MagneticRefinePatchStrategy) or the ADPT div-free touch-up strategy
 * (ADPTMagneticRefinePatchStrategy) behind a single pointer type, selected by
 * RefinementConfig.order. Declares the small contract common to both (registerIDs)
 * and provides the shared no-op boilerplate SAMRAI requires. The concrete refine
 * operator (not the strategy) governs the coarse stencil width through SAMRAI's
 * getMaxStencilGhosts(), which takes the max over strategy and ops — so the
 * strategy reports a width of 1 and the order-4 kernel width (order/2) wins.
 */
class MagneticPatchStrategyBase : public SAMRAI::xfer::RefinePatchStrategy
{
public:
    virtual void registerIDs(int b_id) = 0;

    // shared boilerplate: ADPT inherits these; the legacy TR strategy keeps its own
    // identical overrides (harmless re-override, keeps the TR file untouched beyond
    // the base-class swap).
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch&, double const,
                                       SAMRAI::hier::IntVector const&) override
    {
    }

    void preprocessRefine(SAMRAI::hier::Patch&, SAMRAI::hier::Patch const&,
                          SAMRAI::hier::Box const&, SAMRAI::hier::IntVector const&) override
    {
    }

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector(dim, 1); // stage-1 width comes from the op; SAMRAI maxes
    }

    ~MagneticPatchStrategyBase() override = default;
};

} // namespace PHARE::amr

#endif // PHARE_AMR_MAGNETIC_PATCH_STRATEGY_BASE_HPP
