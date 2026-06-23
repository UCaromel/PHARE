#ifndef PHARE_FIELD_REFINER_KERNEL_HPP
#define PHARE_FIELD_REFINER_KERNEL_HPP


#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>


namespace PHARE::amr
{

/**
 * @brief Prolongation world the kernel's dual ±¼ row is built in.
 *
 * Average    — coarse datum is a cell average; the dual child means back to it (conservative). This
 *              is the time-integration / conservative-ladder world. Default everywhere.
 * PointValue — coarse datum is a point value at the node; the dual child is a plain point Lagrange
 *              at ±¼ (NON-conservative). Required by the flux-stage interlevel ghost fills at order
 *              ≥4, where it differs from Average by an O(H²/24·u″) term (invisible at order 2).
 *
 * Only the DUAL 1-D row differs between worlds; primal even (copy) and primal-odd (half-point) rows
 * are world-shared.
 */
enum class Representation
{
    Average,
    PointValue
};


/**
 * @brief Runtime-dispatched field-refinement seam.
 *
 * Mirrors the contract of the legacy policy functors (constructed per refine() with
 * {centering, destFieldBox, sourceFieldBox, ratio}, then applied per fine index), but hoists
 * the single virtual call to PER OVERLAP BOX granularity: refineBox() receives one intersection
 * box and loops over its fine indices internally, so the virtual-dispatch cost is paid once per
 * box rather than once per index.
 *
 * Concrete kernels (composite Linear/Cubic, magnetic shared-face) implement refineBox().
 * order == 0 (legacy) never reaches this seam: the messengers route order-0 to the unchanged
 * templated FieldRefineOperator<...,Policy> instead.
 */
template<typename GridLayoutT, typename FieldT>
struct IFieldRefineKernel
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    virtual void refineBox(FieldT const& sourceField, FieldT& destinationField,
                           SAMRAI::hier::Box const& intersectionBox,
                           std::array<core::QtyCentering, dimension> const& centering,
                           SAMRAI::hier::Box const& destFieldBox,
                           SAMRAI::hier::Box const& sourceFieldBox,
                           SAMRAI::hier::IntVector const& ratio) const = 0;

    /**
     * @brief Coarse-cell stencil half-width this kernel reads around each anchor.
     *
     * SAMRAI provisions coarse (source) ghost layers from RefineOperator::getStencilWidth before
     * prolongation. order 2 reads ±1 coarse cell, order 4 reads ±2 (both the dual ±¼ ladder and
     * the primal midpoint), so it is order/2. Reported up through the holding operator. A fixed
     * width would under-provision the 2nd coarse ghost for Cubic ⇒ stale/OOB reads at C-F borders.
     */
    virtual int coarseStencilWidth() const = 0;

    virtual ~IFieldRefineKernel() = default;
};


/**
 * @brief Build a composite field-refinement kernel for a given order/limiter.
 *
 * order: 2 = Linear, 4 = Cubic (per the dual ±1/4 ladder; degree-2 skipped by even-term
 * cancellation). limiter: "none" | "minmod" | "vanleer". Definition lives with the concrete
 * composite kernels (Step 3); declared here so the additive operators and the messengers can
 * depend only on the seam.
 */
template<typename GridLayoutT, typename FieldT, Representation R = Representation::Average>
std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel(int order, std::string const& limiter);

/**
 * @brief Build a magnetic shared-face refinement kernel (tangential dual stencil only).
 *
 * Interior fine faces remain owned by the Tóth-Roe MagneticRefinePatchStrategy and are NOT
 * produced here. The shared-face tangential correction is antisymmetric in the child sign, so
 * ∇·B is preserved at every order, limited or not.
 */
template<typename GridLayoutT, typename FieldT, Representation R = Representation::Average>
std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(int order, std::string const& limiter);


} // namespace PHARE::amr


#endif
