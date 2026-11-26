#ifndef PHARE_FIELD_REFINER_KERNEL_HPP
#define PHARE_FIELD_REFINER_KERNEL_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/def.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>


namespace PHARE::amr
{

/**
 * @brief Supported field-refinement orders, shared by RefinementConfig and the kernel factories.
 *
 * Linear (order 2) is the only supported value for now. A raw dict int is validated against this
 * enum in exactly one place, RefinementConfig::FROM; the factories below take the enum and so need
 * no order check of their own.
 */
enum class FieldRefinementOrder { Linear = 2 };


//! Slope rule of the dual (cell-centred) rows, see core/numerics/refinement/slope_rules.hpp.
enum class FieldRefinementLimiter { None, VanLeer, MC, MCGate, AO5Z, MP };

//! How the 1D rows combine in several dimensions: master's tensor product of per-direction rows,
//! or sequential 1D sweeps averaged over direction orders.
enum class RefinementTensorMode { Product, Sweeps };

//! Direction orders averaged in Sweeps mode: all permutations, the cyclic ones, or one.
enum class RefinementSweepOrders { All, Cyclic, Single };

/**
 * @brief Limiting applied by a field-refinement kernel; the defaults are master's table.
 *
 * energyRow selects the total-energy operator, not the kernel of the other variables, so it is
 * not part of isMasterTable.
 */
struct LimiterSettings
{
    FieldRefinementLimiter limiter    = FieldRefinementLimiter::None;
    std::optional<double> param       = std::nullopt; // rule default when empty
    RefinementTensorMode tensor       = RefinementTensorMode::Product;
    RefinementSweepOrders sweepOrders = RefinementSweepOrders::All;
    bool energyRow                    = false;

    //! true: master's compile-time table is used, bitwise
    NO_DISCARD bool isMasterTable() const
    {
        return limiter == FieldRefinementLimiter::None and tensor == RefinementTensorMode::Product;
    }

    bool operator==(LimiterSettings const&) const = default;
};

struct RefineProbe;

/**
 * @brief Runtime-dispatched field-refinement seam.
 *
 * The refinement order is a runtime dict value, so it must not become a template parameter of the
 * refine operators and messengers: that would carry it into the build permutation id and multiply
 * the compiled permutations. This seam is what keeps it runtime — operators and messengers depend
 * on the interface, never on the stencil tables.
 *
 * refineBox() takes one overlap box and loops over its fine indices internally, so the virtual
 * call is paid once per box rather than once per index.
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
                           SAMRAI::hier::IntVector const& ratio) const
        = 0;

    /**
     * @brief Coarse-cell stencil half-width this kernel reads around each anchor.
     *
     * SAMRAI provisions coarse (source) ghost layers from RefineOperator::getStencilWidth before
     * prolongation. order 2 reads ±1 coarse cell (both the dual ±¼ ladder and the primal
     * midpoint), so it is order/2. Reported up through the holding operator.
     */
    virtual int coarseStencilWidth() const = 0;

    virtual ~IFieldRefineKernel() = default;
};


/**
 * @brief Build the field-refinement kernel of an order and limiter settings: master's composite
 * table for the default settings, the limited kernel of the selected rule otherwise.
 *
 * Defined with the concrete kernels (limited_field_refiner.hpp); declared here so the refine
 * operators and the messengers depend only on the seam.
 */
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel(FieldRefinementOrder order, LimiterSettings const& settings = {},
                 std::shared_ptr<RefineProbe> probe = nullptr);

/**
 * @brief Build the stage-1 magnetic refinement kernel of the div-free magnetic prolongation.
 *
 * Fills every fine face per component with the composite tensor stencils, and makes no ∇·B claim:
 * ∇·B-freeness comes from the stage-2 touch-up (magnetic_refine_patch_strategy.hpp).
 */
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(FieldRefinementOrder order, LimiterSettings const& settings = {},
                         std::shared_ptr<RefineProbe> probe = nullptr);


} // namespace PHARE::amr


#endif
