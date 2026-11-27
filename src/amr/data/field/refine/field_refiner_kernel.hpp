#ifndef PHARE_FIELD_REFINER_KERNEL_HPP
#define PHARE_FIELD_REFINER_KERNEL_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>


namespace PHARE::amr
{

// EXP (exp/o4-refine-limit, do not merge): env-knob prototype of limited order-4 refinement.
//   PHARE_REFINE_SLOPE=mp|ao5z|lin   5-point rule on the cell-centred hydro rows (lin = unlimited
//                                    slope through the limited path), sequential symmetrised sweeps
//   PHARE_REFINE_ENERGY=prim         energy row: q = Etot - KE prolonged, E rebuilt with children KE
//   PHARE_REFINE_LIMIT_GHOSTS=0      level creation and regrid only (default: ghost fills too)
// Unset = the base kernel, bitwise. The messenger says which hydro row is being filled.
namespace refine_exp
{
    enum class Qty { Other, Rho, RhoV, Etot };
    inline Qty qty       = Qty::Other;
    inline int comp      = 0; // vector component being refined
    inline int ctx       = 0; // 0 other, 1 level creation / regrid, 2 ghost fill
    inline int coarseKey = 0; // binary exponent of the coarse mesh size of the current refine
    inline long eApplied = 0, eMissed = 0, nLimited = 0;

    inline int rule()
    {
        static int const r = [] {
            char const* e = std::getenv("PHARE_REFINE_SLOPE");
            std::string const v = e ? e : "";
            return v == "mp" ? 1 : v == "ao5z" ? 2 : v == "lin" ? 3 : 0;
        }();
        return r;
    }
    inline bool energy()
    {
        static bool const b = [] {
            char const* e = std::getenv("PHARE_REFINE_ENERGY");
            return e and std::string{e} == "prim";
        }();
        return b;
    }
    inline bool limitGhosts()
    {
        static bool const b = [] {
            char const* e = std::getenv("PHARE_REFINE_LIMIT_GHOSTS");
            return not(e and std::string{e} == "0");
        }();
        return b;
    }
    inline bool active()
    {
        return rule() != 0 and qty != Qty::Other and (ctx == 1 or (ctx == 2 and limitGhosts()));
    }

    struct KeyHash
    {
        std::size_t operator()(std::array<int, 4> const& k) const
        {
            std::size_t h = 1469598103934665603ull;
            for (int const i : k)
                h = (h ^ static_cast<std::size_t>(static_cast<unsigned>(i))) * 1099511628211ull;
            return h;
        }
    };
    struct CoarseHydro
    {
        std::array<double, 4> v{}; // rho, rhoV
        unsigned have = 0;         // bit per slot
        unsigned blockDone = 0;    // bit per slot: the 5^d block around this cell is stored
    };
    // coarse rho, rhoV stored by their fills for the Etot fill of the same group (the three rows
    // are separate schedules: the Etot refine does not see the other rows' coarse temporaries)
    inline std::unordered_map<std::array<int, 4>, CoarseHydro, KeyHash> ecache;

    // scope guard set by the messenger around one row's fill
    struct Row
    {
        Row(Qty const q, int const c)
        {
            qty = q;
            ctx = c;
        }
        ~Row()
        {
            qty = Qty::Other;
            ctx = 0;
        }
    };
} // namespace refine_exp

/**
 * @brief Supported field-refinement orders, shared by RefinementConfig and the kernel factories,
 * which dispatch the value to the corresponding compile-time stencil.
 */
enum class FieldRefinementOrder { Linear = 2, Cubic = 4 };

/**
 * @brief Runtime-dispatched field-refinement seam.
 *
 * The order is passed as a value so it adds no instantiations of the refine operators and
 * messengers, which depend on this interface rather than on the stencil tables.
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
 * @brief Build a composite field-refinement kernel for a given order.
 *
 * Defined with the concrete kernels (composite_field_refiner.hpp); declared here so the refine
 * operators and the messengers depend only on the seam.
 */
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel(FieldRefinementOrder order);

/**
 * @brief Build the stage-1 magnetic refinement kernel of the div-free magnetic prolongation.
 *
 * Fills every fine face per component with the composite tensor stencils, and makes no ∇·B claim:
 * ∇·B-freeness comes from the stage-2 touch-up (magnetic_refine_patch_strategy.hpp).
 */
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(FieldRefinementOrder order);


} // namespace PHARE::amr


#endif
