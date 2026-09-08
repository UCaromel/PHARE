#ifndef PHARE_MAGNETIC_COMPOSITE_REFINER_HPP
#define PHARE_MAGNETIC_COMPOSITE_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "field_refiner_kernel.hpp"
#include "composite_field_refiner.hpp"

#include <memory>
#include <stdexcept>


namespace PHARE::amr
{

/**
 * @brief Stage 1 of the Balsara ADPT divB-free B prolongation: fill every fine face of a B
 * component from its own coarse faces.
 *
 * Per component: primal-even direction (collocated with coarse) is an exact copy, primal-odd (new
 * point, eg. a new fine face) a directionalInterp half-point, dual directions the
 * directionalProlongation — i.e. CompositeFieldRefiner with the magnetic round-out on. Stage-2, the
 * ADPTMagneticRefinePatchStrategy establishes that.
 */
template<typename GridLayoutT, typename FieldT, std::size_t order>
using MagneticCompositeRefiner
    = CompositeFieldRefiner<GridLayoutT, FieldT, order, /*isMagnetic=*/true>;


template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(FieldRefinementOrder const order)
{
    switch (order)
    {
        case FieldRefinementOrder::Linear:
            return std::make_unique<MagneticCompositeRefiner<
                GridLayoutT, FieldT, static_cast<std::size_t>(FieldRefinementOrder::Linear)>>();
        case FieldRefinementOrder::Cubic:
            return std::make_unique<MagneticCompositeRefiner<
                GridLayoutT, FieldT, static_cast<std::size_t>(FieldRefinementOrder::Cubic)>>();
    }

    throw std::logic_error("invalid magnetic field-refinement order");
}


} // namespace PHARE::amr


#endif
