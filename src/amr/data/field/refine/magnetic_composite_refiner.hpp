#ifndef PHARE_MAGNETIC_COMPOSITE_REFINER_HPP
#define PHARE_MAGNETIC_COMPOSITE_REFINER_HPP


#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "field_refiner_kernel.hpp"
#include "composite_field_refiner.hpp"

#include <memory>
#include <stdexcept>
#include <string>


namespace PHARE::amr
{

/**
 * @brief Composable B-field prolongation, restricted to shared (even-normal) faces.
 *
 * B is point-sampled (primal) in its face-normal direction and dual (face-average) in the two
 * tangential directions. The only div-free-compatible, higher-orderable freedom is the tangential
 * dual stencil on the shared faces (copy in normal ⊗ Primitive-B in each tangential dir); the
 * correction is antisymmetric in the child sign ⇒ zero-mean per coarse face ⇒ ∇·B preserved at
 * every order, limited or not. The interior (odd-normal) fine faces are NOT produced here: they are
 * computed div-free by the Tóth-Roe MagneticRefinePatchStrategy::postprocessRefine, which overwrites
 * them and reads only shared faces + dual children. This is the sharedFacesOnly=true specialization
 * of CompositeFieldRefiner.
 */
template<typename GridLayoutT, typename FieldT, std::size_t order, typename Limiter = NoLimiter,
         Representation R = Representation::Average>
using MagneticCompositeRefiner
    = CompositeFieldRefiner<GridLayoutT, FieldT, order, Limiter, /*sharedFacesOnly=*/true, R>;


template<typename GridLayoutT, typename FieldT, Representation R>
std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeMagneticRefineKernel(int order, std::string const& limiter)
{
    auto build = [&limiter]<std::size_t O>() -> std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>> {
        if (limiter == "none" || limiter.empty())
            return std::make_shared<MagneticCompositeRefiner<GridLayoutT, FieldT, O, NoLimiter, R>>();
        if (limiter == "minmod")
            return std::make_shared<
                MagneticCompositeRefiner<GridLayoutT, FieldT, O, core::MinModLimiter, R>>();
        if (limiter == "vanleer")
            return std::make_shared<
                MagneticCompositeRefiner<GridLayoutT, FieldT, O, core::VanLeerLimiter, R>>();
        throw std::runtime_error(
            "makeMagneticRefineKernel: unknown limiter (none|minmod|vanleer): " + limiter);
    };

    switch (order)
    {
        case 2: return build.template operator()<2>();
        case 4: return build.template operator()<4>();
        default:
            throw std::runtime_error(
                "makeMagneticRefineKernel: order must be 2 (Linear) or 4 (Cubic)");
    }
}


} // namespace PHARE::amr


#endif
