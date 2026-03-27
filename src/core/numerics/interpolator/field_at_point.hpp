#ifndef PHARE_CORE_NUMERICS_INTERPOLATOR_FIELD_AT_POINT_HPP
#define PHARE_CORE_NUMERICS_INTERPOLATOR_FIELD_AT_POINT_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "core/numerics/interpolator/interpolator.hpp"
#include "core/utilities/point/point.hpp"

namespace PHARE::core
{

/**
 * @brief Interpolate a scalar field at an arbitrary physical-space point.
 *
 * This class performs the same polynomial interpolation as @ref Interpolator
 * but accepts a physical-space @ref Point instead of a particle's (iCell, delta)
 * pair.  It works for any mix of primal/dual centerings (node-, cell-, face-, or
 * edge-centred fields).
 *
 * Physical coordinates follow the same convention as @ref GridLayout::fieldNodeCoordinates:
 * the coordinate of AMR index @c i with spacing @c dx is simply @c i * dx (no
 * domain-origin offset is applied).
 *
 * @tparam dim         Spatial dimension (1, 2, or 3).
 * @tparam interpOrder Interpolation order (1, 2, or 3).
 */
template<std::size_t dim, std::size_t interpOrder>
class FieldAtPoint : private Interpolator<dim, interpOrder>
{
    using Base = Interpolator<dim, interpOrder>;

public:
    /**
     * @brief Evaluate @p field at @p physPoint using order-interpOrder interpolation.
     *
     * @tparam quantity   Physical quantity enum value (determines centering per dimension).
     * @param  layout     Grid layout of the patch that owns @p field.
     * @param  field      Field to interpolate from.
     * @param  physPoint  Target point in physical coordinates (AMR-index * meshSize).
     * @return            Interpolated value.
     */
    template<auto quantity, typename GridLayout, typename Field>
    double operator()(GridLayout const& layout, Field const& field,
                      Point<double, dim> const& physPoint)
    {
        auto const& dx = layout.meshSize();

        // Convert physical point to cell index and fractional delta.
        // Cell i spans [i*dx, (i+1)*dx], so iCell = floor(x/dx), delta = x/dx - iCell.
        Point<int, dim> iCell;
        std::array<double, dim> delta;
        for (auto d = 0u; d < dim; ++d)
        {
            double const normalizedPos = physPoint[d] / dx[d];
            iCell[d]                   = static_cast<int>(std::floor(normalizedPos));
            delta[d]                   = normalizedPos - iCell[d];
        }

        // Compute start indices and weights for both centerings.
        // MeshToParticle will select the appropriate one per field component dimension.
        this->template indexAndWeights_<QtyCentering, QtyCentering::dual>(layout, iCell, delta);
        this->template indexAndWeights_<QtyCentering, QtyCentering::primal>(layout, iCell, delta);

        auto const indexWeights = std::forward_as_tuple(this->dual_startIndex_, this->dual_weights_,
                                                        this->primal_startIndex_,
                                                        this->primal_weights_);

        return this->meshToParticle_.template operator()<GridLayout, quantity>(field, indexWeights);
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_INTERPOLATOR_FIELD_AT_POINT_HPP
