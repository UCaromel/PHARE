#ifndef PHARE_CORE_GRID_GRIDLAYOUTDEFS_HPP
#define PHARE_CORE_GRID_GRIDLAYOUTDEFS_HPP

#include "core/def.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/point/point.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace PHARE
{
namespace core
{
    enum class Direction { X, Y, Z };

    enum class Side { Lower = -1, Upper = 1 };

    enum class QtyCentering : std::uint16_t { primal = 0, dual = 1 };


    template<std::size_t dim>
    struct WeightPoint
    {
        Point<int, dim> indexes{};
        double coef{0.0};
    };

    // using LinearCombination = std::vector<WeightPoint>;

    enum class Layout { Yee };


    template<typename T, std::size_t dim>
    NO_DISCARD Box<T, dim> toCenteredBox(Box<T, dim> box,
                                         std::array<QtyCentering, dim> const& centerings)
    {
        for (std::size_t i = 0; i < dim; ++i)
            if (centerings[i] == QtyCentering::primal)
                ++box.upper[i];

        return box;
    }


    // Fourth-order prolongation of cell averages by 2: the σ child's average of the quartic
    // through ū_{I-2..I+2} is ū_I + σ·(22(ū_{I+1} - ū_{I-1}) - 3(ū_{I+2} - ū_{I-2}))/128.
    inline constexpr std::array<double, 2> dualProlongation4Slopes{22. / 128., -3. / 128.};
} // namespace core
} // namespace PHARE

#endif // GRIDLAYOUTDEFS_HPP
