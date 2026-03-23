#ifndef _PHARE_CORE_DATA_FIELD_INITIAZILIZERS_FIELD_USER_INITIALIZER_HPP_
#define _PHARE_CORE_DATA_FIELD_INITIAZILIZERS_FIELD_USER_INITIALIZER_HPP_

#include "core/utilities/span.hpp"
#include "initializer/data_provider.hpp"
#include "core/utilities/point/point.hpp"

#include "core/data/grid/gridlayoutdefs.hpp"

#include <cmath>
#include <tuple>
#include <memory>
#include <cassert>

namespace PHARE::core
{
class FieldUserFunctionInitializer
{
public:
    // template<typename Field, typename GridLayout>
    // void static initialize(Field& field, GridLayout const& layout,
    //                        initializer::InitFunction<GridLayout::dimension> const& init)
    // {
    //     auto const indices = layout.indices(layout.AMRGhostBoxFor(field));
    //     auto const coords  = layout.template indexesToCoordVectors</*WithField=*/true>(
    //         indices, field, [](auto& gridLayout, auto& field_, auto const&... args) {
    //             return gridLayout.fieldNodeCoordinates(field_, args...);
    //         });
    //
    //     std::shared_ptr<Span<double>> gridPtr // keep grid data alive
    //         = std::apply([&](auto&... args) { return init(args...); }, coords);
    //     Span<double>& grid = *gridPtr;
    //
    //     for (std::size_t cell_idx = 0; cell_idx < indices.size(); cell_idx++)
    //         std::apply(
    //             [&](auto&... args) { field(layout.AMRToLocal(Point{args...})) = grid[cell_idx];
    //             }, indices[cell_idx]);
    // }

    // 4th order init
    template<typename Field, typename GridLayout>
    void static initialize(Field& field, GridLayout const& layout,
                           initializer::InitFunction<GridLayout::dimension> const& init)
    {
        static constexpr double gl_pt = 1. / (2. * std::sqrt(3.));
        static constexpr double w     = 0.5;

        auto const indices = layout.indices(layout.AMRGhostBoxFor(field));

        auto getCoordsShifted = [&](auto const& idxs, auto const& offsets) {
            return layout.template indexesToCoordVectors</*WithField=*/true>(
                idxs, field, [&offsets](auto& gridLayout, auto& field_, auto const&... args) {
                    auto pt         = gridLayout.fieldNodeCoordinates(field_, args...);
                    auto meshSize   = gridLayout.meshSize();
                    auto centerings = gridLayout.centering(field_);
                    for_N<GridLayout::dimension>([&](auto i) {
                        if (centerings[i] == QtyCentering::dual) // we only want surface/line
                                                                 // integrals for face/edges
                            pt[i] += offsets[i] * meshSize[i];
                    });
                    return pt;
                });
        };

        // Zero out first
        auto const box = layout.AMRGhostBoxFor(field);
        for (auto const& indiceTuple : indices)
            std::apply([&](auto const&... args) { field(layout.AMRToLocal(Point{args...})) = 0.0; },
                       indiceTuple);

        // 2-point GL quadrature in each dimension:
        if constexpr (GridLayout::dimension == 1)
        {
            for (double s : {-gl_pt, +gl_pt})
            {
                std::array<double, 1> offsets{s};
                auto coords  = getCoordsShifted(indices, offsets);
                auto gridPtr = std::apply([&](auto&... args) { return init(args...); }, coords);
                Span<double>& grid = *gridPtr;
                for (std::size_t i = 0; i < indices.size(); i++)
                    std::apply(
                        [&](auto const&... args) {
                            field(layout.AMRToLocal(Point{args...})) += w * grid[i];
                        },
                        indices[i]);
            }
        }
        else if constexpr (GridLayout::dimension == 2)
        {
            for (double sx : {-gl_pt, +gl_pt})
                for (double sy : {-gl_pt, +gl_pt})
                {
                    std::array<double, 2> offsets{sx, sy};
                    auto coords  = getCoordsShifted(indices, offsets);
                    auto gridPtr = std::apply([&](auto&... args) { return init(args...); }, coords);
                    Span<double>& grid = *gridPtr;
                    double ww          = w * w; // 0.25
                    for (std::size_t i = 0; i < indices.size(); i++)
                        std::apply(
                            [&](auto const&... args) {
                                field(layout.AMRToLocal(Point{args...})) += ww * grid[i];
                            },
                            indices[i]);
                }
        }
        else if constexpr (GridLayout::dimension == 3)
        {
            for (double sx : {-gl_pt, +gl_pt})
                for (double sy : {-gl_pt, +gl_pt})
                    for (double sz : {-gl_pt, +gl_pt})
                    {
                        std::array<double, 3> offsets{sx, sy, sz};
                        auto coords = getCoordsShifted(indices, offsets);
                        auto gridPtr
                            = std::apply([&](auto&... args) { return init(args...); }, coords);
                        Span<double>& grid = *gridPtr;
                        double www         = w * w * w; // 0.125
                        for (std::size_t i = 0; i < indices.size(); i++)
                            std::apply(
                                [&](auto const&... args) {
                                    field(layout.AMRToLocal(Point{args...})) += www * grid[i];
                                },
                                indices[i]);
                    }
        }
    }
};

} // namespace PHARE::core

#endif
