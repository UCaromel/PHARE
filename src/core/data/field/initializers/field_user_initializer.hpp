#ifndef _PHARE_CORE_DATA_FIELD_INITIAZILIZERS_FIELD_USER_INITIALIZER_HPP_
#define _PHARE_CORE_DATA_FIELD_INITIAZILIZERS_FIELD_USER_INITIALIZER_HPP_

#include "core/utilities/span.hpp"
#include "core/utilities/types.hpp"
#include "initializer/data_provider.hpp"
#include "core/utilities/point/point.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"

#include <array>
#include <tuple>
#include <memory>
#include <cassert>

namespace PHARE::core
{
// What the user function means once it is stored on the grid. A finite-volume scheme holds cell
// averages, so sampling the analytic function at the node leaves an O(h^2) error in the initial
// condition -- harmless at second order, but it caps a fourth-order scheme on its own.
enum class InitRepresentation { PointValue, CellAverage };

class FieldUserFunctionInitializer
{
public:
    template<InitRepresentation Representation = InitRepresentation::PointValue, typename Field,
             typename GridLayout>
    void static initialize(Field& field, GridLayout const& layout,
                           initializer::InitFunction<GridLayout::dimension> const& init)
    {
        if constexpr (Representation == InitRepresentation::CellAverage)
            cellAverages_(field, layout, init);
        else
            pointValues_(field, layout, init);
    }

private:
    template<typename Field, typename GridLayout>
    void static pointValues_(Field& field, GridLayout const& layout,
                             initializer::InitFunction<GridLayout::dimension> const& init)
    {
        auto const indices = layout.indices(layout.AMRGhostBoxFor(field));
        auto const coords  = layout.template indexesToCoordVectors</*WithField=*/true>(
            indices, field, [](auto& gridLayout, auto& field_, auto const&... args) {
                return gridLayout.fieldNodeCoordinates(field_, args...);
            });

        std::shared_ptr<Span<double>> gridPtr // keep grid data alive
            = std::apply([&](auto&... args) { return init(args...); }, coords);
        Span<double>& grid = *gridPtr;

        for (std::size_t cell_idx = 0; cell_idx < indices.size(); cell_idx++)
            std::apply(
                [&](auto&... args) { field(layout.AMRToLocal(Point{args...})) = grid[cell_idx]; },
                indices[cell_idx]);
    }

    // Tensor-product 2-point Gauss-Legendre average over the element the field lives on, exact for
    // cubics and so O(h^4) for everything else. Only dual directions are integrated: a face-centred
    // component gets the average over its face, an edge-centred one the average along its edge.
    // Primal directions have no extent, so both quadrature nodes land on the same coordinate there
    // and the weights still sum to one.
    template<typename Field, typename GridLayout>
    void static cellAverages_(Field& field, GridLayout const& layout,
                              initializer::InitFunction<GridLayout::dimension> const& init)
    {
        auto constexpr dimension = GridLayout::dimension;
        auto constexpr nbrNodes  = std::size_t{1} << dimension;

        static constexpr double glNode = 0.28867513459481287; // 1 / (2 sqrt(3))
        static constexpr double weight = 1.0 / nbrNodes;

        auto const indices = layout.indices(layout.AMRGhostBoxFor(field));

        for (auto const& indexTuple : indices)
            std::apply([&](auto const&... args) { field(layout.AMRToLocal(Point{args...})) = 0.; },
                       indexTuple);

        for (std::size_t node = 0; node < nbrNodes; ++node)
        {
            std::array<double, dimension> offsets;
            for (std::size_t iDim = 0; iDim < dimension; ++iDim)
                offsets[iDim] = ((node >> iDim) & std::size_t{1}) ? glNode : -glNode;

            auto const coords = layout.template indexesToCoordVectors</*WithField=*/true>(
                indices, field, [&offsets](auto& gridLayout, auto& field_, auto const&... args) {
                    auto point            = gridLayout.fieldNodeCoordinates(field_, args...);
                    auto const meshSize   = gridLayout.meshSize();
                    auto const centerings = gridLayout.centering(field_.physicalQuantity());

                    for_N<dimension>([&](auto iDim) {
                        if (centerings[iDim] == QtyCentering::dual)
                            point[iDim] += offsets[iDim] * meshSize[iDim];
                    });
                    return point;
                });

            std::shared_ptr<Span<double>> gridPtr // keep grid data alive
                = std::apply([&](auto&... args) { return init(args...); }, coords);
            Span<double>& grid = *gridPtr;

            for (std::size_t cell_idx = 0; cell_idx < indices.size(); cell_idx++)
                std::apply(
                    [&](auto&... args) {
                        field(layout.AMRToLocal(Point{args...})) += weight * grid[cell_idx];
                    },
                    indices[cell_idx]);
        }
    }
};

} // namespace PHARE::core

#endif
