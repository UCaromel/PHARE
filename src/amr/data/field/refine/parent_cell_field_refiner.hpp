#ifndef PHARE_PARENT_CELL_FIELD_REFINER_HPP
#define PHARE_PARENT_CELL_FIELD_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/box/box_spans.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "coarse_cell_round_out.hpp"
#include "field_refiner_kernel.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>


namespace PHARE::amr
{
/**
 * @brief Refinement of a cell-centred flag: every fine cell takes the value of its coarse cell.
 *
 * A flag is not interpolated: any stencil would make values that are neither set nor unset, and
 * spread the flag over the neighbours of the cells that carry it.
 */
template<typename GridLayoutT, typename FieldT>
class ParentCellFieldRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

public:
    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& /*ratio*/) const override
    {
        for (auto const c : centering)
            if (c != core::QtyCentering::dual)
                throw std::runtime_error("ParentCellFieldRefiner refines cell-centred fields only");

        core::for_each_point(phare_box_from<dimension>(intersectionBox), [&](auto const& fine) {
            auto const parent = AMRToLocal(toCoarseIndex<dimension>(fine), sourceFieldBox);

            // as the other kernels, never overwrite a value a same-level copy already set
            if (auto& flag = destinationField(AMRToLocal(fine, destFieldBox)); std::isnan(flag))
                flag = sourceField(parent);
        });
    }

    int coarseStencilWidth() const override { return 0; }
};

} // namespace PHARE::amr


#endif
