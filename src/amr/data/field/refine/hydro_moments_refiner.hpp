#ifndef PHARE_HYDRO_MOMENTS_REFINER_HPP
#define PHARE_HYDRO_MOMENTS_REFINER_HPP

#include "core/def/phare_mpi.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "core/utilities/point/point.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"

#include <SAMRAI/hier/Box.h>
#include <cmath>
#include <cstddef>

namespace PHARE::amr
{
// Zero-order refiner for MHD cell-centered (ddd) → Hybrid vertex (ppp) ghost fills.
// For each fine ppp index j, maps to nearest coarse ddd cell via toCoarseIndex(j) = j/2.
// At ratio=2, odd fine vertices coincide with coarse cell centers (exact); even fine vertices
// are equidistant and floor selects the lower cell.
template<std::size_t dimension>
class HydroMomentsRefiner
{
public:
    HydroMomentsRefiner(std::array<core::QtyCentering, dimension> const& /*centering*/,
                        SAMRAI::hier::Box const& destinationGhostBox,
                        SAMRAI::hier::Box const& sourceGhostBox,
                        SAMRAI::hier::IntVector const& /*ratio*/)
        : fineBox_{destinationGhostBox}
        , coarseBox_{sourceGhostBox}
    {
    }

    template<typename SrcFieldT, typename DstFieldT>
    void operator()(SrcFieldT const& coarseField, DstFieldT& fineField,
                    core::Point<int, dimension> fineIndex)
    {
        if constexpr (std::is_same_v<decltype(coarseField.physicalQuantity()),
                                     decltype(fineField.physicalQuantity())>)
            TBOX_ASSERT(coarseField.physicalQuantity() == fineField.physicalQuantity());

        auto const locFineIdx   = AMRToLocal(fineIndex, fineBox_);
        auto const coarseIdx    = toCoarseIndex(fineIndex);
        auto const locCoarseIdx = AMRToLocal(coarseIdx, coarseBox_);

        if constexpr (dimension == 1)
        {
            if (std::isnan(fineField(locFineIdx[core::dirX])))
                fineField(locFineIdx[core::dirX]) = coarseField(locCoarseIdx[core::dirX]);
        }
        else if constexpr (dimension == 2)
        {
            if (std::isnan(fineField(locFineIdx[core::dirX], locFineIdx[core::dirY])))
                fineField(locFineIdx[core::dirX], locFineIdx[core::dirY])
                    = coarseField(locCoarseIdx[core::dirX], locCoarseIdx[core::dirY]);
        }
        else if constexpr (dimension == 3)
        {
            if (std::isnan(fineField(locFineIdx[core::dirX], locFineIdx[core::dirY],
                                     locFineIdx[core::dirZ])))
                fineField(locFineIdx[core::dirX], locFineIdx[core::dirY], locFineIdx[core::dirZ])
                    = coarseField(locCoarseIdx[core::dirX], locCoarseIdx[core::dirY],
                                  locCoarseIdx[core::dirZ]);
        }
    }

private:
    SAMRAI::hier::Box const fineBox_;
    SAMRAI::hier::Box const coarseBox_;
};
} // namespace PHARE::amr

#endif // PHARE_HYDRO_MOMENTS_REFINER_HPP
