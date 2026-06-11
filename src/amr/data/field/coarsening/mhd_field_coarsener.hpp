#ifndef PHARE_MHD_FIELD_COARSENER_HPP
#define PHARE_MHD_FIELD_COARSENER_HPP

#include "core/def/phare_mpi.hpp"

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/constants.hpp"
#include "amr/resources_manager/amr_utils.hpp"

#include <SAMRAI/hier/Box.h>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace PHARE::amr
{
using core::dirX;
using core::dirY;
using core::dirZ;

/** @brief Coarsen MHD hydro quantities by averaging fine cell-centered values onto coarse cells.
 *
 * Intended for cell-centered conservative quantities (rho, rhoV, Etot).
 */
template<std::size_t dimension>
class MHDFieldCoarsener
{
public:
    MHDFieldCoarsener(std::array<core::QtyCentering, dimension> const centering,
                      SAMRAI::hier::Box const& sourceBox,
                      SAMRAI::hier::Box const& destinationBox,
                      SAMRAI::hier::IntVector const& ratio)
        : centering_{centering}
        , sourceBox_{sourceBox}
        , destinationBox_{destinationBox}
        , ratio_{ratio}
    {
    }

    template<typename FieldT>
    void operator()(FieldT const& fineField, FieldT& coarseField,
                    core::Point<int, dimension> coarseIndex)
    {
        TBOX_ASSERT(fineField.physicalQuantity() == coarseField.physicalQuantity());

        core::Point<int, dimension> fineStartIndex;
        for (auto i = std::size_t{0}; i < dimension; ++i)
            fineStartIndex[i] = coarseIndex[i] * ratio_(static_cast<int>(i));

        fineStartIndex = AMRToLocal(fineStartIndex, sourceBox_);
        coarseIndex    = AMRToLocal(coarseIndex, destinationBox_);

        if constexpr (dimension == 1)
        {
            assert(centering_[dirX] == core::QtyCentering::dual
                   && "MHD hydro should be cell-centered in 1D");
            double coarseValue = 0.0;
            for (int iShiftX = 0; iShiftX < ratio_(dirX); ++iShiftX)
                coarseValue += fineField(fineStartIndex[dirX] + iShiftX);
            coarseField(coarseIndex[dirX]) = coarseValue / static_cast<double>(ratio_(dirX));
        }
        else if constexpr (dimension == 2)
        {
            assert(centering_[dirX] == core::QtyCentering::dual
                   && centering_[dirY] == core::QtyCentering::dual
                   && "MHD hydro should be cell-centered in 2D");
            double coarseValue = 0.0;
            for (int iShiftX = 0; iShiftX < ratio_(dirX); ++iShiftX)
                for (int iShiftY = 0; iShiftY < ratio_(dirY); ++iShiftY)
                    coarseValue += fineField(fineStartIndex[dirX] + iShiftX,
                                             fineStartIndex[dirY] + iShiftY);
            coarseField(coarseIndex[dirX], coarseIndex[dirY])
                = coarseValue
                  / static_cast<double>(ratio_(dirX) * ratio_(dirY));
        }
        else if constexpr (dimension == 3)
        {
            assert(centering_[dirX] == core::QtyCentering::dual
                   && centering_[dirY] == core::QtyCentering::dual
                   && centering_[dirZ] == core::QtyCentering::dual
                   && "MHD hydro should be cell-centered in 3D");
            double coarseValue = 0.0;
            for (int iShiftX = 0; iShiftX < ratio_(dirX); ++iShiftX)
                for (int iShiftY = 0; iShiftY < ratio_(dirY); ++iShiftY)
                    for (int iShiftZ = 0; iShiftZ < ratio_(dirZ); ++iShiftZ)
                        coarseValue += fineField(fineStartIndex[dirX] + iShiftX,
                                                 fineStartIndex[dirY] + iShiftY,
                                                 fineStartIndex[dirZ] + iShiftZ);
            coarseField(coarseIndex[dirX], coarseIndex[dirY], coarseIndex[dirZ])
                = coarseValue
                  / static_cast<double>(ratio_(dirX) * ratio_(dirY) * ratio_(dirZ));
        }
    }

private:
    std::array<core::QtyCentering, dimension> const centering_;
    SAMRAI::hier::Box const sourceBox_;
    SAMRAI::hier::Box const destinationBox_;
    SAMRAI::hier::IntVector const ratio_;
};

} // namespace PHARE::amr

#endif
