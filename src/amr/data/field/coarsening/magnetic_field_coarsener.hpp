#ifndef PHARE_MAGNETIC_FIELD_COARSENER_HPP
#define PHARE_MAGNETIC_FIELD_COARSENER_HPP

#include "core/def/phare_mpi.hpp"

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/constants.hpp"
#include "core/hybrid/hybrid_quantities.hpp"
#include "amr/resources_manager/amr_utils.hpp"

#include <SAMRAI/hier/Box.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace PHARE::amr
{
using core::dirX;
using core::dirY;
using core::dirZ;

/** @brief Coarsen magnetic field components by averaging fine face-centered values onto coarse
 * faces.
 */
template<std::size_t dimension>
class MagneticFieldCoarsener
{
public:
    MagneticFieldCoarsener(std::array<core::QtyCentering, dimension> const centering,
                           SAMRAI::hier::Box const& sourceBox,
                           SAMRAI::hier::Box const& destinationBox,
                           SAMRAI::hier::IntVector const& /*ratio*/)
        : centering_{centering}
        , sourceBox_{sourceBox}
        , destinationBox_{destinationBox}
    {
    }

    template<typename FieldT>
    void operator()(FieldT const& fineField, FieldT& coarseField,
                    core::Point<int, dimension> coarseIndex)
    {
        TBOX_ASSERT(fineField.physicalQuantity() == coarseField.physicalQuantity());

        core::Point<int, dimension> fineStartIndex;
        for (auto i = std::size_t{0}; i < dimension; ++i)
            fineStartIndex[i] = coarseIndex[i] * 2;

        fineStartIndex = AMRToLocal(fineStartIndex, sourceBox_);
        coarseIndex    = AMRToLocal(coarseIndex, destinationBox_);

        if constexpr (dimension == 1)
        {
            assert(centering_[dirX] == core::QtyCentering::primal
                   && "Bx should be primal in x in 1D");

            coarseField(coarseIndex[dirX]) = fineField(fineStartIndex[dirX]);
        }
        else if constexpr (dimension == 2)
        {
            if (centering_[dirX] == core::QtyCentering::primal)
            {
                assert(centering_[dirY] == core::QtyCentering::dual
                       && "Bx should be primal in x and dual in y");
                coarseField(coarseIndex[dirX], coarseIndex[dirY])
                    = 0.5
                      * (fineField(fineStartIndex[dirX], fineStartIndex[dirY])
                         + fineField(fineStartIndex[dirX], fineStartIndex[dirY] + 1));
            }
            else if (centering_[dirY] == core::QtyCentering::primal)
            {
                assert(centering_[dirX] == core::QtyCentering::dual
                       && "By should be dual in x and primal in y");
                coarseField(coarseIndex[dirX], coarseIndex[dirY])
                    = 0.5
                      * (fineField(fineStartIndex[dirX], fineStartIndex[dirY])
                         + fineField(fineStartIndex[dirX] + 1, fineStartIndex[dirY]));
            }
            else
            {
                assert(centering_[dirX] == core::QtyCentering::dual
                       && centering_[dirY] == core::QtyCentering::dual
                       && "Bz should be dual in x and y");
                coarseField(coarseIndex[dirX], coarseIndex[dirY])
                    = fineField(fineStartIndex[dirX], fineStartIndex[dirY]);
            }
        }
        else if constexpr (dimension == 3)
        {
            if (centering_[dirX] == core::QtyCentering::primal)
            {
                assert(centering_[dirY] == core::QtyCentering::dual
                       && centering_[dirZ] == core::QtyCentering::dual
                       && "Bx should be primal in x and dual in y/z");
                coarseField(coarseIndex[dirX], coarseIndex[dirY], coarseIndex[dirZ])
                    = 0.25
                      * (fineField(fineStartIndex[dirX], fineStartIndex[dirY], fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX], fineStartIndex[dirY] + 1,
                                     fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX], fineStartIndex[dirY],
                                     fineStartIndex[dirZ] + 1)
                         + fineField(fineStartIndex[dirX], fineStartIndex[dirY] + 1,
                                     fineStartIndex[dirZ] + 1));
            }
            else if (centering_[dirY] == core::QtyCentering::primal)
            {
                assert(centering_[dirX] == core::QtyCentering::dual
                       && centering_[dirZ] == core::QtyCentering::dual
                       && "By should be dual in x/z and primal in y");
                coarseField(coarseIndex[dirX], coarseIndex[dirY], coarseIndex[dirZ])
                    = 0.25
                      * (fineField(fineStartIndex[dirX], fineStartIndex[dirY], fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX] + 1, fineStartIndex[dirY],
                                     fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX], fineStartIndex[dirY],
                                     fineStartIndex[dirZ] + 1)
                         + fineField(fineStartIndex[dirX] + 1, fineStartIndex[dirY],
                                     fineStartIndex[dirZ] + 1));
            }
            else
            {
                assert(centering_[dirX] == core::QtyCentering::dual
                       && centering_[dirY] == core::QtyCentering::dual
                       && centering_[dirZ] == core::QtyCentering::primal
                       && "Bz should be dual in x/y and primal in z");
                coarseField(coarseIndex[dirX], coarseIndex[dirY], coarseIndex[dirZ])
                    = 0.25
                      * (fineField(fineStartIndex[dirX], fineStartIndex[dirY], fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX] + 1, fineStartIndex[dirY],
                                     fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX], fineStartIndex[dirY] + 1,
                                     fineStartIndex[dirZ])
                         + fineField(fineStartIndex[dirX] + 1, fineStartIndex[dirY] + 1,
                                     fineStartIndex[dirZ]));
            }
        }
    }

private:
    std::array<core::QtyCentering, dimension> const centering_;
    SAMRAI::hier::Box const sourceBox_;
    SAMRAI::hier::Box const destinationBox_;
};

} // namespace PHARE::amr

#endif
