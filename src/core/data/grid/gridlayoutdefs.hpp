#ifndef PHARE_CORE_GRID_GRIDLAYOUTDEFS_HPP
#define PHARE_CORE_GRID_GRIDLAYOUTDEFS_HPP


#include "core/utilities/point/point.hpp"
#include "core/physical_quantities.hpp"

#include <cstddef>

namespace PHARE
{
namespace core
{
    enum class Direction { X, Y, Z };


    enum class QtyCentering : std::uint16_t { primal = 0, dual = 1 };


    template<std::size_t dim>
    struct WeightPoint
    {
        constexpr WeightPoint(Point<int, dim> point, double _coef)
            : indexes{std::move(point)}
            , coef{_coef}
        {
        }

        Point<int, dim> indexes;
        double coef;
    };

    // using LinearCombination = std::vector<WeightPoint>;

    enum class Layout { Yee };

    /**
     * @brief gridDataT provides constants used to initialize:
     * - physicalQuantity centerings
     * - physical start/end indexes
     * - ghost start/end indexes
     * - numbers of padding cells and physical cells
     */
    struct gridDataT
    {
        static constexpr Direction dirX = Direction::X;
        static constexpr Direction dirY = Direction::Y;
        static constexpr Direction dirZ = Direction::Z;

        static constexpr QtyCentering primal = QtyCentering::primal;
        static constexpr QtyCentering dual   = QtyCentering::dual;

        static constexpr std::uint32_t idirX = static_cast<std::uint32_t>(Direction::X);
        static constexpr std::uint32_t idirY = static_cast<std::uint32_t>(Direction::Y);
        static constexpr std::uint32_t idirZ = static_cast<std::uint32_t>(Direction::Z);

        // Shared B/E/J (unchanged names)
        static constexpr std::uint32_t iBx = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Bx);
        static constexpr std::uint32_t iBy = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::By);
        static constexpr std::uint32_t iBz = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Bz);

        static constexpr std::uint32_t iEx = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Ex);
        static constexpr std::uint32_t iEy = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Ey);
        static constexpr std::uint32_t iEz = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Ez);

        static constexpr std::uint32_t iJx = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Jx);
        static constexpr std::uint32_t iJy = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Jy);
        static constexpr std::uint32_t iJz = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Jz);

        // Hybrid-specific (ppp)
        static constexpr std::uint32_t iHyb_rho
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_rho);

        static constexpr std::uint32_t iHyb_Vx = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Vx);
        static constexpr std::uint32_t iHyb_Vy = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Vy);
        static constexpr std::uint32_t iHyb_Vz = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Vz);

        static constexpr std::uint32_t iHyb_P = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_P);

        static constexpr std::uint32_t iHyb_Mxx
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Mxx);
        static constexpr std::uint32_t iHyb_Mxy
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Mxy);
        static constexpr std::uint32_t iHyb_Mxz
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Mxz);
        static constexpr std::uint32_t iHyb_Myy
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Myy);
        static constexpr std::uint32_t iHyb_Myz
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Myz);
        static constexpr std::uint32_t iHyb_Mzz
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::Hyb_Mzz);

        // MHD-specific (ddd)
        static constexpr std::uint32_t iMHD_rho
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_rho);

        static constexpr std::uint32_t iMHD_Vx = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_Vx);
        static constexpr std::uint32_t iMHD_Vy = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_Vy);
        static constexpr std::uint32_t iMHD_Vz = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_Vz);

        static constexpr std::uint32_t iMHD_P = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_P);

        static constexpr std::uint32_t iMHD_rhoVx
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_rhoVx);
        static constexpr std::uint32_t iMHD_rhoVy
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_rhoVy);
        static constexpr std::uint32_t iMHD_rhoVz
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_rhoVz);

        static constexpr std::uint32_t iMHD_Etot
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::MHD_Etot);

        // Face-centered flux quantities (shared)
        static constexpr std::uint32_t iScalarFlux_x
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::ScalarFlux_x);
        static constexpr std::uint32_t iScalarFlux_y
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::ScalarFlux_y);
        static constexpr std::uint32_t iScalarFlux_z
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::ScalarFlux_z);

        static constexpr std::uint32_t iVecFluxX_x
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxX_x);
        static constexpr std::uint32_t iVecFluxY_x
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxY_x);
        static constexpr std::uint32_t iVecFluxZ_x
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxZ_x);

        static constexpr std::uint32_t iVecFluxX_y
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxX_y);
        static constexpr std::uint32_t iVecFluxY_y
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxY_y);
        static constexpr std::uint32_t iVecFluxZ_y
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxZ_y);

        static constexpr std::uint32_t iVecFluxX_z
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxX_z);
        static constexpr std::uint32_t iVecFluxY_z
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxY_z);
        static constexpr std::uint32_t iVecFluxZ_z
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecFluxZ_z);

        // All-primal (ppp, MHD synthetic)
        static constexpr std::uint32_t iScalarAllPrimal
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::ScalarAllPrimal);

        static constexpr std::uint32_t iVecAllPrimalX
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecAllPrimalX);
        static constexpr std::uint32_t iVecAllPrimalY
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecAllPrimalY);
        static constexpr std::uint32_t iVecAllPrimalZ
            = static_cast<std::uint32_t>(PhysicalQuantity::Scalar::VecAllPrimalZ);
    };

} // namespace core
} // namespace PHARE

#endif // GRIDLAYOUTDEFS_HPP
