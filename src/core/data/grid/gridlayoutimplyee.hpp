#ifndef PHARE_CORE_GRID_GRIDLAYOUTYEE_HPP
#define PHARE_CORE_GRID_GRIDLAYOUTYEE_HPP

#include <array>
#include <vector>

#include "core/def.hpp"
#include "core/physical_quantities.hpp"
#include "core/utilities/ghost_width_calculator.hpp"
#include "core/utilities/constants.hpp"
#include "core/utilities/types.hpp"
#include "gridlayoutdefs.hpp"

namespace PHARE
{
namespace core
{
    /**
     * @brief GridLayoutNdArrayImplYee class is a concrete GridLayoutNdArrayImpl used a Yee
     * type grid layout is needed.
     *
     * It provides methods related to grid layout operations:
     * - physical domain start/end indexes
     * - indexes of the first and last ghost nodes
     * - allocation sizes for Field attributes of other classes
     * - partial derivative operator (Faraday)
     * - physical coordinate given a field and a primal point (ix, iy, iz)
     * - cell centered coordinate given a primal point (ix, iy, iz)
     */
    template<std::size_t dim, std::size_t interpOrder, std::uint32_t reconstruction_nghosts_ = 0>
    class GridLayoutImplYee
    {
        // ------------------------------------------------------------------------
        //                              PRIVATE
        // ------------------------------------------------------------------------
    public:
        static constexpr std::size_t dimension    = dim;
        static constexpr std::size_t interp_order = interpOrder;
        static constexpr std::string_view type    = "yee";
        using quantity_type                       = PhysicalQuantity;
        static constexpr std::uint32_t reconstruction_nghosts = reconstruction_nghosts_;
        static constexpr std::uint32_t ghost_width =
            (reconstruction_nghosts_ > 0)
                ? nbrGhostsFromReconstruction<reconstruction_nghosts_>()
                : nbrGhostsFromInterpOrder<interpOrder>();
        /*
    void constexpr initLinearCombinations_();

    LinearCombination momentsToEx_;
    LinearCombination momentsToEy_;
    LinearCombination momentsToEz_;
    LinearCombination BxToEy_;
    LinearCombination BxToEz_;
    LinearCombination ByToEx_;
    LinearCombination ByToEz_;
    LinearCombination BzToEx_;
    LinearCombination BzToEy_;
    LinearCombination ExToMoment_;
    LinearCombination EyToMoment_;
    LinearCombination EzToMoment_;
    */

        /**
         * @brief GridLayoutImpl<Selector<Layout,Layout::Yee>,dim>::initLayoutCentering_ initialize
         * the table _QuantityCentering_. This is THE important array in the GridLayout module.
         * This table knows which quantity is primal/dual along each direction. It is **this** array
         * that
         * **defines** what a Yee Layout is. Once this array is defined, the rest of the GridLayout
         * needs this array OK and can go on from here... hence all other functions in the Yee
         * interface are just calling private implementation common to all layouts
         */
        constexpr auto static initLayoutCentering_()
        {
            gridDataT const data{};

            // shared Yee-staggered (B/E/J)
            std::array<QtyCentering, NBR_COMPO> const Bx = {{data.primal, data.dual, data.dual}};
            std::array<QtyCentering, NBR_COMPO> const By = {{data.dual, data.primal, data.dual}};
            std::array<QtyCentering, NBR_COMPO> const Bz = {{data.dual, data.dual, data.primal}};

            std::array<QtyCentering, NBR_COMPO> const Ex = {{data.dual, data.primal, data.primal}};
            std::array<QtyCentering, NBR_COMPO> const Ey = {{data.primal, data.dual, data.primal}};
            std::array<QtyCentering, NBR_COMPO> const Ez = {{data.primal, data.primal, data.dual}};

            std::array<QtyCentering, NBR_COMPO> const Jx = {{data.dual, data.primal, data.primal}};
            std::array<QtyCentering, NBR_COMPO> const Jy = {{data.primal, data.dual, data.primal}};
            std::array<QtyCentering, NBR_COMPO> const Jz = {{data.primal, data.primal, data.dual}};

            // Hybrid ppp quantities
            std::array<QtyCentering, NBR_COMPO> const ppp = {{data.primal, data.primal, data.primal}};

            // MHD ddd (cell-centered) quantities
            std::array<QtyCentering, NBR_COMPO> const ddd = {{data.dual, data.dual, data.dual}};

            // face-centered flux quantities
            std::array<QtyCentering, NBR_COMPO> const ScalarFlux_x = {{data.primal, data.dual, data.dual}};
            std::array<QtyCentering, NBR_COMPO> const ScalarFlux_y = {{data.dual, data.primal, data.dual}};
            std::array<QtyCentering, NBR_COMPO> const ScalarFlux_z = {{data.dual, data.dual, data.primal}};

            // Centering array indexed by PhysicalQuantity::Scalar enum value (count=45).
            // Order must match the enum exactly.
            std::array<std::array<QtyCentering, NBR_COMPO>,
                       static_cast<std::size_t>(PhysicalQuantity::Scalar::count)> const _QtyCentering{
                // idx 0-8: Bx By Bz Ex Ey Ez Jx Jy Jz (shared)
                Bx,  By,  Bz,  Ex,  Ey,  Ez,  Jx,  Jy,  Jz,
                // idx 9-19: Hybrid ppp quantities
                ppp, ppp, ppp, ppp, ppp,              // Hyb_rho, Hyb_Vx, Hyb_Vy, Hyb_Vz, Hyb_P
                ppp, ppp, ppp, ppp, ppp, ppp,          // Hyb_Mxx..Hyb_Mzz
                // idx 20-28: MHD ddd quantities
                ddd, ddd, ddd, ddd, ddd, ddd, ddd, ddd, ddd, // MHD_rho..MHD_Etot
                // idx 29-31: ScalarFlux_x/y/z (face-centered)
                ScalarFlux_x, ScalarFlux_y, ScalarFlux_z,
                // idx 32-34: VecFlux*_x (x-face = pdd)
                Bx, Bx, Bx,
                // idx 35-37: VecFlux*_y (y-face = dpd)
                By, By, By,
                // idx 38-40: VecFlux*_z (z-face = ddp)
                Bz, Bz, Bz,
                // idx 41-44: ScalarAllPrimal, VecAllPrimalX/Y/Z (ppp)
                ppp, ppp, ppp, ppp};

            return _QtyCentering;
        }

        //! says for each PhysicalQuantity::Scalar whether it is primal or dual, in each direction
        constexpr static std::array<std::array<QtyCentering, NBR_COMPO>,
                                    static_cast<std::size_t>(PhysicalQuantity::Scalar::count)> const
            _QtyCentering_{initLayoutCentering_()};

        static std::size_t const dim_{dim};

        // ------------------------------------------------------------------------
        //                          PUBLIC INTERFACE
        // ------------------------------------------------------------------------
    public:
        NO_DISCARD constexpr static std::array<QtyCentering, dim>
        centering(PhysicalQuantity::Scalar _Quantity)
        {
            constexpr gridDataT gridData_{};
            if constexpr (dim == 1)
            {
                switch (_Quantity)
                {
                    case PhysicalQuantity::Scalar::Bx:
                        return {{_QtyCentering_[gridData_.iBx][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::By:
                        return {{_QtyCentering_[gridData_.iBy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Bz:
                        return {{_QtyCentering_[gridData_.iBz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Ex:
                        return {{_QtyCentering_[gridData_.iEx][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Ey:
                        return {{_QtyCentering_[gridData_.iEy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Ez:
                        return {{_QtyCentering_[gridData_.iEz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Jx:
                        return {{_QtyCentering_[gridData_.iJx][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Jy:
                        return {{_QtyCentering_[gridData_.iJy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Jz:
                        return {{_QtyCentering_[gridData_.iJz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_rho:
                        return {{_QtyCentering_[gridData_.iHyb_rho][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Vx:
                        return {{_QtyCentering_[gridData_.iHyb_Vx][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Vy:
                        return {{_QtyCentering_[gridData_.iHyb_Vy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Vz:
                        return {{_QtyCentering_[gridData_.iHyb_Vz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_P:
                        return {{_QtyCentering_[gridData_.iHyb_P][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxx:
                        return {{_QtyCentering_[gridData_.iHyb_Mxx][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxy:
                        return {{_QtyCentering_[gridData_.iHyb_Mxy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxz:
                        return {{_QtyCentering_[gridData_.iHyb_Mxz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Myy:
                        return {{_QtyCentering_[gridData_.iHyb_Myy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Myz:
                        return {{_QtyCentering_[gridData_.iHyb_Myz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::Hyb_Mzz:
                        return {{_QtyCentering_[gridData_.iHyb_Mzz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::MHD_rho:
                    case PhysicalQuantity::Scalar::MHD_Vx:
                    case PhysicalQuantity::Scalar::MHD_Vy:
                    case PhysicalQuantity::Scalar::MHD_Vz:
                    case PhysicalQuantity::Scalar::MHD_P:
                    case PhysicalQuantity::Scalar::MHD_rhoVx:
                    case PhysicalQuantity::Scalar::MHD_rhoVy:
                    case PhysicalQuantity::Scalar::MHD_rhoVz:
                    case PhysicalQuantity::Scalar::MHD_Etot:
                        return {{_QtyCentering_[gridData_.iMHD_rho][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_x:
                    case PhysicalQuantity::Scalar::VecFluxX_x:
                    case PhysicalQuantity::Scalar::VecFluxY_x:
                    case PhysicalQuantity::Scalar::VecFluxZ_x:
                        return {{_QtyCentering_[gridData_.iBx][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_y:
                    case PhysicalQuantity::Scalar::VecFluxX_y:
                    case PhysicalQuantity::Scalar::VecFluxY_y:
                    case PhysicalQuantity::Scalar::VecFluxZ_y:
                        return {{_QtyCentering_[gridData_.iBy][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_z:
                    case PhysicalQuantity::Scalar::VecFluxX_z:
                    case PhysicalQuantity::Scalar::VecFluxY_z:
                    case PhysicalQuantity::Scalar::VecFluxZ_z:
                        return {{_QtyCentering_[gridData_.iBz][gridData_.idirX]}};
                    case PhysicalQuantity::Scalar::ScalarAllPrimal:
                    case PhysicalQuantity::Scalar::VecAllPrimalX:
                    case PhysicalQuantity::Scalar::VecAllPrimalY:
                    case PhysicalQuantity::Scalar::VecAllPrimalZ:
                        return {{_QtyCentering_[gridData_.iScalarAllPrimal][gridData_.idirX]}};
                    default: throw std::runtime_error("Wrong _Quantity");
                }
            }

            else if constexpr (dim == 2)
            {
                switch (_Quantity)
                {
                    case PhysicalQuantity::Scalar::Bx:
                        return {{_QtyCentering_[gridData_.iBx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::By:
                        return {{_QtyCentering_[gridData_.iBy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Bz:
                        return {{_QtyCentering_[gridData_.iBz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Ex:
                        return {{_QtyCentering_[gridData_.iEx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEx][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Ey:
                        return {{_QtyCentering_[gridData_.iEy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Ez:
                        return {{_QtyCentering_[gridData_.iEz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Jx:
                        return {{_QtyCentering_[gridData_.iJx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJx][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Jy:
                        return {{_QtyCentering_[gridData_.iJy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Jz:
                        return {{_QtyCentering_[gridData_.iJz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_rho:
                        return {{_QtyCentering_[gridData_.iHyb_rho][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_rho][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Vx:
                        return {{_QtyCentering_[gridData_.iHyb_Vx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Vx][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Vy:
                        return {{_QtyCentering_[gridData_.iHyb_Vy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Vy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Vz:
                        return {{_QtyCentering_[gridData_.iHyb_Vz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Vz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_P:
                        return {{_QtyCentering_[gridData_.iHyb_P][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_P][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxx:
                        return {{_QtyCentering_[gridData_.iHyb_Mxx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mxx][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxy:
                        return {{_QtyCentering_[gridData_.iHyb_Mxy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mxy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxz:
                        return {{_QtyCentering_[gridData_.iHyb_Mxz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mxz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Myy:
                        return {{_QtyCentering_[gridData_.iHyb_Myy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Myy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Myz:
                        return {{_QtyCentering_[gridData_.iHyb_Myz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Myz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::Hyb_Mzz:
                        return {{_QtyCentering_[gridData_.iHyb_Mzz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mzz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::MHD_rho:
                    case PhysicalQuantity::Scalar::MHD_Vx:
                    case PhysicalQuantity::Scalar::MHD_Vy:
                    case PhysicalQuantity::Scalar::MHD_Vz:
                    case PhysicalQuantity::Scalar::MHD_P:
                    case PhysicalQuantity::Scalar::MHD_rhoVx:
                    case PhysicalQuantity::Scalar::MHD_rhoVy:
                    case PhysicalQuantity::Scalar::MHD_rhoVz:
                    case PhysicalQuantity::Scalar::MHD_Etot:
                        return {{_QtyCentering_[gridData_.iMHD_rho][gridData_.idirX],
                                 _QtyCentering_[gridData_.iMHD_rho][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_x:
                    case PhysicalQuantity::Scalar::VecFluxX_x:
                    case PhysicalQuantity::Scalar::VecFluxY_x:
                    case PhysicalQuantity::Scalar::VecFluxZ_x:
                        return {{_QtyCentering_[gridData_.iBx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_y:
                    case PhysicalQuantity::Scalar::VecFluxX_y:
                    case PhysicalQuantity::Scalar::VecFluxY_y:
                    case PhysicalQuantity::Scalar::VecFluxZ_y:
                        return {{_QtyCentering_[gridData_.iBy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_z:
                    case PhysicalQuantity::Scalar::VecFluxX_z:
                    case PhysicalQuantity::Scalar::VecFluxY_z:
                    case PhysicalQuantity::Scalar::VecFluxZ_z:
                        return {{_QtyCentering_[gridData_.iBz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz][gridData_.idirY]}};
                    case PhysicalQuantity::Scalar::ScalarAllPrimal:
                    case PhysicalQuantity::Scalar::VecAllPrimalX:
                    case PhysicalQuantity::Scalar::VecAllPrimalY:
                    case PhysicalQuantity::Scalar::VecAllPrimalZ:
                        return {{_QtyCentering_[gridData_.iScalarAllPrimal][gridData_.idirX],
                                 _QtyCentering_[gridData_.iScalarAllPrimal][gridData_.idirY]}};
                    default: throw std::runtime_error("Wrong _Quantity");
                }
            }

            else if constexpr (dim == 3)
            {
                switch (_Quantity)
                {
                    case PhysicalQuantity::Scalar::Bx:
                        return {{_QtyCentering_[gridData_.iBx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::By:
                        return {{_QtyCentering_[gridData_.iBy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Bz:
                        return {{_QtyCentering_[gridData_.iBz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Ex:
                        return {{_QtyCentering_[gridData_.iEx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iEx][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Ey:
                        return {{_QtyCentering_[gridData_.iEy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iEy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Ez:
                        return {{_QtyCentering_[gridData_.iEz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iEz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Jx:
                        return {{_QtyCentering_[gridData_.iJx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iJx][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Jy:
                        return {{_QtyCentering_[gridData_.iJy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iJy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Jz:
                        return {{_QtyCentering_[gridData_.iJz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iJz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_rho:
                        return {{_QtyCentering_[gridData_.iHyb_rho][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_rho][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_rho][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Vx:
                        return {{_QtyCentering_[gridData_.iHyb_Vx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Vx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Vx][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Vy:
                        return {{_QtyCentering_[gridData_.iHyb_Vy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Vy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Vy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Vz:
                        return {{_QtyCentering_[gridData_.iHyb_Vz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Vz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Vz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_P:
                        return {{_QtyCentering_[gridData_.iHyb_P][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_P][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_P][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxx:
                        return {{_QtyCentering_[gridData_.iHyb_Mxx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mxx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Mxx][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxy:
                        return {{_QtyCentering_[gridData_.iHyb_Mxy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mxy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Mxy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Mxz:
                        return {{_QtyCentering_[gridData_.iHyb_Mxz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mxz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Mxz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Myy:
                        return {{_QtyCentering_[gridData_.iHyb_Myy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Myy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Myy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Myz:
                        return {{_QtyCentering_[gridData_.iHyb_Myz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Myz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Myz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::Hyb_Mzz:
                        return {{_QtyCentering_[gridData_.iHyb_Mzz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iHyb_Mzz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iHyb_Mzz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::MHD_rho:
                    case PhysicalQuantity::Scalar::MHD_Vx:
                    case PhysicalQuantity::Scalar::MHD_Vy:
                    case PhysicalQuantity::Scalar::MHD_Vz:
                    case PhysicalQuantity::Scalar::MHD_P:
                    case PhysicalQuantity::Scalar::MHD_rhoVx:
                    case PhysicalQuantity::Scalar::MHD_rhoVy:
                    case PhysicalQuantity::Scalar::MHD_rhoVz:
                    case PhysicalQuantity::Scalar::MHD_Etot:
                        return {{_QtyCentering_[gridData_.iMHD_rho][gridData_.idirX],
                                 _QtyCentering_[gridData_.iMHD_rho][gridData_.idirY],
                                 _QtyCentering_[gridData_.iMHD_rho][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_x:
                    case PhysicalQuantity::Scalar::VecFluxX_x:
                    case PhysicalQuantity::Scalar::VecFluxY_x:
                    case PhysicalQuantity::Scalar::VecFluxZ_x:
                        return {{_QtyCentering_[gridData_.iBx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_y:
                    case PhysicalQuantity::Scalar::VecFluxX_y:
                    case PhysicalQuantity::Scalar::VecFluxY_y:
                    case PhysicalQuantity::Scalar::VecFluxZ_y:
                        return {{_QtyCentering_[gridData_.iBy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::ScalarFlux_z:
                    case PhysicalQuantity::Scalar::VecFluxX_z:
                    case PhysicalQuantity::Scalar::VecFluxY_z:
                    case PhysicalQuantity::Scalar::VecFluxZ_z:
                        return {{_QtyCentering_[gridData_.iBz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz][gridData_.idirZ]}};
                    case PhysicalQuantity::Scalar::ScalarAllPrimal:
                    case PhysicalQuantity::Scalar::VecAllPrimalX:
                    case PhysicalQuantity::Scalar::VecAllPrimalY:
                    case PhysicalQuantity::Scalar::VecAllPrimalZ:
                        return {{_QtyCentering_[gridData_.iScalarAllPrimal][gridData_.idirX],
                                 _QtyCentering_[gridData_.iScalarAllPrimal][gridData_.idirY],
                                 _QtyCentering_[gridData_.iScalarAllPrimal][gridData_.idirZ]}};
                    default: throw std::runtime_error("Wrong _Quantity");
                }
            }
        }



        NO_DISCARD constexpr static std::array<std::array<QtyCentering, dim>, 3>
        centering(PhysicalQuantity::Vector _Quantity)
        {
            switch (_Quantity)
            {
                case PhysicalQuantity::Vector::B:
                    return {{centering(PhysicalQuantity::Scalar::Bx),
                             centering(PhysicalQuantity::Scalar::By),
                             centering(PhysicalQuantity::Scalar::Bz)}};

                case PhysicalQuantity::Vector::Hyb_V:
                    return {{centering(PhysicalQuantity::Scalar::Hyb_Vx),
                             centering(PhysicalQuantity::Scalar::Hyb_Vy),
                             centering(PhysicalQuantity::Scalar::Hyb_Vz)}};

                case PhysicalQuantity::Vector::J:
                    return {{centering(PhysicalQuantity::Scalar::Jx),
                             centering(PhysicalQuantity::Scalar::Jy),
                             centering(PhysicalQuantity::Scalar::Jz)}};

                case PhysicalQuantity::Vector::E:
                    return {{centering(PhysicalQuantity::Scalar::Ex),
                             centering(PhysicalQuantity::Scalar::Ey),
                             centering(PhysicalQuantity::Scalar::Ez)}};

                case PhysicalQuantity::Vector::MHD_V:
                    return {{centering(PhysicalQuantity::Scalar::MHD_Vx),
                             centering(PhysicalQuantity::Scalar::MHD_Vy),
                             centering(PhysicalQuantity::Scalar::MHD_Vz)}};

                case PhysicalQuantity::Vector::MHD_rhoV:
                    return {{centering(PhysicalQuantity::Scalar::MHD_rhoVx),
                             centering(PhysicalQuantity::Scalar::MHD_rhoVy),
                             centering(PhysicalQuantity::Scalar::MHD_rhoVz)}};

                case PhysicalQuantity::Vector::VecFlux_x:
                    return {{centering(PhysicalQuantity::Scalar::VecFluxX_x),
                             centering(PhysicalQuantity::Scalar::VecFluxY_x),
                             centering(PhysicalQuantity::Scalar::VecFluxZ_x)}};

                case PhysicalQuantity::Vector::VecFlux_y:
                    return {{centering(PhysicalQuantity::Scalar::VecFluxX_y),
                             centering(PhysicalQuantity::Scalar::VecFluxY_y),
                             centering(PhysicalQuantity::Scalar::VecFluxZ_y)}};

                case PhysicalQuantity::Vector::VecFlux_z:
                    return {{centering(PhysicalQuantity::Scalar::VecFluxX_z),
                             centering(PhysicalQuantity::Scalar::VecFluxY_z),
                             centering(PhysicalQuantity::Scalar::VecFluxZ_z)}};

                case PhysicalQuantity::Vector::VecAllPrimal:
                    return {{centering(PhysicalQuantity::Scalar::VecAllPrimalX),
                             centering(PhysicalQuantity::Scalar::VecAllPrimalY),
                             centering(PhysicalQuantity::Scalar::VecAllPrimalZ)}};

                default: throw std::runtime_error("Wrong _Quantity");
            }
        }

        NO_DISCARD auto static constexpr dualToPrimal()
        {
            /*
             * the following is only valid when dual and primal do not have the same number of
            ghosts
             * and that depends on the interp order
             * It is commented out because ghosts are hard coded to 5 for now.
             *
            if constexpr (interp_order == 1 || interp_order == 2 || interp_order == 4)
                return -1;
            else if constexpr (interp_order == 3)
                return 1;
               */
            return -1;
        }

        NO_DISCARD auto static constexpr primalToDual()
        {
            return 1;
            /*
             * the following is only valid when dual and primal do not have the same number of
            ghosts
             * and that depends on the interp order
             * It is commented out because ghosts are hard coded to 5 for now.
             *
            if constexpr (interp_order == 1 || interp_order == 2 || interp_order == 4)
                return 1;
            else if constexpr (interp_order == 3)
                return -1;
                */
        }

        NO_DISCARD auto static constexpr momentsToEx()
        {
            // Ex is dual primal primal
            // moments are primal primal primal
            // operation is thus Ppp to Dpp
            // shift only in the X direction
            auto constexpr iShift = primalToDual();

            // P1 is always on top of Ex so no shift

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr momentsToEy()
        {
            // Ey is primal dual primal
            // moments are primal primal primal
            // operation is thus pPp to pDp
            // shift only in the Y direction
            [[maybe_unused]] auto constexpr iShift = primalToDual();

            if constexpr (dimension == 1)
            {
                // since the linear combination is in the Y direction
                // in 1D the moment is already on Ey so return 1 point with no shift
                // with coef 1.
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr momentsToEz()
        {
            // Ez is primal  primal dual
            // moments are primal primal primal
            // operation is thus ppP to ppD
            // shift only in the Z direction
            [[maybe_unused]] auto constexpr iShift = primalToDual();

            if constexpr (dimension == 1)
            {
                // since the linear combination is in the Z direction
                // in 1D or 2D the moment is already on Ez so return 1 point with no shift
                // with coef 1.
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                // in 3D we need two points, the second with a primalToDual shift along Z
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, iShift}, 0.5};
                return std::array{P1, P2};
            }
        }



        NO_DISCARD auto static constexpr cellCenterToFullPrimal()
        {
            // DDD → PPP: average over 2^dim neighboring dual cells
            auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.125};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.125};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift, 0}, 0.125};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, 0, iShift}, 0.125};
                constexpr WeightPoint<dimension> P5{Point<int, dimension>{iShift, iShift, 0}, 0.125};
                constexpr WeightPoint<dimension> P6{Point<int, dimension>{0, iShift, iShift}, 0.125};
                constexpr WeightPoint<dimension> P7{Point<int, dimension>{iShift, 0, iShift}, 0.125};
                constexpr WeightPoint<dimension> P8{Point<int, dimension>{iShift, iShift, iShift}, 0.125};
                return std::array{P1, P2, P3, P4, P5, P6, P7, P8};
            }
        }



        // MHD face/edge projection methods (ported from gridlayoutimplyee_mhd.hpp)

        NO_DISCARD auto static constexpr cellCenterToEdgeX()
        {
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array<WeightPoint<dimension>, 1>{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, iShift, iShift}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr cellCenterToEdgeY()
        {
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, 0, iShift}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr cellCenterToEdgeZ()
        {
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift}, 0.25};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift, 0}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr faceXToCellCenter()
        {
            auto constexpr iShift = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr faceYToCellCenter()
        {
            [[maybe_unused]] auto constexpr iShift = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array<WeightPoint<dimension>, 1>{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr faceZToCellCenter()
        {
            [[maybe_unused]] auto constexpr iShift = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array<WeightPoint<dimension>, 1>{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.};
                return std::array<WeightPoint<dimension>, 1>{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr edgeXToCellCenter()
        {
            auto constexpr iShift = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array<WeightPoint<dimension>, 1>{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, iShift, iShift}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr edgeYToCellCenter()
        {
            [[maybe_unused]] auto constexpr iShift = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, 0, iShift}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr edgeZToCellCenter()
        {
            [[maybe_unused]] auto constexpr iShift = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array<WeightPoint<dimension>, 2>{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift, 0}, 0.25};
                return std::array<WeightPoint<dimension>, 4>{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr BxToMoments()
        {
            // Bx is primal dual dual
            // moments are primal primal primal
            // operation is thus Pdd to Ppp
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, iShift, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }



        NO_DISCARD auto static constexpr ByToMoments()
        {
            // By is dual primal dual
            // moments are primal primal primal
            // operation is thus Dpd to Ppp

            auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, 0, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }




        NO_DISCARD auto static constexpr BzToMoments()
        {
            // Bz is dual dual primal
            // moments are primal primal primal
            // operation is thus Ddp to Ppp

            auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift, 0}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr momentsToBx()
        {
            // moments are primal primal primal
            // Bx is primal dual dual
            // operation is thus Ppp to Pdd
            // shift in Y and Z directions
            [[maybe_unused]] auto constexpr iShift = primalToDual();

            if constexpr (dimension == 1)
            {
                // in 1D Bx is primal in x — no shift needed
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // shift in Y only
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                // shift in Y and Z
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, iShift, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }


        NO_DISCARD auto static constexpr momentsToBy()
        {
            // moments are primal primal primal
            // By is dual primal dual
            // operation is thus Ppp to Dpd
            // shift in X and Z directions
            [[maybe_unused]] auto constexpr iShift = primalToDual();

            if constexpr (dimension == 1)
            {
                // in 1D By is dual in x — shift in X
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                // shift in X only (Z dimension absent)
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                // shift in X and Z
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, 0, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }


        NO_DISCARD auto static constexpr momentsToBz()
        {
            // moments are primal primal primal
            // Bz is dual dual primal
            // operation is thus Ppp to Ddp
            // shift in X and Y directions
            [[maybe_unused]] auto constexpr iShift = primalToDual();

            if constexpr (dimension == 1)
            {
                // in 1D Bz is dual in x — shift in X
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                // shift in X and Y
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                // shift in X and Y
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, iShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{iShift, iShift, 0}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }


        // -----------------------------------------------------------------------
        // B→B cross-centering stencils (face→face, 4-point avg in 2 directions)
        // Used for Maxwell stress contributions to the momentum flux.
        // -----------------------------------------------------------------------

        NO_DISCARD auto static constexpr ByToBx()
        {
            // By: dpd  →  Bx: pdd
            // x: dual→primal (dualToPrimal), y: primal→dual (primalToDual)
            auto constexpr d2p = dualToPrimal();
            auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2p}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2p, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, p2d}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2p, p2d}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2p, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, p2d, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2p, p2d, 0}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr BzToBx()
        {
            // Bz: ddp  →  Bx: pdd
            // x: dual→primal (dualToPrimal), z: primal→dual (primalToDual)
            auto constexpr d2p = dualToPrimal();
            [[maybe_unused]] auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2p}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                // In 2D, Bz is dd and Bx is pd: only x-shift needed
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2p, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                // x: d→p, z: p→d
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2p, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, p2d}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2p, 0, p2d}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr BxToBy()
        {
            // Bx: pdd  →  By: dpd
            // x: primal→dual (primalToDual), y: dual→primal (dualToPrimal)
            auto constexpr p2d = primalToDual();
            auto constexpr d2p = dualToPrimal();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, d2p}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{p2d, d2p}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, d2p, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{p2d, d2p, 0}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr BzToBy()
        {
            // Bz: ddp  →  By: dpd
            // y: dual→primal (dualToPrimal), z: primal→dual (primalToDual)
            auto constexpr d2p = dualToPrimal();
            [[maybe_unused]] auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                // Both dual in 1D — identity
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // Bz in 2D: dd, By in 2D: dp — only y-shift
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, d2p}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                // y: d→p, z: p→d
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, d2p, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, p2d}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, d2p, p2d}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr BxToBz()
        {
            // Bx: pdd  →  Bz: ddp
            // x: primal→dual (primalToDual), z: dual→primal (dualToPrimal)
            auto constexpr p2d = primalToDual();
            [[maybe_unused]] auto constexpr d2p = dualToPrimal();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                // Bx in 2D: pd, Bz in 2D: dd — only x-shift
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                // x: p→d, z: d→p
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, d2p}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{p2d, 0, d2p}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        NO_DISCARD auto static constexpr ByToBz()
        {
            // By: dpd  →  Bz: ddp
            // y: primal→dual (primalToDual), z: dual→primal (dualToPrimal)
            [[maybe_unused]] auto constexpr p2d = primalToDual();
            auto constexpr d2p                  = dualToPrimal();
            if constexpr (dimension == 1)
            {
                // Both dual in 1D — identity
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // By in 2D: dp, Bz in 2D: dd — only y-shift
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                // y: p→d, z: d→p
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, p2d, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, 0, d2p}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{0, p2d, d2p}, 0.25};
                return std::array{P1, P2, P3, P4};
            }
        }

        // -----------------------------------------------------------------------
        // E→face stencils (2-point avg in 1 direction) — for Poynting flux
        // -----------------------------------------------------------------------

        NO_DISCARD auto static constexpr EyToBx()
        {
            // Ey: pdp  →  Bx: pdd — shift in z only (primal→dual)
            [[maybe_unused]] auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // Ey in 2D: pd (no z in 2D) = Bx in 2D: pd — identity
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, p2d}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr EzToBx()
        {
            // Ez: ppd  →  Bx: pdd — shift in y only (primal→dual)
            auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // Ez in 2D: pp → Bx in 2D: pd — shift in y
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, p2d, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr EzToBy()
        {
            // Ez: ppd  →  By: dpd — shift in x only (primal→dual)
            auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                // Ez in 2D: pp → By in 2D: dp — shift in x
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr ExToBy()
        {
            // Ex: dpp  →  By: dpd — shift in z only (primal→dual)
            [[maybe_unused]] auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                // Both dual in 1D — identity
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // Ex in 2D: dp (no z) = By in 2D: dp — identity
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, p2d}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr ExToBz()
        {
            // Ex: dpp  →  Bz: ddp — shift in y only (primal→dual)
            auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                // Both dual in 1D — identity
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                // Ex in 2D: dp → Bz in 2D: dd — shift in y
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, p2d, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr EyToBz()
        {
            // Ey: pdp  →  Bz: ddp — shift in x only (primal→dual)
            auto constexpr p2d = primalToDual();
            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                // Ey in 2D: pd → Bz in 2D: dd — shift in x
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2d, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr ExToMoments()
        {
            // Ex is dual primal primal
            // moments are primal primal primal
            // operation is thus Dpp to Ppp
            // shift only in the X direction
            auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr EyToMoments()
        {
            // Ey is       primal dual   primal
            // moments are primal primal primal
            // operation is thus pDp to pPp
            // shift only in the Y direction
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr EzToMoments()
        {
            // Ez is       primal primal dual
            // moments are primal primal primal
            // operation is thus ppD to ppP
            // shift only in the Z direction
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, iShift}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr JxToMoments()
        {
            // Jx is dual primal primal
            // moments are primal primal primal
            // operation is thus Dpp to Ppp
            // shift only in the X direction

            auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr JyToMoments()
        {
            // Jy is primal dual primal
            // moments are primal primal primal
            // operation is thus pDp to pPp
            // shift only in the Y direction

            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr JzToMoments()
        {
            // Jy is primal primal dual
            // moments are primal primal primal
            // operation is thus ppD to ppP
            // shift only in the Z direction

            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, iShift}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr BxToEx()
        {
            // Bx is primal dual dual
            // Ex is dual primal primal
            // operation is pdd to dpp
            [[maybe_unused]] auto constexpr p2dShift = primalToDual();
            [[maybe_unused]] auto constexpr d2pShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{p2dShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, d2pShift}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{p2dShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{p2dShift, d2pShift},
                                                    0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.125};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, d2pShift, 0}, 0.125};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{p2dShift, 0, 0}, 0.125};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{p2dShift, d2pShift, 0},
                                                    0.125};

                constexpr WeightPoint<dimension> P5{Point<int, dimension>{0, 0, d2pShift}, 0.125};
                constexpr WeightPoint<dimension> P6{Point<int, dimension>{0, d2pShift, d2pShift},
                                                    0.125};
                constexpr WeightPoint<dimension> P7{Point<int, dimension>{p2dShift, 0, d2pShift},
                                                    0.125};
                constexpr WeightPoint<dimension> P8{
                    Point<int, dimension>{p2dShift, d2pShift, d2pShift}, 0.125};
                return std::array{P1, P2, P3, P4, P5, P6, P7, P8};
            }
        }

        NO_DISCARD auto static constexpr ByToEx()
        {
            // By is dual primal dual
            // Ex is dual primal primal
            // operation is thus dpD to dpP
            // shift only in the Z direction
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.0};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, iShift}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr BzToEx()
        {
            // Bz is dual dual primal
            // Ex is dual primal primal
            // operation is thus pDp to pPp
            // shift only in the Y direction
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }

            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr BzToEz()
        {
            // Bz is dual dual primal
            // Ez is primal primal dual
            // operation is thus ddp to ppd
            auto constexpr p2dShift = primalToDual();
            auto constexpr d2pShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2pShift}, 0.5};
                return std::array{P1, P2};
            }

            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2pShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, d2pShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2pShift, d2pShift},
                                                    0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2pShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, d2pShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2pShift, d2pShift, 0},
                                                    0.25};
                constexpr WeightPoint<dimension> P5{Point<int, dimension>{0, 0, p2dShift}, 0.25};
                constexpr WeightPoint<dimension> P6{Point<int, dimension>{d2pShift, 0, p2dShift},
                                                    0.25};
                constexpr WeightPoint<dimension> P7{Point<int, dimension>{0, d2pShift, p2dShift},
                                                    0.25};
                constexpr WeightPoint<dimension> P8{
                    Point<int, dimension>{d2pShift, d2pShift, p2dShift}, 0.25};
                return std::array{P1, P2, P3, P4, P5, P6, P7, P8};
            }
        }

        NO_DISCARD auto static constexpr ByToEz()
        {
            // By is dual primal dual
            // Ez is primal primal dual
            // operation is thus Dpd to Ppd
            // shift only in the X direction
            auto constexpr iShift = dualToPrimal();

            // P1 is always on top of Ez so no shift

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr BxToEz()
        {
            // Bx is primal dual dual
            // Ez is primal primal dual
            // operation is thus pDd to pPd
            // shift only in the Y direction
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift}, 0.5};
                return std::array{P1, P2};
            }

            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr BxToEy()
        {
            // Bx is primal dual dual
            // Ey is primal dual primal
            // operation is thus pdD to pdP
            // shift only in the Z direction
            [[maybe_unused]] auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                // in 3D we need two points, the second with a dualToPrimal shift along Z
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{0, 0, iShift}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr ByToEy()
        {
            // By is dual primal dual
            // Ey is primal dual primal
            // the operation is thus dpd to pdp
            auto constexpr p2dShift = primalToDual();
            auto constexpr d2pShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2pShift}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2pShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, p2dShift}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2pShift, p2dShift},
                                                    0.25};
                return std::array{P1, P2, P3, P4};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{d2pShift, 0, 0}, 0.25};
                constexpr WeightPoint<dimension> P3{Point<int, dimension>{0, p2dShift, 0}, 0.25};
                constexpr WeightPoint<dimension> P4{Point<int, dimension>{d2pShift, p2dShift, 0},
                                                    0.25};
                constexpr WeightPoint<dimension> P5{Point<int, dimension>{0, 0, d2pShift}, 0.25};
                constexpr WeightPoint<dimension> P6{Point<int, dimension>{d2pShift, 0, d2pShift},
                                                    0.25};
                constexpr WeightPoint<dimension> P7{Point<int, dimension>{0, p2dShift, d2pShift},
                                                    0.25};
                constexpr WeightPoint<dimension> P8{
                    Point<int, dimension>{d2pShift, p2dShift, d2pShift}, 0.25};
                return std::array{P1, P2, P3, P4, P5, P6, P7, P8};
            }
        }

        NO_DISCARD auto static constexpr BzToEy()
        {
            // Bz is dual dual primal
            // Ey is primal dual primal
            // operation is thus Ddp to Pdp
            // shift only in the X direction
            auto constexpr iShift = dualToPrimal();

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift}, 0.5};
                return std::array{P1, P2};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0}, 0.5};
                return std::array{P1, P2};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 0.5};
                constexpr WeightPoint<dimension> P2{Point<int, dimension>{iShift, 0, 0}, 0.5};
                return std::array{P1, P2};
            }
        }

        NO_DISCARD auto static constexpr JxToEx()
        {
            // Jx is dual primal primal
            // Ex is dual primal primal
            // operation is thus dpp to dpp
            // no shift for a yee grid

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 1.0};
                return std::array{P1};
            }
        }

        NO_DISCARD auto static constexpr JyToEy()
        {
            // Jy is primal dual primal
            // Ey is primal dual primal
            // operation is thus pdp to pdp
            // no shift for a yee grid

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 1.0};
                return std::array{P1};
            }
        }

        NO_DISCARD auto static constexpr JzToEz()
        {
            // Jz is primal primal dual
            // Ez is primal primal dual
            // operation is thus ppd to ppd
            // no shift for a yee grid

            if constexpr (dimension == 1)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0}, 1};
                return std::array{P1};
            }
            if constexpr (dimension == 2)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0}, 1.0};
                return std::array{P1};
            }
            else if constexpr (dimension == 3)
            {
                constexpr WeightPoint<dimension> P1{Point<int, dimension>{0, 0, 0}, 1.0};
                return std::array{P1};
            }
        }
    }; // namespace core

    /*

    template<std::size_t dim, std::size_t interpOrder>
    void constexpr GridLayoutImpl<dim, interpOrder>::initLinearCombinations_()
    {
        // cf https://hephaistos.lpp.polytechnique.fr/redmine/projects/hyb-par/wiki/Ohm
        // for how to calculate coefficients and shift indexes.

        int dualToPrimal = 0;
        int primalTodual = 0;

        if constexpr (interpOrder == 1 || interpOrder == 2 || interpOrder == 4)
        {
            dualToPrimal = -1;
            primalTodual = 1;
        }
        else if constexpr (interpOrder == 3)
        {
            dualToPrimal = 1;
            primalTodual = -1;
        }

        WeightPoint P1;
        WeightPoint P2;

        // moment to Ex is Ppp to Dpp
        // shift only in X
        // the average is done for all simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = 0.5;
        P2.ix   = primalTodual;
        P2.iy   = 0;
        P2.iz   = 0;
        P2.coef = 0.5;
        momentsToEx_.push_back(P1);
        momentsToEx_.push_back(P2);


        // moment to Ey is pPp to pDp
        // shift only in Y
        // the average is done only for 2D and 3D simulation
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim >= 2) ? 0.5 : 1.;
        momentsToEy_.push_back(P1);

        // in 2 and 3D, add another point and average
        if (dim >= 2)
        {
            P2.ix   = 0;
            P2.iy   = primalTodual;
            P2.iz   = 0;
            P2.coef = 0.5;
            momentsToEy_.push_back(P2);
        }


        // moment to Ez is ppP to ppD
        // shift only in Z
        // the average is done only for 3D simulation
        // hence for 1D and 2D runs coef==1
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim == 3) ? 0.5 : 1;
        momentsToEz_.push_back(P1);

        if (dim == 3)
        {
            P2.ix   = 0;
            P2.iy   = 0;
            P2.iz   = primalTodual;
            P2.coef = 0.5;
            momentsToEz_.push_back(P2);
        }




        // Bx to Ey is pdD to pdP
        // shift only in Z
        // the average is done only for 3D simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim == 3) ? 0.5 : 1;
        BxToEy_.push_back(P1);

        if (dim == 3)
        {
            P2.ix   = 0;
            P2.iy   = 0;
            P2.iz   = dualToPrimal;
            P2.coef = 0.5;
            BxToEy_.push_back(P2);
        }



        // Bx to Ez is pDd to pPd
        // shift in the Y direction only
        // the average is done for 2D and 3D simulations
        // hence for 1D simulations coef is 1
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim >= 2) ? 0.5 : 1;
        BxToEz_.push_back(P1);

        if (dim >= 2)
        {
            P2.ix   = 0;
            P2.iy   = dualToPrimal;
            P2.iz   = 0;
            P2.coef = 0.5;
            BxToEz_.push_back(P2);
        }


        // By to Ex is dpD to dpP
        // shift only in the Z direction
        // averaging is done only for 3D simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim == 3) ? 0.5 : 1;
        ByToEx_.push_back(P1);

        if (dim == 3)
        {
            P2.ix   = 0;
            P2.iy   = 0;
            P2.iz   = dualToPrimal;
            P2.coef = 0.5;
            ByToEx_.push_back(P2);
        }

        // By to Ez is Dpd to Ppd
        // shift only in the X direction
        // the averaging is done in all simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = 0.5;
        P2.ix   = dualToPrimal;
        P2.iy   = 0;
        P2.iz   = 0;
        P2.coef = 0.5;
        ByToEz_.push_back(P1);
        ByToEz_.push_back(P2);


        // Bz to Ex is dDp to dPp
        // shift only in the Y direction
        // the averaging is done for 2D and 3D simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim >= 2) ? 0.5 : 1;
        BzToEx_.push_back(P1);

        if (dim >= 2)
        {
            P2.ix   = 0;
            P2.iy   = dualToPrimal;
            P2.iz   = 0;
            P2.coef = 0.5;
            BzToEx_.push_back(P2);
        }


        // Bz to Ey is Ddp to Pdp
        // shift only in the X direction
        // the averaging is done for all simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = 0.5;
        BzToEy_.push_back(P1);
        P2.ix   = dualToPrimal;
        P2.iy   = 0;
        P2.iz   = 0;
        P2.coef = 0.5;
        BzToEy_.push_back(P2);


        // Ex to Moment is Dpp to Ppp
        // shift only in the X direction
        // the averaging is done for all simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = 0.5;
        ExToMoment_.push_back(P1);
        P2.ix   = dualToPrimal;
        P2.iy   = 0;
        P2.iz   = 0;
        P2.coef = 0.5;
        ExToMoment_.push_back(P2);


        // Ey to Moment is pDp to PPP
        // shift tis only in the Y direction
        // the averaging is done for 2D and 3D simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim >= 2) ? 0.5 : 1;
        EyToMoment_.push_back(P1);

        if (dim >= 2)
        {
            P2.ix   = 0;
            P2.iy   = dualToPrimal;
            P2.iz   = 0;
            P2.coef = 0.5;
            EyToMoment_.push_back(P2);
        }


        // Ez to Moment is ppD on ppP
        // shift only in the Z direction
        // the averaging is only for 3D simulations
        P1.ix   = 0;
        P1.iy   = 0;
        P1.iz   = 0;
        P1.coef = (dim == 3) ? 0.5 : 1;
        EzToMoment_.push_back(P1);

        if (dim == 3)
        {
            P2.ix   = 0;
            P2.iy   = 0;
            P2.iz   = dualToPrimal;
            P2.coef = 0.5;
            EzToMoment_.push_back(P2);
        }
    }




    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::momentsToEx() const
    {
        return this->momentsToEx_;
    }


    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::momentsToEy() const
    {
        return this->momentsToEy_;
    }


    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::momentsToEz() const
    {
        return this->momentsToEz_;
    }

    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::ByToEx() const
    {
        return this->ByToEx_;
    }

    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::ByToEz() const
    {
        return this->ByToEz_;
    }

    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::BxToEy() const
    {
        return this->BxToEy_;
    }

    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::BxToEz() const
    {
        return this->BxToEz_;
    }

    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::BzToEy() const
    {
        return this->BzToEy_;
    }

    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::BzToEx() const
    {
        return this->BzToEx_;
    }



    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::ExToMoment() const
    {
        return this->ExToMoment_;
    }


    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::EyToMoment() const
    {
        return this->EyToMoment_;
    }


    template<std::size_t dim>
    LinearCombination const& GridLayoutImpl<Layout::Yee, dim>::EzToMoment() const
    {
        return this->EzToMoment_;
    }

    */

} // namespace core
} // namespace PHARE

#endif // PHARE_CORE_GRID_GRIDLAYOUTYEE_HPP
