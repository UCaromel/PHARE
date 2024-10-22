#ifndef PHARE_CORE_GRID_GRIDLAYOUTYEE_MHD_HPP
#define PHARE_CORE_GRID_GRIDLAYOUTYEE_MHD_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "core/def.hpp"
#include "core/mhd/mhd_quantities.hpp"
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
    template<std::size_t dim, std::size_t interpOrder>
    class GridLayoutImplYeeMHD
    {
        // ------------------------------------------------------------------------
        //                              PRIVATE
        // ------------------------------------------------------------------------
    public:
        static constexpr std::size_t dimension    = dim;
        static constexpr std::size_t interp_order = interpOrder;
        static constexpr std::string_view type    = "yee";
        using quantity_type                       = MHDQuantity;

        /**
         * @brief GridLayoutImpl<Selector<Layout,Layout::Yee>,dim>::initLayoutCentering_ initialize
         * the table MHDQuantityCentering_. This is THE important array in the GridLayout module.
         * This table knows which quantity is primal/dual along each direction. It is **this** array
         * that
         * **defines** what a Yee Layout is. Once this array is defined, the rest of the GridLayout
         * needs this array OK and can go on from here... hence all other functions in the Yee
         * interface are just calling private implementation common to all layouts
         */
        constexpr auto static initLayoutCentering_()
        {
            const gridDataT data{};

            const std::array<QtyCentering, NBR_COMPO> Rho = {{data.dual, data.dual, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> Vx = {{data.dual, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> Vy = {{data.dual, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> Vz = {{data.dual, data.dual, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> Bx_FV = {{data.dual, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> By_FV = {{data.dual, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> Bz_FV = {{data.dual, data.dual, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> P = {{data.dual, data.dual, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> rhoVx = {{data.dual, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> rhoVy = {{data.dual, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> rhoVz = {{data.dual, data.dual, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> Etot = {{data.dual, data.dual, data.dual}};


            const std::array<QtyCentering, NBR_COMPO> Bx_CT = {{data.primal, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> By_CT = {{data.dual, data.primal, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> Bz_CT = {{data.dual, data.dual, data.primal}};

            const std::array<QtyCentering, NBR_COMPO> Ex = {{data.dual, data.primal, data.primal}};
            const std::array<QtyCentering, NBR_COMPO> Ey = {{data.primal, data.dual, data.primal}};
            const std::array<QtyCentering, NBR_COMPO> Ez = {{data.primal, data.primal, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> Jx = {{data.dual, data.primal, data.primal}};
            const std::array<QtyCentering, NBR_COMPO> Jy = {{data.primal, data.dual, data.primal}};
            const std::array<QtyCentering, NBR_COMPO> Jz = {{data.primal, data.primal, data.dual}};


            const std::array<QtyCentering, NBR_COMPO> Bx_RSx
                = {{data.primal, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> By_RSx
                = {{data.primal, data.dual, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> Bz_RSx
                = {{data.primal, data.dual, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> Bx_RSy
                = {{data.dual, data.primal, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> By_RSy
                = {{data.dual, data.primal, data.dual}};
            const std::array<QtyCentering, NBR_COMPO> Bz_RSy
                = {{data.dual, data.primal, data.dual}};

            const std::array<QtyCentering, NBR_COMPO> Bx_RSz
                = {{data.dual, data.dual, data.primal}};
            const std::array<QtyCentering, NBR_COMPO> By_RSz
                = {{data.dual, data.dual, data.primal}};
            const std::array<QtyCentering, NBR_COMPO> Bz_RSz
                = {{data.dual, data.dual, data.primal}};


            const std::array<std::array<QtyCentering, NBR_COMPO>,
                             static_cast<std::size_t>(MHDQuantity::Scalar::count)>
                _QtyCentering{Rho,    Vx,     Vy,     Vz,     P,      rhoVx,  rhoVy,  rhoVz,
                              Bx_FV,  By_FV,  Bz_FV,  Etot,   Bx_CT,  By_CT,  Bz_CT,  Ex,
                              Ey,     Ez,     Jx,     Jy,     Jz,     Bx_RSx, By_RSx, Bz_RSx,
                              Bx_RSy, By_RSy, Bz_RSy, Bx_RSz, By_RSz, Bz_RSz};

            return _QtyCentering;
        }

        //! says for each MHDQuantity::Quantity whether it is primal or dual, in each direction
        constexpr const static std::array<std::array<QtyCentering, NBR_COMPO>,
                                          static_cast<std::size_t>(MHDQuantity::Scalar::count)>
            _QtyCentering_{initLayoutCentering_()};

        static const std::size_t dim_{dim};

        // ------------------------------------------------------------------------
        //                          PUBLIC INTERFACE
        // ------------------------------------------------------------------------
    public:
        NO_DISCARD constexpr static std::array<QtyCentering, dim>
        centering(MHDQuantity::Scalar MHDQuantity)
        {
            constexpr gridDataT_mhd gridData_{};
            if constexpr (dim == 1)
            {
                switch (MHDQuantity)
                {
                    case MHDQuantity::Scalar::rho:
                        return {{_QtyCentering_[gridData_.irho][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Vx:
                        return {{_QtyCentering_[gridData_.iVx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Vy:
                        return {{_QtyCentering_[gridData_.iVy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Vz:
                        return {{_QtyCentering_[gridData_.iVz][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bx_FV:
                        return {{_QtyCentering_[gridData_.iBx_FV][gridData_.idirX]}};
                    case MHDQuantity::Scalar::By_FV:
                        return {{_QtyCentering_[gridData_.iBy_FV][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bz_FV:
                        return {{_QtyCentering_[gridData_.iBz_FV][gridData_.idirX]}};
                    case MHDQuantity::Scalar::P:
                        return {{_QtyCentering_[gridData_.iP][gridData_.idirX]}};
                    case MHDQuantity::Scalar::rhoVx:
                        return {{_QtyCentering_[gridData_.irhoVx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::rhoVy:
                        return {{_QtyCentering_[gridData_.irhoVy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::rhoVz:
                        return {{_QtyCentering_[gridData_.irhoVz][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bx_CT:
                        return {{_QtyCentering_[gridData_.iBx_CT][gridData_.idirX]}};
                    case MHDQuantity::Scalar::By_CT:
                        return {{_QtyCentering_[gridData_.iBy_CT][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bz_CT:
                        return {{_QtyCentering_[gridData_.iBz_CT][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Ex:
                        return {{_QtyCentering_[gridData_.iEx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Ey:
                        return {{_QtyCentering_[gridData_.iEy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Ez:
                        return {{_QtyCentering_[gridData_.iEz][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Jx:
                        return {{_QtyCentering_[gridData_.iJx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Jy:
                        return {{_QtyCentering_[gridData_.iJy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Jz:
                        return {{_QtyCentering_[gridData_.iJz][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bx_RSx:
                        return {{_QtyCentering_[gridData_.iBx_RSx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::By_RSx:
                        return {{_QtyCentering_[gridData_.iBy_RSx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bz_RSx:
                        return {{_QtyCentering_[gridData_.iBz_RSx][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bx_RSy:
                        return {{_QtyCentering_[gridData_.iBx_RSy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::By_RSy:
                        return {{_QtyCentering_[gridData_.iBy_RSy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bz_RSy:
                        return {{_QtyCentering_[gridData_.iBz_RSy][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bx_RSz:
                        return {{_QtyCentering_[gridData_.iBx_RSz][gridData_.idirX]}};
                    case MHDQuantity::Scalar::By_RSz:
                        return {{_QtyCentering_[gridData_.iBy_RSz][gridData_.idirX]}};
                    case MHDQuantity::Scalar::Bz_RSz:
                        return {{_QtyCentering_[gridData_.iBz_RSz][gridData_.idirX]}};
                    default: throw std::runtime_error("Wrong MHDQuantity");
                }
            }

            else if constexpr (dim == 2)
            {
                switch (MHDQuantity)
                {
                    case MHDQuantity::Scalar::rho:
                        return {{_QtyCentering_[gridData_.irho][gridData_.idirX],
                                 _QtyCentering_[gridData_.irho][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Vx:
                        return {{_QtyCentering_[gridData_.iVx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iVx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Vy:
                        return {{_QtyCentering_[gridData_.iVy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iVy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Vz:
                        return {{_QtyCentering_[gridData_.iVz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iVz][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bx_FV:
                        return {{_QtyCentering_[gridData_.iBx_FV][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_FV][gridData_.idirY]}};
                    case MHDQuantity::Scalar::By_FV:
                        return {{_QtyCentering_[gridData_.iBy_FV][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_FV][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bz_FV:
                        return {{_QtyCentering_[gridData_.iBz_FV][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_FV][gridData_.idirY]}};
                    case MHDQuantity::Scalar::P:
                        return {{_QtyCentering_[gridData_.iP][gridData_.idirX],
                                 _QtyCentering_[gridData_.iP][gridData_.idirY]}};
                    case MHDQuantity::Scalar::rhoVx:
                        return {{_QtyCentering_[gridData_.irhoVx][gridData_.idirX],
                                 _QtyCentering_[gridData_.irhoVx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::rhoVy:
                        return {{_QtyCentering_[gridData_.irhoVy][gridData_.idirX],
                                 _QtyCentering_[gridData_.irhoVy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::rhoVz:
                        return {{_QtyCentering_[gridData_.irhoVz][gridData_.idirX],
                                 _QtyCentering_[gridData_.irhoVz][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bx_CT:
                        return {{_QtyCentering_[gridData_.iBx_CT][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_CT][gridData_.idirY]}};
                    case MHDQuantity::Scalar::By_CT:
                        return {{_QtyCentering_[gridData_.iBy_CT][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_CT][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bz_CT:
                        return {{_QtyCentering_[gridData_.iBz_CT][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_CT][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Ex:
                        return {{_QtyCentering_[gridData_.iEx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Ey:
                        return {{_QtyCentering_[gridData_.iEy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Ez:
                        return {{_QtyCentering_[gridData_.iEz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEz][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Jx:
                        return {{_QtyCentering_[gridData_.iJx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Jy:
                        return {{_QtyCentering_[gridData_.iJy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Jz:
                        return {{_QtyCentering_[gridData_.iJz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJz][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bx_RSx:
                        return {{_QtyCentering_[gridData_.iBx_RSx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_RSx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::By_RSx:
                        return {{_QtyCentering_[gridData_.iBy_RSx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_RSx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bz_RSx:
                        return {{_QtyCentering_[gridData_.iBz_RSx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_RSx][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bx_RSy:
                        return {{_QtyCentering_[gridData_.iBx_RSy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_RSy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::By_RSy:
                        return {{_QtyCentering_[gridData_.iBy_RSy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_RSy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bz_RSy:
                        return {{_QtyCentering_[gridData_.iBz_RSy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_RSy][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bx_RSz:
                        return {{_QtyCentering_[gridData_.iBx_RSz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_RSz][gridData_.idirY]}};
                    case MHDQuantity::Scalar::By_RSz:
                        return {{_QtyCentering_[gridData_.iBy_RSz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_RSz][gridData_.idirY]}};
                    case MHDQuantity::Scalar::Bz_RSz:
                        return {{_QtyCentering_[gridData_.iBz_RSz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_RSz][gridData_.idirY]}};
                    default: throw std::runtime_error("Wrong MHDQuantity");
                }
            }

            else if constexpr (dim == 3)
            {
                switch (MHDQuantity)
                {
                    case MHDQuantity::Scalar::rho:
                        return {{_QtyCentering_[gridData_.irho][gridData_.idirX],
                                 _QtyCentering_[gridData_.irho][gridData_.idirY],
                                 _QtyCentering_[gridData_.irho][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Vx:
                        return {{_QtyCentering_[gridData_.iVx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iVx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iVx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Vy:
                        return {{_QtyCentering_[gridData_.iVy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iVy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iVy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Vz:
                        return {{_QtyCentering_[gridData_.iVz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iVz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iVz][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bx_FV:
                        return {{_QtyCentering_[gridData_.iBx_FV][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_FV][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx_FV][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::By_FV:
                        return {{_QtyCentering_[gridData_.iBy_FV][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_FV][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy_FV][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bz_FV:
                        return {{_QtyCentering_[gridData_.iBz_FV][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_FV][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz_FV][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::P:
                        return {{_QtyCentering_[gridData_.iP][gridData_.idirX],
                                 _QtyCentering_[gridData_.iP][gridData_.idirY],
                                 _QtyCentering_[gridData_.iP][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::rhoVx:
                        return {{_QtyCentering_[gridData_.irhoVx][gridData_.idirX],
                                 _QtyCentering_[gridData_.irhoVx][gridData_.idirY],
                                 _QtyCentering_[gridData_.irhoVx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::rhoVy:
                        return {{_QtyCentering_[gridData_.irhoVy][gridData_.idirX],
                                 _QtyCentering_[gridData_.irhoVy][gridData_.idirY],
                                 _QtyCentering_[gridData_.irhoVy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::rhoVz:
                        return {{_QtyCentering_[gridData_.irhoVz][gridData_.idirX],
                                 _QtyCentering_[gridData_.irhoVz][gridData_.idirY],
                                 _QtyCentering_[gridData_.irhoVz][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bx_CT:
                        return {{_QtyCentering_[gridData_.iBx_CT][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_CT][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx_CT][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::By_CT:
                        return {{_QtyCentering_[gridData_.iBy_CT][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_CT][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy_CT][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bz_CT:
                        return {{_QtyCentering_[gridData_.iBz_CT][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_CT][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz_CT][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Ex:
                        return {{_QtyCentering_[gridData_.iEx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iEx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Ey:
                        return {{_QtyCentering_[gridData_.iEy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iEy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Ez:
                        return {{_QtyCentering_[gridData_.iEz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iEz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iEz][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Jx:
                        return {{_QtyCentering_[gridData_.iJx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iJx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Jy:
                        return {{_QtyCentering_[gridData_.iJy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iJy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Jz:
                        return {{_QtyCentering_[gridData_.iJz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iJz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iJz][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bx_RSx:
                        return {{_QtyCentering_[gridData_.iBx_RSx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_RSx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx_RSx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::By_RSx:
                        return {{_QtyCentering_[gridData_.iBy_RSx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_RSx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy_RSx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bz_RSx:
                        return {{_QtyCentering_[gridData_.iBz_RSx][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_RSx][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz_RSx][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bx_RSy:
                        return {{_QtyCentering_[gridData_.iBx_RSy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_RSy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx_RSy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::By_RSy:
                        return {{_QtyCentering_[gridData_.iBy_RSy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_RSy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy_RSy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bz_RSy:
                        return {{_QtyCentering_[gridData_.iBz_RSy][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_RSy][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz_RSy][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bx_RSz:
                        return {{_QtyCentering_[gridData_.iBx_RSz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBx_RSz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBx_RSz][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::By_RSz:
                        return {{_QtyCentering_[gridData_.iBy_RSz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBy_RSz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBy_RSz][gridData_.idirZ]}};
                    case MHDQuantity::Scalar::Bz_RSz:
                        return {{_QtyCentering_[gridData_.iBz_RSz][gridData_.idirX],
                                 _QtyCentering_[gridData_.iBz_RSz][gridData_.idirY],
                                 _QtyCentering_[gridData_.iBz_RSz][gridData_.idirZ]}};
                    default: throw std::runtime_error("Wrong MHDQuantity");
                }
            }
        }

        NO_DISCARD constexpr static std::array<std::array<QtyCentering, dim>, 3>
        centering(MHDQuantity::Vector MHDQuantity)
        {
            switch (MHDQuantity)
            {
                case MHDQuantity::Vector::V:
                    return {{centering(MHDQuantity::Scalar::Vx), centering(MHDQuantity::Scalar::Vy),
                             centering(MHDQuantity::Scalar::Vz)}};

                case MHDQuantity::Vector::B_FV:
                    return {{centering(MHDQuantity::Scalar::Bx_FV),
                             centering(MHDQuantity::Scalar::By_FV),
                             centering(MHDQuantity::Scalar::Bz_FV)}};

                case MHDQuantity::Vector::rhoV:
                    return {{centering(MHDQuantity::Scalar::rhoVx),
                             centering(MHDQuantity::Scalar::rhoVy),
                             centering(MHDQuantity::Scalar::rhoVz)}};

                case MHDQuantity::Vector::B_CT:
                    return {{centering(MHDQuantity::Scalar::Bx_CT),
                             centering(MHDQuantity::Scalar::By_CT),
                             centering(MHDQuantity::Scalar::Bz_CT)}};

                case MHDQuantity::Vector::E:
                    return {{centering(MHDQuantity::Scalar::Ex), centering(MHDQuantity::Scalar::Ey),
                             centering(MHDQuantity::Scalar::Ez)}};

                case MHDQuantity::Vector::J:
                    return {{centering(MHDQuantity::Scalar::Jx), centering(MHDQuantity::Scalar::Jy),
                             centering(MHDQuantity::Scalar::Jz)}};

                case MHDQuantity::Vector::B_RSx:
                    return {{centering(MHDQuantity::Scalar::Bx_RSx),
                             centering(MHDQuantity::Scalar::By_RSx),
                             centering(MHDQuantity::Scalar::Bz_RSx)}};

                case MHDQuantity::Vector::B_RSy:
                    return {{centering(MHDQuantity::Scalar::Bx_RSy),
                             centering(MHDQuantity::Scalar::By_RSy),
                             centering(MHDQuantity::Scalar::Bz_RSy)}};

                case MHDQuantity::Vector::B_RSz:
                    return {{centering(MHDQuantity::Scalar::Bx_RSz),
                             centering(MHDQuantity::Scalar::By_RSz),
                             centering(MHDQuantity::Scalar::Bz_RSz)}};

                default: throw std::runtime_error("Wrong MHDQuantity");
            }
        }

        NO_DISCARD auto static constexpr dualToPrimal() { return -1; }

        NO_DISCARD auto static constexpr primalToDual() { return 1; }
    }; // namespace core

} // namespace core
} // namespace PHARE

#endif // PHARE_CORE_GRID_GRIDLAYOUTYEE_MHD_HPP
