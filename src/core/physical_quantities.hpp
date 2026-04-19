#ifndef PHARE_CORE_PHYSICAL_QUANTITIES_HPP
#define PHARE_CORE_PHYSICAL_QUANTITIES_HPP

#include "core/def.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <stdexcept>


namespace PHARE::core
{
class PhysicalQuantity
{
public:
    enum class Scalar : uint32_t {
        Bx, // magnetic field components (shared, Yee face-centered)
        By,
        Bz,
        Ex, // electric field components (shared, Yee edge-centered)
        Ey,
        Ez,
        Jx, // current density components (shared, Yee edge-centered)
        Jy,
        Jz,

        Hyb_rho, // Hybrid charge density (ppp)
        Hyb_Vx,  // Hybrid bulk velocity components (ppp)
        Hyb_Vy,
        Hyb_Vz,
        Hyb_P, // Hybrid pressure (ppp)

        Hyb_Mxx, // Hybrid momentum tensor components (ppp)
        Hyb_Mxy,
        Hyb_Mxz,
        Hyb_Myy,
        Hyb_Myz,
        Hyb_Mzz,

        MHD_rho, // MHD mass density (ddd)
        MHD_Vx,  // MHD velocity components (ddd)
        MHD_Vy,
        MHD_Vz,
        MHD_P,    // MHD pressure (ddd)
        MHD_rhoVx, // MHD momentum components (ddd)
        MHD_rhoVy,
        MHD_rhoVz,
        MHD_Etot, // MHD total energy (ddd)

        ScalarFlux_x, // face-centered scalar flux quantities
        ScalarFlux_y,
        ScalarFlux_z,
        VecFluxX_x, // x-face vector flux components (pdd)
        VecFluxY_x,
        VecFluxZ_x,
        VecFluxX_y, // y-face vector flux components (dpd)
        VecFluxY_y,
        VecFluxZ_y,
        VecFluxX_z, // z-face vector flux components (ddp)
        VecFluxY_z,
        VecFluxZ_z,

        ScalarAllPrimal, // MHD synthetic all-primal scalar (ppp)
        VecAllPrimalX,   // MHD synthetic all-primal vector components (ppp)
        VecAllPrimalY,
        VecAllPrimalZ,

        count
    };

    enum class Vector : uint32_t {
        B,
        E,
        J,
        Hyb_V,
        MHD_V,
        MHD_rhoV,
        VecFlux_x,
        VecFlux_y,
        VecFlux_z,
        VecAllPrimal,
        count
    };

    enum class Tensor : uint32_t { M, count };

    static constexpr auto all_primal_field = Scalar::Hyb_rho;

    template<std::size_t rank, typename = std::enable_if_t<rank == 1 or rank == 2, void>>
    using TensorType = std::conditional_t<rank == 1, Vector, Tensor>;

    NO_DISCARD static constexpr auto B() { return componentsQuantities(Vector::B); }
    NO_DISCARD static constexpr auto E() { return componentsQuantities(Vector::E); }
    NO_DISCARD static constexpr auto J() { return componentsQuantities(Vector::J); }
    NO_DISCARD static constexpr auto Hyb_V() { return componentsQuantities(Vector::Hyb_V); }
    NO_DISCARD static constexpr auto MHD_V() { return componentsQuantities(Vector::MHD_V); }
    NO_DISCARD static constexpr auto MHD_rhoV() { return componentsQuantities(Vector::MHD_rhoV); }
    NO_DISCARD static constexpr auto VecFlux_x() { return componentsQuantities(Vector::VecFlux_x); }
    NO_DISCARD static constexpr auto VecFlux_y() { return componentsQuantities(Vector::VecFlux_y); }
    NO_DISCARD static constexpr auto VecFlux_z() { return componentsQuantities(Vector::VecFlux_z); }
    NO_DISCARD static constexpr auto VecAllPrimal()
    {
        return componentsQuantities(Vector::VecAllPrimal);
    }

    NO_DISCARD static constexpr std::array<Scalar, 3> componentsQuantities(Vector qty)
    {
        if (qty == Vector::B)
            return {{Scalar::Bx, Scalar::By, Scalar::Bz}};

        if (qty == Vector::E)
            return {{Scalar::Ex, Scalar::Ey, Scalar::Ez}};

        if (qty == Vector::J)
            return {{Scalar::Jx, Scalar::Jy, Scalar::Jz}};

        if (qty == Vector::Hyb_V)
            return {{Scalar::Hyb_Vx, Scalar::Hyb_Vy, Scalar::Hyb_Vz}};

        if (qty == Vector::MHD_V)
            return {{Scalar::MHD_Vx, Scalar::MHD_Vy, Scalar::MHD_Vz}};

        if (qty == Vector::MHD_rhoV)
            return {{Scalar::MHD_rhoVx, Scalar::MHD_rhoVy, Scalar::MHD_rhoVz}};

        if (qty == Vector::VecFlux_x)
            return {{Scalar::VecFluxX_x, Scalar::VecFluxY_x, Scalar::VecFluxZ_x}};

        if (qty == Vector::VecFlux_y)
            return {{Scalar::VecFluxX_y, Scalar::VecFluxY_y, Scalar::VecFluxZ_y}};

        if (qty == Vector::VecFlux_z)
            return {{Scalar::VecFluxX_z, Scalar::VecFluxY_z, Scalar::VecFluxZ_z}};

        if (qty == Vector::VecAllPrimal)
            return {{Scalar::VecAllPrimalX, Scalar::VecAllPrimalY, Scalar::VecAllPrimalZ}};

        throw std::runtime_error("Error - invalid Vector");
    }

    NO_DISCARD static constexpr std::array<Scalar, 6> componentsQuantities(Tensor qty)
    {
        // only the Hybrid momentum tensor M exists
        return {{Scalar::Hyb_Mxx, Scalar::Hyb_Mxy, Scalar::Hyb_Mxz, Scalar::Hyb_Myy,
                 Scalar::Hyb_Myz, Scalar::Hyb_Mzz}};
    }

    NO_DISCARD static constexpr auto B_items()
    {
        auto const& [Bx, By, Bz] = B();
        return std::make_tuple(std::make_pair("Bx", Bx), std::make_pair("By", By),
                               std::make_pair("Bz", Bz));
    }

    NO_DISCARD static constexpr auto E_items()
    {
        auto const& [Ex, Ey, Ez] = E();
        return std::make_tuple(std::make_pair("Ex", Ex), std::make_pair("Ey", Ey),
                               std::make_pair("Ez", Ez));
    }
};

} // namespace PHARE::core

#endif
