#ifndef PHARE_CORE_MHD_MHD_QUANTITIES_HPP
#define PHARE_CORE_MHD_MHD_QUANTITIES_HPP

#include "core/def.hpp"

#include <array>
#include <tuple>
#include <stdexcept>


namespace PHARE::core
{
class MHDQuantity
{
public:
    enum class Scalar {
        rho, // density
        Vx,  // velocity components
        Vy,
        Vz,
        Bx_FV, // magnetic field components
        By_FV,
        Bz_FV,
        P,     // pressure
        Etot,  // total energy
        rhoVx, // momentum components
        rhoVy,
        rhoVz,

        Bx_CT,
        By_CT,
        Bz_CT,
        Ex, // electric field components
        Ey,
        Ez,
        Jx, // current density components
        Jy,
        Jz,

        Bx_RSx,
        By_RSx,
        Bz_RSx,

        Bx_RSy,
        By_RSy,
        Bz_RSy,

        Bx_RSz,
        By_RSz,
        Bz_RSz,

        count
    };
    enum class Vector { V, B_FV, rhoV, B_CT, E, J, B_RSx, B_RSy, B_RSz };
    enum class Tensor { count };

    template<std::size_t rank, typename = std::enable_if_t<rank == 1 or rank == 2, void>>
    using TensorType = std::conditional_t<rank == 1, Vector, Tensor>;

    NO_DISCARD static constexpr auto V() { return componentsQuantities(Vector::V); }
    NO_DISCARD static constexpr auto B_FV() { return componentsQuantities(Vector::B_FV); }
    NO_DISCARD static constexpr auto rhoV() { return componentsQuantities(Vector::rhoV); }

    NO_DISCARD static constexpr auto B_CT() { return componentsQuantities(Vector::B_CT); }
    NO_DISCARD static constexpr auto E() { return componentsQuantities(Vector::E); }
    NO_DISCARD static constexpr auto J() { return componentsQuantities(Vector::J); }

    NO_DISCARD static constexpr auto B_RSx() { return componentsQuantities(Vector::B_RSx); }
    NO_DISCARD static constexpr auto B_RSy() { return componentsQuantities(Vector::B_RSy); }
    NO_DISCARD static constexpr auto B_RSz() { return componentsQuantities(Vector::B_RSz); }

    NO_DISCARD static constexpr std::array<Scalar, 3> componentsQuantities(Vector qty)
    {
        if (qty == Vector::V)
            return {{Scalar::Vx, Scalar::Vy, Scalar::Vz}};

        if (qty == Vector::B_FV)
            return {{Scalar::Bx_FV, Scalar::By_FV, Scalar::Bz_FV}};

        if (qty == Vector::rhoV)
            return {{Scalar::rhoVx, Scalar::rhoVy, Scalar::rhoVz}};


        if (qty == Vector::B_CT)
            return {{Scalar::Bx_CT, Scalar::By_CT, Scalar::Bz_CT}};

        if (qty == Vector::E)
            return {{Scalar::Ex, Scalar::Ey, Scalar::Ez}};

        if (qty == Vector::J)
            return {{Scalar::Jx, Scalar::Jy, Scalar::Jz}};


        if (qty == Vector::B_RSx)
            return {{Scalar::Bx_RSx, Scalar::By_RSx, Scalar::Bz_RSx}};

        if (qty == Vector::B_RSy)
            return {{Scalar::Bx_RSy, Scalar::By_RSy, Scalar::Bz_RSy}};

        if (qty == Vector::B_RSz)
            return {{Scalar::Bx_RSz, Scalar::By_RSz, Scalar::Bz_RSz}};

        throw std::runtime_error("Error - invalid Vector");
    }

    NO_DISCARD static constexpr auto B_CT_items()
    {
        auto const& [Bx_CT, By_CT, Bz_CT] = B_CT();
        return std::make_tuple(std::make_pair("Bx_CT", Bx_CT), std::make_pair("By_CT", By_CT),
                               std::make_pair("Bz_CT", Bz_CT));
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
