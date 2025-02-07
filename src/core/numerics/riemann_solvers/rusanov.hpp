#ifndef CORE_NUMERICS_RIEMANN_SOLVERS_RUSANOV_HPP
#define CORE_NUMERICS_RIEMANN_SOLVERS_RUSANOV_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/riemann_solvers/mhd_speeds.hpp"

namespace PHARE::core
{
template<typename GridLayout, bool Hall>
class Rusanov
{
public:
    Rusanov(GridLayout const& layout, double const gamma)
        : layout_{layout}
        , gamma_{gamma}
    {
    }

    template<auto direction>
<<<<<<< HEAD
    auto solve(auto& uL, auto& uR, auto const& fL, auto const& fR) const
    {
        auto const speeds = rusanov_speeds_<direction>(uL, uR);

        uL.to_conservative(gamma_);
        uR.to_conservative(gamma_);

        auto const [hydro_speed, mag_speed] = speeds;

        if constexpr (Hall)
        {
            auto split = [](auto const& a) {
                auto hydro = std::make_tuple(a.rho, a.rhoV().x, a.rhoV().y, a.rhoV().z);
                auto mag   = std::make_tuple(a.B.x, a.B.y, a.B.z, a.Etot());
                return std::make_pair(hydro, mag);
            };

            auto [uLhydro, uLmag] = split(uL);
            auto [uRhydro, uRmag] = split(uR);

            auto const [fLhydro, fLmag] = split(fL);
            auto const [fRhydro, fRmag] = split(fR);

            auto [Frho, FrhoVx, FrhoVy, FrhoVz]
                = rusanov_(uLhydro, uRhydro, fLhydro, fRhydro, hydro_speed);
            auto [FBx, FBy, FBz, FEtot] = rusanov_(uLmag, uRmag, fLmag, fRmag, mag_speed);

            return PerIndex{Frho, {FrhoVx, FrhoVy, FrhoVz}, {FBx, FBy, FBz}, FEtot};
=======
    PerIndex solve(PerIndex& uL, PerIndex& uR, PerIndex const& fL, PerIndex const& fR) const
    {
        auto const s = rusanov_speeds_<direction>(uL, uR);

        uL.toConservative(gamma_);
        uR.toConservative(gamma_);

        if constexpr (Hall)
        {
            auto const [S, Sb] = s;

            auto ul  = std::make_tuple(uL.rho, uL.Vx, uL.Vy, uL.Vz);
            auto ulb = std::make_tuple(uL.Bx, uL.By, uL.Bz, uL.P);
            auto ur  = std::make_tuple(uR.rho, uR.Vx, uR.Vy, uR.Vz);
            auto urb = std::make_tuple(uR.Bx, uR.By, uR.Bz, uR.P);

            auto fl  = std::make_tuple(fL.rho, fL.Vx, fL.Vy, fL.Vz);
            auto flb = std::make_tuple(fL.Bx, fL.By, fL.Bz, fL.P);
            auto fr  = std::make_tuple(fR.rho, fR.Vx, fR.Vy, fR.Vz);
            auto frb = std::make_tuple(fR.Bx, fR.By, fR.Bz, fR.P);

            auto [Frho, FrhoVx, FrhoVy, FrhoVz] = rusanov_(ul, ur, fl, fr, S);
            auto [FBx, FBy, FBz, FEtot]         = rusanov_(ulb, urb, flb, frb, Sb);

            return PerIndex{Frho, FrhoVx, FrhoVy, FrhoVz, FBx, FBy, FBz, FEtot};
>>>>>>> a29ca475 (starting refactor)
        }
        else
        {
            auto const [Frho, FrhoVx, FrhoVy, FrhoVz, FBx, FBy, FBz, FEtot]
<<<<<<< HEAD
                = rusanov_(uL.as_tuple(), uR.as_tuple(), fL.as_tuple(), fR.as_tuple(), hydro_speed);

            return PerIndex{Frho, {FrhoVx, FrhoVy, FrhoVz}, {FBx, FBy, FBz}, FEtot};
=======
                = rusanov_(uL(), uR(), fL(), fR(), std::get<0>(s));

            return PerIndex{Frho, FrhoVx, FrhoVy, FrhoVz, FBx, FBy, FBz, FEtot};
>>>>>>> a29ca475 (starting refactor)
        }
    }

private:
    GridLayout layout_;
    double const gamma_;

    template<auto direction>
    auto rusanov_speeds_(auto const& uL, auto const& uR) const
    {
<<<<<<< HEAD
        auto const BdotBL = uL.B.x * uL.B.x + uL.B.y * uL.B.y + uL.B.z * uL.B.z;
        auto const BdotBR = uR.B.x * uR.B.x + uR.B.y * uR.B.y + uR.B.z * uR.B.z;
=======
        auto const BdotBL = uL.Bx * uL.Bx + uL.By * uL.By + uL.Bz * uL.Bz;
        auto const BdotBR = uR.Bx * uR.Bx + uR.By * uR.By + uR.Bz * uR.Bz;
>>>>>>> a29ca475 (starting refactor)

        auto compute_speeds = [&](auto rhoL, auto rhoR, auto PL, auto PR, auto BdotBL, auto BdotBR,
                                  auto VcompL, auto VcompR, auto BcompL, auto BcompR) {
            auto cfastL = compute_fast_magnetosonic_(gamma_, uL.rho, BcompL, BdotBL, uL.P);
            auto cfastR = compute_fast_magnetosonic_(gamma_, uR.rho, BcompR, BdotBR, uR.P);
            auto S      = std::max(std::abs(VcompL) + cfastL, std::abs(VcompR) + cfastR);
            auto Sb     = S;

            if constexpr (Hall)
            {
                auto cwL = compute_whistler_(layout_.inverseMeshSize(direction), uL.rho, BdotBL);
                auto cwR = compute_whistler_(layout_.inverseMeshSize(direction), uR.rho, BdotBR);
                Sb = std::max(std::abs(VcompL) + cfastL + cwL, std::abs(VcompR) + cfastR + cwR);
            }

<<<<<<< HEAD
            return std::make_pair(S, Sb);
        };

        if constexpr (direction == Direction::X)
            return compute_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.V.x, uR.V.x,
                                  uL.B.x, uR.B.x);
        else if constexpr (direction == Direction::Y)
            return compute_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.V.y, uR.V.y,
                                  uL.B.y, uR.B.y);
        else if constexpr (direction == Direction::Z)
            return compute_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.V.z, uR.V.z,
                                  uL.B.z, uR.B.z);
=======
            return std::make_tuple(S, Sb);
        };

        if constexpr (direction == Direction::X)
            return compute_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.Vx, uR.Vx, uL.Bx,
                                  uR.Bx);
        else if constexpr (direction == Direction::Y)
            return compute_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.Vy, uR.Vy, uL.By,
                                  uR.By);
        else if constexpr (direction == Direction::Z)
            return compute_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.Vz, uR.Vz, uL.Bz,
                                  uR.Bz);
>>>>>>> a29ca475 (starting refactor)
    }

    auto rusanov_(auto const& uL, auto const& uR, auto const& fL, auto const& fR,
                  auto const S) const
    {
        auto constexpr N_elements = std::tuple_size_v<std::decay_t<decltype(uL)>>;

        return for_N<N_elements, for_N_R_mode::make_tuple>([&](auto i) {
            return (std::get<i>(fL) + std::get<i>(fR)) * 0.5
                   - S * (std::get<i>(uR) - std::get<i>(uL)) * 0.5;
        });
    }
};
} // namespace PHARE::core

#endif
