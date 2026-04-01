#ifndef CORE_NUMERICS_RIEMANN_SOLVERS_HLL_HPP
#define CORE_NUMERICS_RIEMANN_SOLVERS_HLL_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/riemann_solvers/mhd_speeds.hpp"
#include <cstdlib>

namespace PHARE::core
{
template<bool Hall>
class HLL
{
public:
    HLL(double const gamma)
        : gamma_{gamma}
    {
    }

    template<auto direction>
    auto solve(auto& uL, auto& uR, auto const& fL, auto const& fR)
    {
        auto const [SL, SR] = hll_speeds_<direction>(uL, uR);
        auto uct             = uct_coefs_(uL, uR, SL, SR);

        uL.to_conservative(gamma_);
        uR.to_conservative(gamma_);

        auto const [Frho, FrhoVx, FrhoVy, FrhoVz, FBx, FBy, FBz, FEtot]
            = hll_(uL.as_tuple(), uR.as_tuple(), fL.as_tuple(), fR.as_tuple(), SL, SR);

        return std::make_pair(PerIndex{Frho, {FrhoVx, FrhoVy, FrhoVz}, {FBx, FBy, FBz}, FEtot},
                              uct);
    }

    template<auto direction>
    auto solve(auto& uL, auto& uR, auto const& fL, auto const& fR, auto const& jL, auto const& jR)
    {
        auto const [SL, SR, SLb, SRb] = hll_speeds_<direction>(uL, uR, jL, jR);
        auto uct                        = uct_coefs_(uL, uR, jL, jR, SLb, SRb);

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
            = hll_(uLhydro, uRhydro, fLhydro, fRhydro, SL, SR);
        auto [FBx, FBy, FBz, FEtot] = hll_(uLmag, uRmag, fLmag, fRmag, SLb, SRb);

        return std::make_pair(PerIndex{Frho, {FrhoVx, FrhoVy, FrhoVz}, {FBx, FBy, FBz}, FEtot},
                              uct);
    }

    auto riemann_averaging(auto const& L, auto const& R) const
    {
        auto const inv = 1.0 / (SR_ - SL_);
        return (SR_ * L - SL_ * R) * inv;
    }

    // the normal component is actually needed for the 1D riemann solver for E in 1D and 2D. We
    // could have an if constexpr on the dimension there, and have an extra template on the
    // direction.
    auto vector_riemann_averaging(auto const& L, auto const& R) const
    {
        auto const inv = 1.0 / (SR_ - SL_);
        return PerIndexVector<double>{(SR_ * L.x - SL_ * R.x) * inv, (SR_ * L.y - SL_ * R.y) * inv,
                                      (SR_ * L.z - SL_ * R.z) * inv};
    }

private:
    double const gamma_;

    // these are used for the riemann averagings that are always done after speed computation per
    // index. This is needed for the save transverse magnetic field in godunov fluxes for any
    // riemann solver, but maybe not the best interface.
    double SL_;
    double SR_;

    template<auto direction>
    auto hll_speeds_(auto const& uL, auto const& uR)
    {
        auto const BdotBL = uL.B.x * uL.B.x + uL.B.y * uL.B.y + uL.B.z * uL.B.z;
        auto const BdotBR = uR.B.x * uR.B.x + uR.B.y * uR.B.y + uR.B.z * uR.B.z;

        auto compute_speeds = [&](auto rhoL, auto rhoR, auto PL, auto PR, auto BdotBL, auto BdotBR,
                                  auto VcompL, auto VcompR, auto BcompL, auto BcompR) {
            auto cfastL = compute_fast_magnetosonic_(gamma_, uL.rho, BcompL, BdotBL, uL.P);
            auto cfastR = compute_fast_magnetosonic_(gamma_, uR.rho, BcompR, BdotBR, uR.P);
            auto SL     = std::min({VcompL - cfastL, VcompR - cfastR});
            auto SR     = std::max({VcompL + cfastL, VcompR + cfastR});

            return std::make_tuple(SL, SR);
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
    }

    template<auto direction>
    auto hll_speeds_(auto const& uL, auto const& uR, auto const& jL, auto const& jR)
    {
        auto const BdotBL = uL.B.x * uL.B.x + uL.B.y * uL.B.y + uL.B.z * uL.B.z;
        auto const BdotBR = uR.B.x * uR.B.x + uR.B.y * uR.B.y + uR.B.z * uR.B.z;

        auto compute_speeds = [&](auto rhoL, auto rhoR, auto PL, auto PR, auto BdotBL, auto BdotBR,
                                  auto VcompL, auto VcompR, auto BcompL, auto BcompR) {
            auto cfastL = compute_fast_magnetosonic_(gamma_, uL.rho, BcompL, BdotBL, uL.P);
            auto cfastR = compute_fast_magnetosonic_(gamma_, uR.rho, BcompR, BdotBR, uR.P);
            auto SL     = std::min({VcompL - cfastL, VcompR - cfastR});
            auto SR     = std::max({VcompL + cfastL, VcompR + cfastR});

            auto cwL = 0.; // compute_whistler_(layout_.inverseMeshSize(direction), uL.rho, BdotBL);
            auto cwR = 0.; // compute_whistler_(layout_.inverseMeshSize(direction), uR.rho, BdotBR);
            auto SLb = std::min({VcompL - cfastL - cwL, VcompR - cfastR - cwR});
            auto SRb = std::max({VcompL + cfastL + cwL, VcompR + cfastR + cwR});
            return std::make_tuple(SL, SR, SLb, SRb);
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
    }

    auto hll_(auto const& uL, auto const& uR, auto const& fL, auto const& fR, auto const& SL,
              auto const& SR) const
    {
        auto constexpr N_elements = std::tuple_size_v<std::decay_t<decltype(uL)>>;

        auto hll = [&](auto const ul, auto const ur, auto const fl, auto const fr) {
            if (SL > 0.0)
                return fl;
            else if (SR < 0.0)
                return fr;
            else
                return (SR * fl - SL * fr + SL * SR * (ur - ul)) / (SR - SL);
        };

        return for_N<N_elements, for_N_R_mode::make_tuple>([&](auto i) {
            return hll(std::get<i>(uL), std::get<i>(uR), std::get<i>(fL), std::get<i>(fR));
        });
    }

    // this is different from the mignone et al. 2021 formulation, but the same as in the idefix
    // code (Lesur et al. 2023). We could also consider computing the transverse speeds in the uct
    // instead of here (that would allow us to have the exact formulation of Mignone et al. 2021 for
    // vt, at the cost of genericity).
    UCTData uct_coefs_(auto const& uL, auto const& uR, auto const SL, auto const SR)
    {
        SL_ = SL;
        SR_ = SR;

        auto const inv = 1.0 / (SR - SL);

        UCTData uct;
        uct.coefs[0] = SR * inv;
        uct.coefs[1] = -SL * inv;
        uct.coefs[2] = -SR * SL * inv;
        uct.coefs[3] = uct.coefs[2];
        // probably can be optimized as we only need it in the tranverse direction(s)
        uct.vt = vector_riemann_averaging(uL.V, uR.V);
        return uct;
    }

    UCTData uct_coefs_(auto const& uL, auto const& uR, auto const& jL, auto const& jR,
                       auto const SL, auto const SR)
    {
        auto uct = uct_coefs_(uL, uR, SL, SR);
        uct.jt   = vector_riemann_averaging(jL, jR);
        uct.rhot = riemann_averaging(uL.rho, uR.rho);
        return uct;
    }
};
} // namespace PHARE::core

#endif
