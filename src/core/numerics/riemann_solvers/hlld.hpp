#ifndef CORE_NUMERICS_RIEMANN_SOLVERS_HLLD_HPP
#define CORE_NUMERICS_RIEMANN_SOLVERS_HLLD_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/riemann_solvers/mhd_speeds.hpp"
#include <cstdlib>

namespace PHARE::core
{
template<bool Hall>
class HLLD
{
public:
    HLLD(double const gamma)
        : gamma_{gamma}
    {
    }

    template<auto direction>
    auto solve(auto& uL, auto& uR, auto const& fL, auto const& fR)
    {
        auto hlld_speeds = hlld_speeds_<direction>(uL, uR);

        auto const [uL_s, uL_ss, uR_ss, uR_s]
            = hlld_intermediate_states_<direction>(uL, uR, fL, fR, hlld_speeds);

        uL.to_conservative(gamma_);
        uR.to_conservative(gamma_);

        auto const [Frho, FrhoVx, FrhoVy, FrhoVz, FBx, FBy, FBz, FEtot]
            = hlld_(uL.as_tuple(), uL_s.as_tuple(), uL_ss.as_tuple(), uR_ss.as_tuple(),
                    uR_s.as_tuple(), uR.as_tuple(), fL.as_tuple(), fR.as_tuple(), hlld_speeds);

        return PerIndex{Frho, {FrhoVx, FrhoVy, FrhoVz}, {FBx, FBy, FBz}, FEtot};
    }

    template<auto direction>
    auto solve(auto& uL, auto& uR, auto const& fL, auto const& fR, auto const& jL, auto const& jR)
    {
        static_assert(Hall && "HLLD not supported for Hall MHD");
    }

    // using HLL averages here following PLUTO and Idefix implementation
    auto riemann_averaging(auto const& L, auto const& R) const
    {
        auto const inv = 1.0 / (SR_ - SL_);
        return (SR_ * L - SL_ * R) * inv;
    }

    auto vector_riemann_averaging(auto const& L, auto const& R) const
    {
        auto const inv = 1.0 / (SR_ - SL_);
        return PerIndexVector<double>{(SR_ * L.x - SL_ * R.x) * inv, (SR_ * L.y - SL_ * R.y) * inv,
                                      (SR_ * L.z - SL_ * R.z) * inv};
    }

    std::array<double, 4> uct_coefs;
    PerIndexVector<double> vt{std::nan(""), std::nan(""), std::nan("")};

    PerIndexVector<double> jt{std::nan(""), std::nan(""), std::nan("")};
    double rhot{std::nan("")};

private:
    double const gamma_;

    // these are used for the riemann averagings that are always done after speed computation per
    // index. This is needed for the save transverse magnetic field in godunov fluxes for any
    // riemann solver, but maybe not the best interface.
    double SL_;
    double SR_;

    template<auto direction>
    auto hlld_speeds_(auto const& uL, auto const& uR)
    {
        auto const BdotBL = uL.B.x * uL.B.x + uL.B.y * uL.B.y + uL.B.z * uL.B.z;
        auto const BdotBR = uR.B.x * uR.B.x + uR.B.y * uR.B.y + uR.B.z * uR.B.z;

        auto compute_hll_speeds
            = [&](auto rhoL, auto rhoR, auto PL, auto PR, auto BdotBL, auto BdotBR, auto VcompL,
                  auto VcompR, auto BcompL, auto BcompR) {
                  auto cfastL = compute_fast_magnetosonic_(gamma_, uL.rho, BcompL, BdotBL, uL.P);
                  auto cfastR = compute_fast_magnetosonic_(gamma_, uR.rho, BcompR, BdotBR, uR.P);
                  auto SL     = std::min({VcompL - cfastL, VcompR - cfastR});
                  auto SR     = std::max({VcompL + cfastL, VcompR + cfastR});

                  return std::make_tuple(SL, SR);
              };

        auto compute_hlld_speeds
            = [&](auto rhoL, auto rhoR, auto PL, auto PR, auto BdotBL, auto BdotBR, auto VcompL,
                  auto VcompR, auto BcompL, auto BcompR) {
                  auto [SL, SR] = compute_hll_speeds(rhoL, rhoR, PL, PR, BdotBL, BdotBR, VcompL,
                                                     VcompR, BcompL, BcompR);

                  auto PtL = PL + 0.5 * BdotBL;
                  auto PtR = PR + 0.5 * BdotBR;

                  auto SM_numerator
                      = rhoR * VcompR * (SR - VcompR) - rhoL * VcompL * (SL - VcompL) - PtR + PtL;
                  auto SM_denominator = rhoR * (SR - VcompR) - rhoL * (SL - VcompL);
                  auto SM             = SM_numerator / SM_denominator;

                  auto const rhoL_s = rhoL * (SL - VcompL) / (SL - SM);
                  auto const rhoR_s = rhoR * (SR - VcompR) / (SR - SM);

                  auto const SL_s = SM - std::abs(BcompL) / std::sqrt(rhoL_s);
                  auto const SR_s = SM + std::abs(BcompR) / std::sqrt(rhoR_s);

                  auto hlld_speeds = std::make_tuple(SL, SL_s, SM, SR_s, SR);

                  uct_coefs_<direction>(uL, uR, hlld_speeds);


                  return hlld_speeds;
              };


        if constexpr (direction == Direction::X)
            return compute_hlld_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.V.x, uR.V.x,
                                       uL.B.x, uR.B.x);
        else if constexpr (direction == Direction::Y)
            return compute_hlld_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.V.y, uR.V.y,
                                       uL.B.y, uR.B.y);
        else if constexpr (direction == Direction::Z)
            return compute_hlld_speeds(uL.rho, uR.rho, uL.P, uR.P, BdotBL, BdotBR, uL.V.z, uR.V.z,
                                       uL.B.z, uR.B.z);
    }

    // need some optimization, the star states should only be computed if we are in a star region
    template<auto direction>
    auto hlld_intermediate_states_(auto const& uL, auto const& uR, auto const& fL, auto const& fR,
                                   auto& hlld_speeds) const
    {
        auto& [SL, SL_s, SM, SR_s, SR] = hlld_speeds;

        auto constexpr transverse = [&]() {
            if constexpr (direction == Direction::X)
                return std::array{Direction::Y, Direction::Z};
            else if constexpr (direction == Direction::Y)
                return std::array{Direction::X, Direction::Z};
            else if constexpr (direction == Direction::Z)
                return std::array{Direction::X, Direction::Y};
        };

        auto sgn = [](auto const x) { return (x > 0) - (x < 0); };

        auto const etotL
            = eosPToEtot(gamma_, uL.rho, uL.V.x, uL.V.y, uL.V.z, uL.B.x, uL.B.y, uL.B.z, uL.P);
        auto const etotR
            = eosPToEtot(gamma_, uR.rho, uR.V.x, uR.V.y, uR.V.z, uR.B.x, uR.B.y, uR.B.z, uR.P);

        auto compute_tranverse_velocity_s = [&](auto const u, auto const S, auto const tdir) {
            auto const vt_s = u.V(tdir)
                              - u.B(tdir) * u.B(direction) * (SM - u.V(direction))
                                    / (u.rho * (S - u.V(direction)) * (S - SM)
                                       - u.B(direction) * u.B(direction));

            return vt_s;
        };

        auto compute_tranverse_magnetic_s = [&](auto const u, auto const S, auto const tdir) {
            auto const Bt_s
                = u.B(tdir)
                  * (u.rho * (S - u.V(direction)) * (S - u.V(direction))
                     - u.B(direction) * u.B(direction))
                  / (u.rho * (S - u.V(direction)) * (S - SM) - u.B(direction) * u.B(direction));

            return Bt_s;
        };

        auto compute_tranverse_magnetic_s_hllc = [&](auto const tdir) {
            auto const Bhll
                = (SR * uR.B(tdir) - SL * uL.B(tdir) + fL.B(tdir) - fR.B(tdir)) / (SR - SL);

            return Bhll;
        };

        auto hlld_intermediate_states = [&](auto const fl, auto const fr) {
            auto const rhoL_s = uL.rho * (SL - uL.V(direction)) / (SL - SM);
            auto const rhoR_s = uR.rho * (SR - uR.V(direction)) / (SR - SM);

            auto const vn = SM;
            // this should probably not be reconstructed in the normal direction as B is already
            // face centered there
            // auto const Bn = uL.B(direction); // should be the same on both sides
            auto const Bn = compute_tranverse_magnetic_s_hllc(direction);

            auto const vt0L_s = compute_tranverse_velocity_s(uL, SL, transverse()[0]);
            auto const vt1L_s = compute_tranverse_velocity_s(uL, SL, transverse()[1]);
            auto const vt0R_s = compute_tranverse_velocity_s(uR, SR, transverse()[0]);
            auto const vt1R_s = compute_tranverse_velocity_s(uR, SR, transverse()[1]);

            auto Bt0L_s = compute_tranverse_magnetic_s(uL, SL, transverse()[0]);
            auto Bt1L_s = compute_tranverse_magnetic_s(uL, SL, transverse()[1]);
            auto Bt0R_s = compute_tranverse_magnetic_s(uR, SR, transverse()[0]);
            auto Bt1R_s = compute_tranverse_magnetic_s(uR, SR, transverse()[1]);

            bool const hllc_fallback
                = (SL_s - SL) < 1e-4 * (SM - SL) || (SR_s - SR) > -1e-4 * (SR - SM);

            // Fallback to HLLC as in idefix/PLUTO (Maybe Bn should also use HLL averaging?)
            if (hllc_fallback)
            {
                Bt0L_s = compute_tranverse_magnetic_s_hllc(transverse()[0]);
                Bt1L_s = compute_tranverse_magnetic_s_hllc(transverse()[1]);
                Bt0R_s = Bt0L_s;
                Bt1R_s = Bt1L_s;

                SL_s = SR_s = SM;
            }

            auto const p_tot_L = uL.P + 0.5 * (uL.B.x * uL.B.x + uL.B.y * uL.B.y + uL.B.z * uL.B.z);
            auto const p_tot_R = uR.P + 0.5 * (uR.B.x * uR.B.x + uR.B.y * uR.B.y + uR.B.z * uR.B.z);

            auto const p_tot_s
                = ((SR - uR.V(direction)) * uR.rho * p_tot_L
                   - (SL - uL.V(direction)) * uL.rho * p_tot_R
                   + uL.rho * uR.rho * (SL - uL.V(direction)) * (SR - uR.V(direction))
                         * (uR.V(direction) - uL.V(direction)))
                  / (uR.rho * (SR - uR.V(direction)) - uL.rho * (SL - uL.V(direction)));

            auto const EtotL_s
                = ((SL - uL.V(direction)) * etotL - p_tot_L * uL.V(direction) + p_tot_s * SM
                   + Bn
                         * (uL.B.x * uL.V.x + uL.B.y * uL.V.y + uL.B.z * uL.V.z
                            - (Bn * vn + vt0L_s * Bt0L_s + vt1L_s * Bt1L_s)))
                  / (SL - SM);
            auto const EtotR_s
                = ((SR - uR.V(direction)) * etotR - p_tot_R * uR.V(direction) + p_tot_s * SM
                   + Bn
                         * (uR.B.x * uR.V.x + uR.B.y * uR.V.y + uR.B.z * uR.V.z
                            - (Bn * vn + vt0R_s * Bt0R_s + vt1R_s * Bt1R_s)))
                  / (SR - SM);


            auto const vt0_ss = (std::sqrt(rhoL_s) * vt0L_s + std::sqrt(rhoR_s) * vt0R_s
                                 + (Bt0R_s - Bt0L_s) * sgn(Bn))
                                / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));
            auto const vt1_ss = (std::sqrt(rhoL_s) * vt1L_s + std::sqrt(rhoR_s) * vt1R_s
                                 + (Bt1R_s - Bt1L_s) * sgn(Bn))
                                / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));

            auto const Bt0_ss = (std::sqrt(rhoL_s) * Bt0R_s + std::sqrt(rhoR_s) * Bt0L_s
                                 + std::sqrt(rhoL_s * rhoR_s) * (vt0R_s - vt0L_s) * sgn(Bn))
                                / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));

            auto const Bt1_ss = (std::sqrt(rhoL_s) * Bt1R_s + std::sqrt(rhoR_s) * Bt1L_s
                                 + std::sqrt(rhoL_s * rhoR_s) * (vt1R_s - vt1L_s) * sgn(Bn))
                                / (std::sqrt(rhoL_s) + std::sqrt(rhoR_s));

            auto const EtotL_ss = EtotL_s
                                  - std::sqrt(rhoL_s)
                                        * (vn * Bn + vt0L_s * Bt0L_s + vt1L_s * Bt1L_s
                                           - (vn * Bn + vt0_ss * Bt0_ss + vt1_ss * Bt1_ss))
                                        * sgn(Bn);
            auto const EtotR_ss = EtotR_s
                                  + std::sqrt(rhoR_s)
                                        * (vn * Bn + vt0R_s * Bt0R_s + vt1R_s * Bt1R_s
                                           - (vn * Bn + vt0_ss * Bt0_ss + vt1_ss * Bt1_ss))
                                        * sgn(Bn);

            if constexpr (direction == Direction::X)
            {
                auto const uL_s = PerIndex{rhoL_s,
                                           {rhoL_s * vn, rhoL_s * vt0L_s, rhoL_s * vt1L_s},
                                           {Bn, Bt0L_s, Bt1L_s},
                                           EtotL_s};
                auto const uR_s = PerIndex{rhoR_s,
                                           {rhoR_s * vn, rhoR_s * vt0R_s, rhoR_s * vt1R_s},
                                           {Bn, Bt0R_s, Bt1R_s},
                                           EtotR_s};

                auto const uL_ss = PerIndex{rhoL_s,
                                            {rhoL_s * vn, rhoL_s * vt0_ss, rhoL_s * vt1_ss},
                                            {Bn, Bt0_ss, Bt1_ss},
                                            EtotL_ss};
                auto const uR_ss = PerIndex{rhoR_s,
                                            {rhoR_s * vn, rhoR_s * vt0_ss, rhoR_s * vt1_ss},
                                            {Bn, Bt0_ss, Bt1_ss},
                                            EtotR_ss};

                return std::make_tuple(uL_s, uL_ss, uR_ss, uR_s);
            }
            else if constexpr (direction == Direction::Y)
            {
                auto const uL_s = PerIndex{rhoL_s,
                                           {rhoL_s * vt0L_s, rhoL_s * vn, rhoL_s * vt1L_s},
                                           {Bt0L_s, Bn, Bt1L_s},
                                           EtotL_s};
                auto const uR_s = PerIndex{rhoR_s,
                                           {rhoR_s * vt0R_s, rhoR_s * vn, rhoR_s * vt1R_s},
                                           {Bt0R_s, Bn, Bt1R_s},
                                           EtotR_s};

                auto const uL_ss = PerIndex{rhoL_s,
                                            {rhoL_s * vt0_ss, rhoL_s * vn, rhoL_s * vt1_ss},
                                            {Bt0_ss, Bn, Bt1_ss},
                                            EtotL_ss};
                auto const uR_ss = PerIndex{rhoR_s,
                                            {rhoR_s * vt0_ss, rhoR_s * vn, rhoR_s * vt1_ss},
                                            {Bt0_ss, Bn, Bt1_ss},
                                            EtotR_ss};

                return std::make_tuple(uL_s, uL_ss, uR_ss, uR_s);
            }
            else // direction == Direction::Z
            {
                auto const uL_s = PerIndex{rhoL_s,
                                           {rhoL_s * vt0L_s, rhoL_s * vt1L_s, rhoL_s * vn},
                                           {Bt0L_s, Bt1L_s, Bn},
                                           EtotL_s};
                auto const uR_s = PerIndex{rhoR_s,
                                           {rhoR_s * vt0R_s, rhoR_s * vt1R_s, rhoR_s * vn},
                                           {Bt0R_s, Bt1R_s, Bn},
                                           EtotR_s};

                auto const uL_ss = PerIndex{rhoL_s,
                                            {rhoL_s * vt0_ss, rhoL_s * vt1_ss, rhoL_s * vn},
                                            {Bt0_ss, Bt1_ss, Bn},
                                            EtotL_ss};
                auto const uR_ss = PerIndex{rhoR_s,
                                            {rhoR_s * vt0_ss, rhoR_s * vt1_ss, rhoR_s * vn},
                                            {Bt0_ss, Bt1_ss, Bn},
                                            EtotR_ss};

                return std::make_tuple(uL_s, uL_ss, uR_ss, uR_s);
            }
        };

        return hlld_intermediate_states(fL, fR);
    }

    template<auto direction>
    auto hlld_speeds_(auto const& uL, auto const& uR, auto const& jL, auto const& jR)
    {
        static_assert(Hall && "HLLD not supported for Hall MHD");
    }

    auto hlld_(auto const uL, auto const uL_s, auto const uL_ss, auto const uR_ss, auto const uR_s,
               auto const uR, auto const fL, auto const fR, auto const hlld_speeds) const
    {
        auto const [SL, SL_s, SM, SR_s, SR] = hlld_speeds;

        auto constexpr N_elements = std::tuple_size_v<std::decay_t<decltype(fL)>>;

        auto const hlld = [&](auto const ul, auto const ul_s, auto const ul_ss, auto const ur_ss,
                              auto const ur_s, auto const ur, auto const fl, auto const fr) {
            if (SL > 0.0) // L
                return fl;
            else if (SR < 0.0) // R
                return fr;
            else if (SL_s >= 0) // L*
                return fl + SL * ul_s - SL * ul;
            else if (SR_s <= 0) // R*
                return fr + SR * ur_s - SR * ur;
            else if (SM >= 0) // L**
                return fl + SL_s * ul_ss - (SL_s - SL) * ul_s - SL * ul;
            else // R**
                return fr + SR_s * ur_ss - (SR_s - SR) * ur_s - SR * ur;
        };

        return for_N<N_elements, for_N_R_mode::make_tuple>([&](auto i) {
            return hlld(std::get<i>(uL), std::get<i>(uL_s), std::get<i>(uL_ss), std::get<i>(uR_ss),
                        std::get<i>(uR_s), std::get<i>(uR), std::get<i>(fL), std::get<i>(fR));
        });
    }

    // in the hllc fallback used in idefix/pluto, they use hll averages for the magnetic field. we
    // do the same here for consistency.
    template<auto direction>
    void uct_coefs_(auto const& uL, auto const& uR, auto const& hlld_speeds)
    {
        auto const [SL, SL_s, SM, SR_s, SR] = hlld_speeds;

        SL_ = SL;
        SR_ = SR;

        auto const fallback = ((SL_s - SL) < 1e-4 * (SM - SL)) || ((SR_s - SR) > -1e-4 * (SR - SM));

        if (fallback)
        {
            auto const sl = std::min(0.0, SL);
            auto const sr = std::max(0.0, SR);

            auto const inv = 1.0 / (sr - sl);

            uct_coefs[0] = sr * inv;
            uct_coefs[1] = -sl * inv;
            uct_coefs[2] = -sr * sl * inv;
            uct_coefs[3] = uct_coefs[2];
        }
        else
        {
            auto const nuL = (SL_s + SL) / (std::abs(SL_s) + std::abs(SL));
            auto const nuR = (SR + SR_s) / (std::abs(SR) + std::abs(SR_s));

            auto const nu_s = std::abs(SR_s - SL_s) > 1e-9 * std::abs(SR - SL)
                                  ? (SR_s + SL_s) / (std::abs(SR_s) + std::abs(SL_s))
                                  : 0.0;

            uct_coefs[0] = (1 + nu_s) * 0.5;
            uct_coefs[1] = (1 - nu_s) * 0.5;

            auto const xiL = ((uL.V(direction) - SM) * (SL - SM)) / (SL_s + SL - 2. * SM);
            auto const xiR = ((uR.V(direction) - SM) * (SR - SM)) / (SR_s + SR - 2. * SM);

            uct_coefs[2] = 0.5 * (nuL - nu_s) * xiL + 0.5 * (std::abs(SL_s) - nu_s * SL_s);
            uct_coefs[3] = 0.5 * (nuR - nu_s) * xiR + 0.5 * (std::abs(SR_s) - nu_s * SR_s);
        }

        vt = vector_riemann_averaging(uL.V, uR.V);
    }

    void uct_coefs_(auto const& uL, auto const& uR, auto const& jL, auto const& jR, auto const SL,
                    auto const SR)
    {
        static_assert(Hall && "HLLD not supported for Hall MHD");
    }
};
} // namespace PHARE::core

#endif
