#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP

#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include <cstddef>
#include <tuple>

namespace PHARE::core
{
template<typename GridLayout>
class GodunovFluxes : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

public:
    template<typename Field, typename VecField, typename... Fluxes>
    void operator()(Field const& rho, VecField const& V, VecField const& B_CT, Field const& P,
                    VecField const& J, Fluxes&... fluxes)
    {
        if (!this->hasLayout())
            throw std::runtime_error("Error - Reconstruction - GridLayout not set, cannot proceed "
                                     "to reconstruction");

        if constexpr (dimension == 1)
        {
            auto& [rho_x, rhoV_x, B_x, Etot_x] = std::forward_as_tuple(fluxes...);

            layout_->evalOnBox(rho_x, [&](auto&... args) mutable {
                this->template godunov_fluxes_<Direction::X>(rho, V, B_CT, P, J, rho_x, rhoV_x, B_x,
                                                             Etot_x, {args...});
            });
        }
        if constexpr (dimension == 2)
        {
            auto& [rho_x, rhoV_x, B_x, Etot_x, rho_y, rhoV_y, B_y, Etot_y]
                = std::forward_as_tuple(fluxes...);

            layout_->evalOnBox(rho_x, [&](auto&... args) mutable {
                this->template godunov_fluxes_<Direction::X>(rho, V, B_CT, P, J, rho_x, rhoV_x, B_x,
                                                             Etot_x, {args...});
            });

            layout_->evalOnBox(rho_y, [&](auto&... args) mutable {
                this->template godunov_fluxes_<Direction::Y>(rho, V, B_CT, P, J, rho_y, rhoV_y, B_y,
                                                             Etot_y, {args...});
            });
        }
        if constexpr (dimension == 3)
        {
            auto& [rho_x, rhoV_x, B_x, Etot_x, rho_y, rhoV_y, B_y, Etot_y, rho_z, rhoV_z, B_z,
                   Etot_z]
                = std::forward_as_tuple(fluxes...);

            layout_->evalOnBox(rho_x, [&](auto&... args) mutable {
                this->template godunov_fluxes_<Direction::X>(rho, V, B_CT, P, J, rho_x, rhoV_x, B_x,
                                                             Etot_x, {args...});
            });

            layout_->evalOnBox(rho_y, [&](auto&... args) mutable {
                this->template godunov_fluxes_<Direction::Y>(rho, V, B_CT, P, J, rho_y, rhoV_y, B_y,
                                                             Etot_y, {args...});
            });

            layout_->evalOnBox(rho_z, [&](auto&... args) mutable {
                this->template godunov_fluxes_<Direction::Z>(rho, V, B_CT, P, J, rho_z, rhoV_z, B_z,
                                                             Etot_z, {args...});
            });
        }
    }

private:
    double const gamma_;
    double const eta_;
    double const nu_;

    template<auto direction, typename Field, typename VecField>
    void godunov_fluxes_(Field const& rho, VecField const& V, VecField const& B_CT, Field const& P,
                         VecField const& J, Field& rho_flux, VecField& rhoV_flux, VecField& B_flux,
                         Field& Etot_flux, MeshIndex<Field::dimension> index) const
    {
        auto const& [Vx, Vy, Vz] = V();
        auto const& [Bx, By, Bz] = B_CT();
        auto const& [Jx, Jy, Jz] = J();

        auto const& Quantities = std::forward_as_tuple(rho, Vx, Vy, Vz, Bx, By, Bz, P, Jx, Jy, Jz);

        // Left and Right state reconstructions
        auto [rhoL, VxL, VyL, VzL, BxL, ByL, BzL, PL, JxL, JyL, JzL] = std::apply(
            [&](auto const&... fields) {
                return std::make_tuple(reconstruct_uL_<direction>(fields, index)...);
            },
            Quantities);

        auto [rhoR, VxR, VyR, VzR, BxR, ByR, BzR, PR, JxR, JyR, JzR] = std::apply(
            [&](auto const&... fields) {
                return std::make_tuple(reconstruct_uR_<direction>(fields, index)...);
            },
            Quantities);

        // Compute ideal flux vector for Left and Right states
        auto [F_rhoL, F_rhoVxL, F_rhoVyL, F_rhoVzL, F_BxL, F_ByL, F_BzL, F_EtotL]
            = ideal_flux_vector_<direction>(rhoL, VxL, VyL, VzL, BxL, ByL, BzL, PL);

        auto [F_rhoR, F_rhoVxR, F_rhoVyR, F_rhoVzR, F_BxR, F_ByR, F_BzR, F_EtotR]
            = ideal_flux_vector_<direction>(rhoR, VxR, VyR, VzR, BxR, ByR, BzR, PR);

        // Non ideal contributions
        hall_contribution_<direction>(rhoL, BxL, ByL, BzL, JxL, JyL, JzL, F_BxL, F_ByL, F_BzL,
                                      F_EtotL);

        hall_contribution_<direction>(rhoR, BxR, ByR, BzR, JxR, JyR, JzR, F_BxR, F_ByR, F_BzR,
                                      F_EtotR);

        resistive_contributions_<direction>(eta_, JxL, JyL, JzL, BxL, ByL, BzL, F_BxL, F_ByL, F_BzL,
                                            F_EtotL);

        resistive_contributions_<direction>(eta_, JxR, JyR, JzR, BxR, ByR, BzR, F_BxR, F_ByR, F_BzR,
                                            F_EtotR);

        auto [LaplJxL, LaplJxR] = reconstructed_lapacian(Jx, index);
        auto [LaplJyL, LaplJyR] = reconstructed_lapacian(Jy, index);
        auto [LaplJzL, LaplJzR] = reconstructed_lapacian(Jz, index);

        resistive_contributions_<direction>(nu_, LaplJxL, LaplJyL, LaplJzL, BxL, ByL, BzL, F_BxL,
                                            F_ByL, F_BzL, F_EtotL);

        resistive_contributions_<direction>(nu_, LaplJxR, LaplJyR, LaplJzR, BxR, ByR, BzR, F_BxR,
                                            F_ByR, F_BzR, F_EtotR);

        auto uL = std::forward_as_tuple(rhoL, VxL, VyL, VzL, BxL, ByL, BzL, PL);
        auto uR = std::forward_as_tuple(rhoR, VxR, VyR, VzR, BxR, ByR, BzR, PR);
        auto fL = std::forward_as_tuple(F_rhoL, F_rhoVxL, F_rhoVyL, F_rhoVzL, F_BxL, F_ByL, F_BzL,
                                        F_EtotL);
        auto fR = std::forward_as_tuple(F_rhoR, F_rhoVxR, F_rhoVyR, F_rhoVzR, F_BxR, F_ByR, F_BzR,
                                        F_EtotR);

        auto [rho_, rhoVx_, rhoVy_, rhoVz_, Bx_, By_, Bz_, Etot_] = riemann_solver_(uL, uR, fL, fR);

        rho_flux                = rho_;
        rhoV_flux(Component::X) = rhoVx_;
        rhoV_flux(Component::Y) = rhoVy_;
        rhoV_flux(Component::Z) = rhoVz_;
        B_flux(Component::X)    = Bx_;
        B_flux(Component::Y)    = By_;
        B_flux(Component::Z)    = Bz_;
        Etot_flux               = Etot_;
    }

    template<auto direction>
    auto riemann_solver_(auto const& uL, auto const& uR, auto const& fL, auto const& fR) const
    {
        auto const& [rhoL, VxL, VyL, VzL, BxL, ByL, BzL, PL]                             = uL;
        auto const& [rhoR, VxR, VyR, VzR, BxR, ByR, BzR, PR]                             = uR;
        auto const& [F_rhoL, F_rhoVxL, F_rhoVyL, F_rhoVzL, F_BxL, F_ByL, F_BzL, F_EtotL] = fL;
        auto const& [F_rhoR, F_rhoVxR, F_rhoVyR, F_rhoVzR, F_BxR, F_ByR, F_BzR, F_EtotR] = fR;

        // Convert to conserved variables
        auto rhoVxL = rhoL * VxL;
        auto rhoVyL = rhoL * VyL;
        auto rhoVzL = rhoL * VzL;
        auto EtotL  = PL / (gamma_ - 1) + 0.5 * rhoL * (VxL * VxL + VyL * VyL + VzL * VzL)
                     + 0.5 * (BxL * BxL + ByL * ByL + BzL * BzL);

        auto rhoVxR = rhoR * VxR;
        auto rhoVyR = rhoR * VyR;
        auto rhoVzR = rhoR * VzR;
        auto EtotR  = PR / (gamma_ - 1) + 0.5 * rhoR * (VxR * VxR + VyR * VyR + VzR * VzR)
                     + 0.5 * (BxR * BxR + ByR * ByR + BzR * BzR);

        auto uL_ = std::forward_as_tuple(rhoL, rhoVxL, rhoVyL, rhoVzL);
        auto uR_ = std::forward_as_tuple(rhoR, rhoVxR, rhoVyR, rhoVzR);
        auto fL_ = std::forward_as_tuple(F_rhoL, F_rhoVxL, F_rhoVyL, F_rhoVzL);
        auto fR_ = std::forward_as_tuple(F_rhoR, F_rhoVxR, F_rhoVyR, F_rhoVzR);

        auto ubL = std::forward_as_tuple(BxL, ByL, BzL, EtotL);
        auto ubR = std::forward_as_tuple(BxR, ByR, BzR, EtotR);
        auto fbL = std::forward_as_tuple(F_BxL, F_ByL, F_BzL, F_EtotL);
        auto fbR = std::forward_as_tuple(F_BxR, F_ByR, F_BzR, F_EtotR);

        // for rusanov riemann solver
        auto const [S, Sb]                  = rusanov_speeds_<direction>(uL, uR);
        auto [rho_, rhoVx_, rhoVy_, rhoVz_] = rusanov_(uL_, uR_, fL_, fR_, S);
        auto [Bx_, By_, Bz_, Etot_]         = rusanov_(ubL, ubR, fbL, fbR, Sb);

        return std::make_tuple(rho_, rhoVx_, rhoVy_, rhoVz_, Bx_, By_, Bz_, Etot_);
    }

    template<auto direction>
    auto rusanov_speeds_(auto const& uL, auto const& uR) const
    {
        auto [rhoL, VxL, VyL, VzL, BxL, ByL, BzL, PL] = uL;
        auto [rhoR, VxR, VyR, VzR, BxR, ByR, BzR, PR] = uR;

        auto BdotBL = BxL * BxL + ByL * ByL + BzL * BzL;
        auto BdotBR = BxR * BxR + ByR * ByR + BzR * BzR;

        auto compute_speeds = [&](auto Bcomp, auto Vcomp, auto dirIdx) {
            auto cfastL = compute_fast_magnetosonic_(rhoL, Bcomp, BdotBL, PL);
            auto cfastR = compute_fast_magnetosonic_(rhoR, Bcomp, BdotBR, PR);
            auto cwL    = compute_whistler_(layout_->inverseMeshSize_[dirIdx], rhoL, BdotBL);
            auto cwR    = compute_whistler_(layout_->inverseMeshSize_[dirIdx], rhoR, BdotBR);
            auto S      = std::max(std::abs(Vcomp) + cfastL, std::abs(Vcomp) + cfastR);
            auto Sb     = std::max(std::abs(Vcomp) + cfastL + cwL, std::abs(Vcomp) + cfastR + cwR);
            return std::make_tuple(S, Sb);
        };

        if constexpr (direction == Direction::X)
            return compute_speeds(BxL, VxL, dirX);
        else if constexpr (direction == Direction::Y)
            return compute_speeds(ByL, VyL, dirY);
        else if constexpr (direction == Direction::Z)
            return compute_speeds(BzL, VzL, dirZ);
    }

    auto rusanov_(auto const& uL, auto const& uR, auto const& fL, auto const& fR,
                  auto const S) const
    {
        // to be used 2 times in hall mhd (the second time for B with whisler contribution).
        return std::apply(
            [&](auto... i) {
                return std::make_tuple(((std::get<i>(fL) + std::get<i>(fR)) * 0.5
                                        - S * (std::get<i>(uR) - std::get<i>(uL)) * 0.5)...);
            },
            std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(uL)>>>{});
    }

    auto compute_fast_magnetosonic_(auto const& rho, auto const& B, auto const& BdotB,
                                    auto const& P) const
    {
        auto Sound     = std::sqrt((gamma_ * P) / rho);
        auto AlfvenDir = std::sqrt(B * B / rho); // diectionnal alfven
        auto Alfven    = std::sqrt(BdotB / rho);

        auto c02    = Sound * Sound;
        auto cA2    = Alfven * Alfven;
        auto cAdir2 = AlfvenDir * AlfvenDir;

        return std::sqrt((c02 + cA2) * 0.5
                         + std::sqrt((c02 + cA2) * (c02 + cA2) - 4.0 * c02 * cAdir2));
    }

    auto compute_whistler_(auto const& invMeshSize, auto const& rho, auto const& BdotB) const
    {
        auto vw = std::sqrt(1 + 0.25 * invMeshSize * invMeshSize) + 0.5 * invMeshSize;
        return std::sqrt(BdotB) * vw / rho;
    }

    template<auto direction>
    auto ideal_flux_vector_(auto const& rho, auto const& Vx, auto const& Vy, auto const& Vz,
                            auto const& Bx, auto const& By, auto const& Bz, auto const& P) const
    {
        auto GeneralisedPressure = P + 0.5 * (Bx * Bx + By * By + Bz * Bz);
        auto TotalEnergy         = P / (gamma_ - 1) + 0.5 * rho * (Vx * Vx + Vy * Vy + Vz * Vz)
                           + 0.5 * (Bx * Bx + By * By + Bz * Bz);
        if constexpr (direction == Direction::X)
        {
            auto F_rho   = rho * Vx;
            auto F_rhoVx = rho * Vx * Vx + GeneralisedPressure + Bx * Bx;
            auto F_rhoVy = rho * Vx * Vy + Bx * By;
            auto F_rhoVz = rho * Vx * Vz + Bx * Bz;
            auto F_Bx    = 0.0;
            auto F_By    = By * Vx - Vy * Bx;
            auto F_Bz    = Bz * Vx - Vz * Bx;
            auto F_Etot
                = (TotalEnergy + GeneralisedPressure) * Vx - Bx * (Vx * Bx + Vy * By + Vz * Bz);

            return std::make_tuple(F_rho, F_rhoVx, F_rhoVy, F_rhoVz, F_Bx, F_By, F_Bz, F_Etot);
        }
        if constexpr (direction == Direction::Y)
        {
            auto F_rho   = rho * Vy;
            auto F_rhoVx = rho * Vy * Vx + By * Bx;
            auto F_rhoVy = rho * Vy * Vy + GeneralisedPressure + By * By;
            auto F_rhoVz = rho * Vy * Vz + By * Bz;
            auto F_Bx    = Bx * Vy - Vx * By;
            auto F_By    = 0.0;
            auto F_Bz    = Bz * Vy - Vz * By;
            auto F_Etot
                = (TotalEnergy + GeneralisedPressure) * Vy - By * (Vx * Bx + Vy * By + Vz * Bz);

            return std::make_tuple(F_rho, F_rhoVx, F_rhoVy, F_rhoVz, F_Bx, F_By, F_Bz, F_Etot);
        }
        if constexpr (direction == Direction::Z)
        {
            auto F_rho   = rho * Vz;
            auto F_rhoVx = rho * Vz * Vx + Bz * Bx;
            auto F_rhoVy = rho * Vz * Vy + Bz * By;
            auto F_rhoVz = rho * Vz * Vz + GeneralisedPressure + Bz * Bz;
            auto F_Bx    = Bx * Vz - Vx * Bz;
            auto F_By    = By * Vz - Vy * Bz;
            auto F_Bz    = 0.0;
            auto F_Etot
                = (TotalEnergy + GeneralisedPressure) * Vz - Bz * (Vx * Bx + Vy * By + Vz * Bz);

            return std::make_tuple(F_rho, F_rhoVx, F_rhoVy, F_rhoVz, F_Bx, F_By, F_Bz, F_Etot);
        }
    }

    template<auto direction>
    void hall_contribution_(auto const& rho, auto const& Bx, auto const& By, auto const& Bz,
                            auto const& Jx, auto const& Jy, auto const& Jz, auto& F_Bx, auto& F_By,
                            auto& F_Bz, auto& F_Etot) const
    {
        auto invRho = 1.0 / rho;

        auto JxB_x = Jy * Bz - Jz * By;
        auto JxB_y = Jz * Bx - Jx * Bz;
        auto JxB_z = Jx * By - Jy * Bx;

        auto BdotJ = Bx * Jx + By * Jy + Bz * Jz;
        auto BdotB = Bx * Bx + By * By + Bz * Bz;

        if constexpr (direction == Direction::X)
        {
            F_By += -JxB_z * invRho;
            F_Bz += JxB_y * invRho;
            F_Etot += (BdotJ * Bx - BdotB * Jx) * invRho;
        }
        if constexpr (direction == Direction::Y)
        {
            F_Bx += JxB_z * invRho;
            F_Bz += -JxB_x * invRho;
            F_Etot += (BdotJ * By - BdotB * Jy) * invRho;
        }
        if constexpr (direction == Direction::Z)
        {
            F_Bx += -JxB_y * invRho;
            F_By += JxB_x * invRho;
            F_Etot += (BdotJ * Bz - BdotB * Jz) * invRho;
        }
    }

    template<auto direction>
    void resistive_contributions_(auto const& pc, auto const& Jx, auto const& Jy, auto const& Jz,
                                  auto const& Bx, auto const& By, auto const& Bz, auto& F_Bx,
                                  auto& F_By, auto& F_Bz, auto& F_Etot) const
    // Can be used for both resistivity with J and eta and hyper resistivity with laplJ and nu
    {
        if constexpr (direction == Direction::X)
        {
            F_By += -Jz * pc;
            F_Bz += Jy * pc;
            F_Etot += (Jy * Bz - Jz * By) * pc;
        }
        if constexpr (direction == Direction::Y)
        {
            F_Bx += Jz * pc;
            F_Bz += -Jx * pc;
            F_Etot += (Jz * Bx - Jx * Bz) * pc;
        }
        if constexpr (direction == Direction::Y)
        {
            F_Bx += -Jy * pc;
            F_By += Jx * pc;
            F_Etot += (Jx * By - Jy * Bx) * pc;
        }
    }

    template<typename Field>
    auto reconstructed_lapacian(Field const& J, MeshIndex<Field::dimension> index) const
    {
        auto d2 = [&](auto const& dir, auto const& prevValue, auto const& Value,
                      auto const& nextValue) {
            return (layout_->inverseMeshSize_[dir]) * (layout_->inverseMeshSize_[dir])
                   * (prevValue - 2.0 * Value + nextValue);
        };

        auto LR = [&](auto const& index, Direction dir) {
            return std::make_tuple(reconstruct_uL_<dir>(J, index), reconstruct_uR_<dir>(J, index));
        };

        auto [JL, JR] = LR(index, Direction::X);

        if constexpr (dimension == 1)
        {
            MeshIndex<1> prevX = index;
            prevX[0] -= 1;
            MeshIndex<1> nextX = index;
            nextX[0] += 1;

            auto [JL_X_1, JR_X_1] = LR(prevX, Direction::X);
            auto [JL_X1, JR_X1]   = LR(nextX, Direction::X);

            auto LaplJL = d2(dirX, JL_X_1, JL, JL_X1);
            auto LaplJR = d2(dirX, JR_X_1, JR, JR_X1);

            return std::make_tuple(LaplJL, LaplJR);
        }
        if constexpr (dimension == 2)
        {
            MeshIndex<2> prevX = index;
            prevX[0] -= 1;
            MeshIndex<2> nextX = index;
            nextX[0] += 1;

            MeshIndex<2> prevY = index;
            prevY[1] -= 1;
            MeshIndex<2> nextY = index;
            nextY[1] += 1;

            auto [JL_X_1, JR_X_1] = LR(prevX, Direction::X);
            auto [JL_X1, JR_X1]   = LR(nextX, Direction::X);

            auto [JL_Y_1, JR_Y_1] = LR(prevY, Direction::Y);
            auto [JL_Y1, JR_Y1]   = LR(nextY, Direction::Y);

            auto LaplJL = d2(dirX, JL_X_1, JL, JL_X1) + d2(dirY, JL_Y_1, JL, JL_Y1);
            auto LaplJR = d2(dirX, JR_X_1, JR, JR_X1) + d2(dirY, JR_Y_1, JR, JR_Y1);

            return std::make_tuple(LaplJL, LaplJR);
        }
        if constexpr (dimension == 3)
        {
            MeshIndex<3> prevX = index;
            prevX[0] -= 1;
            MeshIndex<3> nextX = index;
            nextX[0] += 1;

            MeshIndex<3> prevY = index;
            prevY[1] -= 1;
            MeshIndex<3> nextY = index;
            nextY[1] += 1;

            MeshIndex<3> prevZ = index;
            prevZ[2] -= 1;
            MeshIndex<3> nextZ = index;
            nextZ[2] += 1;

            auto [JL_X_1, JR_X_1] = LR(prevX, Direction::X);
            auto [JL_X1, JR_X1]   = LR(nextX, Direction::X);

            auto [JL_Y_1, JR_Y_1] = LR(prevY, Direction::Y);
            auto [JL_Y1, JR_Y1]   = LR(nextY, Direction::Y);

            auto [JL_Z_1, JR_Z_1] = LR(prevZ, Direction::Z);
            auto [JL_Z1, JR_Z1]   = LR(nextZ, Direction::Z);

            auto LaplJL = d2(dirX, JL_X_1, JL, JL_X1) + d2(dirY, JL_Y_1, JL, JL_Y1)
                          + d2(dirZ, JL_Z_1, JL, JL_Z1);
            auto LaplJR = d2(dirX, JR_X_1, JR, JR_X1) + d2(dirY, JR_Y_1, JR, JR_Y1)
                          + d2(dirZ, JR_Z_1, JR, JR_Z1);

            return std::make_tuple(LaplJL, LaplJR);
        }
    }

    template<auto direction, typename Field>
    auto reconstruct_uL_(Field const& F, MeshIndex<Field::dimension> index) const
    {
        return constant_uL_<direction>(F, index);
    }

    template<auto direction, typename Field>
    auto reconstruct_uR_(Field const& F, MeshIndex<Field::dimension> index) const
    {
        return constant_uR_<direction>(F, index);
    }

    template<auto direction, typename Field>
    auto constant_uL_(Field const& F, MeshIndex<Field::dimension> index) const
    {
        auto fieldCentering = layout_->centering(F.physicalQuantity());

        std::size_t dir;
        if (direction == Direction::X)
            dir = PHARE::core::dirX;
        else if (direction == Direction::Y)
            dir = PHARE::core::dirY;
        else
            dir = PHARE::core::dirZ;

        return (layout_->prevIndex(fieldCentering[dir], index[0]));
    }

    template<auto direction, typename Field>
    auto constant_uR_(Field const& F, MeshIndex<Field::dimension> index) const
    {
        return F(index[0]);
    }
};

} // namespace PHARE::core

#endif
