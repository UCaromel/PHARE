#ifndef PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP
#define PHARE_UPWIND_CONSTRAINED_TRANSPORT_HPP


#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/index/index.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"

#include <cmath>
#include <cstddef>

namespace PHARE::core
{
using UpwindConstrainedTransportInfo = OhmInfo;

template<typename GridLayout, template<typename> typename Reconstruction, bool Hall>
class UpwindConstrainedTransport : UpwindConstrainedTransportInfo
{
    using Super                     = UpwindConstrainedTransportInfo;
    using Reconstruction_t          = Reconstruction<GridLayout>;
    constexpr static auto dimension = GridLayout::dimension;
    using Super::hyper_mode;

public:
    using Info_t = Super;

    UpwindConstrainedTransport(UpwindConstrainedTransportInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
        , is_dissipative_{info.isResistive() || info.isHyperResistive()}
    {
    }

    void operator()(auto& ct_state, auto const& dissipative_electric_state,
                    auto const& ssi_state, auto& mhd_state) const
    {
        if constexpr (HasSharedSmoothness<Reconstruction_t>)
            if (ssi_state.enabled())
                return compute_<true>(ct_state, dissipative_electric_state, ssi_state, mhd_state);

        compute_<false>(ct_state, dissipative_electric_state, ssi_state, mhd_state);
    }

private:
    template<bool ssi>
    void compute_(auto& ct_state, auto const& dissipative_electric_state, auto const& ssi_state,
                  auto& mhd_state) const
    {
        auto& E       = mhd_state.E;
        auto const& B = mhd_state.B;

        auto& Ex = E(Component::X);
        auto& Ey = E(Component::Y);
        auto& Ez = E(Component::Z);

        layout_.evalOnBox(
            Ex, [&](auto&... args) { ExEq_<ssi>(ct_state, ssi_state, Ex, B, {args...}); });
        layout_.evalOnBox(
            Ey, [&](auto&... args) { EyEq_<ssi>(ct_state, ssi_state, Ey, B, {args...}); });
        layout_.evalOnBox(
            Ez, [&](auto&... args) { EzEq_<ssi>(ct_state, ssi_state, Ez, B, {args...}); });

        if (is_dissipative_)
        {
            auto const& Ediss = dissipative_electric_state.E();
            for (auto c : {Component::X, Component::Y, Component::Z})
                layout_.evalOnBox(E(c), [&](auto&... args) { E(c)(args...) += Ediss(c)(args...); });
        }
    }

    template<bool ssi>
    void ExEq_(auto& ct_state, auto const& ssi_state, auto& Ex, auto const& B,
              MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 2)
        {
            auto [BzL, BzR] = reconstruct_<ssi, Direction::Y>(ssi_state, B(Component::Z), idx);

            auto FL = BzL * ct_state.vt_y(Component::Y)(idx)
                      - B(Component::Y)(idx) * ct_state.vt_y(Component::Z)(idx);
            auto FR = BzR * ct_state.vt_y(Component::Y)(idx)
                      - B(Component::Y)(idx) * ct_state.vt_y(Component::Z)(idx);

            Ex(idx) = -(ct_state.aL_y(idx) * FL + ct_state.aR_y(idx) * FR)
                      + (ct_state.dR_y(idx) * BzR - ct_state.dL_y(idx) * BzL);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / ct_state.template getRhot<Direction::Y>()(idx);
                auto JxB_x_L = ct_state.template getJt<Direction::Y>()(Component::Y)(idx)*BzL
                               - ct_state.template getJt<Direction::Y>()(Component::Z)(idx)*B(
                                   Component::Y)(idx);
                auto JxB_x_R = ct_state.template getJt<Direction::Y>()(Component::Y)(idx)*BzR
                               - ct_state.template getJt<Direction::Y>()(Component::Z)(idx)*B(
                                   Component::Y)(idx);

                auto HallL = -JxB_x_L * invRho;
                auto HallR = -JxB_x_R * invRho;

                auto F_Bz_y = ct_state.aL_y(idx) * HallL + ct_state.aR_y(idx) * HallR;

                Ex(idx) += -F_Bz_y;
            }
        }
        else if constexpr (dimension == 3)
        {
            auto aS = 0.5
                      * (ct_state.aL_y(idx)
                         + ct_state.aL_y(layout_.template previous<Direction::Z>(idx)));
            auto aN = 0.5
                      * (ct_state.aR_y(idx)
                         + ct_state.aR_y(layout_.template previous<Direction::Z>(idx)));
            auto aB = 0.5
                      * (ct_state.aL_z(idx)
                         + ct_state.aL_z(layout_.template previous<Direction::Y>(idx)));
            auto aT = 0.5
                      * (ct_state.aR_z(idx)
                         + ct_state.aR_z(layout_.template previous<Direction::Y>(idx)));
            auto dS = 0.5
                      * (ct_state.dL_y(idx)
                         + ct_state.dL_y(layout_.template previous<Direction::Z>(idx)));
            auto dN = 0.5
                      * (ct_state.dR_y(idx)
                         + ct_state.dR_y(layout_.template previous<Direction::Z>(idx)));
            auto dB = 0.5
                      * (ct_state.dL_z(idx)
                         + ct_state.dL_z(layout_.template previous<Direction::Y>(idx)));
            auto dT = 0.5
                      * (ct_state.dR_z(idx)
                         + ct_state.dR_z(layout_.template previous<Direction::Y>(idx)));

            auto [vyS, vyN] = reconstruct_<ssi, Direction::Y, Direction::Z>(ssi_state, ct_state.vt_z(Component::Y), idx);
            auto [vzB, vzT] = reconstruct_<ssi, Direction::Z, Direction::Y>(ssi_state, ct_state.vt_y(Component::Z), idx);

            auto [BzS, BzN] = reconstruct_<ssi, Direction::Y, Direction::Z>(ssi_state, B(Component::Z), idx);
            auto [ByB, ByT] = reconstruct_<ssi, Direction::Z, Direction::Y>(ssi_state, B(Component::Y), idx);

            Ex(idx) = (aB * vzB * ByB + aT * vzT * ByT) - (aS * vyS * BzS + aN * vyN * BzN)
                      - (dT * ByT - dB * ByB) + (dN * BzN - dS * BzS);

            if constexpr (Hall)
            {
                auto [jyS, jyN] = reconstruct_<ssi, Direction::Y, Direction::Z>(ssi_state, ct_state.template getJt<Direction::Z>()(Component::Y), idx);
                auto [jzB, jzT] = reconstruct_<ssi, Direction::Z, Direction::Y>(ssi_state, ct_state.template getJt<Direction::Y>()(Component::Z), idx);

                auto [rhoS, rhoN] = reconstruct_<ssi, Direction::Y, Direction::Z>(ssi_state, ct_state.template getRhot<Direction::Z>(), idx);
                auto [rhoB, rhoT] = reconstruct_<ssi, Direction::Z, Direction::Y>(ssi_state, ct_state.template getRhot<Direction::Y>(), idx);

                Ex(idx) += -(aB * jzB * ByB / rhoB + aT * jzT * ByT / rhoT)
                           + (aS * jyS * BzS / rhoS + aN * jyN * BzN / rhoN);
            }
        }
    }

    template<bool ssi>
    void EyEq_(auto& ct_state, auto const& ssi_state, auto& Ey, auto const& B,
              MeshIndex<dimension> idx) const
    {
        if constexpr (dimension <= 2)
        {
            auto [BzL, BzR] = reconstruct_<ssi, Direction::X>(ssi_state, B(Component::Z), idx);

            auto FL = BzL * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Z)(idx);
            auto FR = BzR * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Z)(idx);

            Ey(idx) = (ct_state.aL_x(idx) * FL + ct_state.aR_x(idx) * FR)
                      - (ct_state.dR_x(idx) * BzR - ct_state.dL_x(idx) * BzL);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / ct_state.template getRhot<Direction::X>()(idx);
                auto JxB_y_L = ct_state.template getJt<Direction::X>()(Component::Z)(idx)*B(
                                   Component::X)(idx)
                               - ct_state.template getJt<Direction::X>()(Component::X)(idx)*BzL;
                auto JxB_y_R = ct_state.template getJt<Direction::X>()(Component::Z)(idx)*B(
                                   Component::X)(idx)
                               - ct_state.template getJt<Direction::X>()(Component::X)(idx)*BzR;

                auto HallL = JxB_y_L * invRho;
                auto HallR = JxB_y_R * invRho;

                auto F_Bz_x = ct_state.aL_x(idx) * HallL + ct_state.aR_x(idx) * HallR;

                Ey(idx) += F_Bz_x;
            }
        }
        else if constexpr (dimension == 3)
        {
            auto aW = 0.5
                      * (ct_state.aL_x(idx)
                         + ct_state.aL_x(layout_.template previous<Direction::Z>(idx)));
            auto aE = 0.5
                      * (ct_state.aR_x(idx)
                         + ct_state.aR_x(layout_.template previous<Direction::Z>(idx)));
            auto aB = 0.5
                      * (ct_state.aL_z(idx)
                         + ct_state.aL_z(layout_.template previous<Direction::X>(idx)));
            auto aT = 0.5
                      * (ct_state.aR_z(idx)
                         + ct_state.aR_z(layout_.template previous<Direction::X>(idx)));
            auto dW = 0.5
                      * (ct_state.dL_x(idx)
                         + ct_state.dL_x(layout_.template previous<Direction::Z>(idx)));
            auto dE = 0.5
                      * (ct_state.dR_x(idx)
                         + ct_state.dR_x(layout_.template previous<Direction::Z>(idx)));
            auto dB = 0.5
                      * (ct_state.dL_z(idx)
                         + ct_state.dL_z(layout_.template previous<Direction::X>(idx)));
            auto dT = 0.5
                      * (ct_state.dR_z(idx)
                         + ct_state.dR_z(layout_.template previous<Direction::X>(idx)));

            auto [vxW, vxE] = reconstruct_<ssi, Direction::X, Direction::Z>(ssi_state, ct_state.vt_z(Component::X), idx);
            auto [vzB, vzT] = reconstruct_<ssi, Direction::Z, Direction::X>(ssi_state, ct_state.vt_x(Component::Z), idx);
            auto [BzW, BzE] = reconstruct_<ssi, Direction::X, Direction::Z>(ssi_state, B(Component::Z), idx);
            auto [BxB, BxT] = reconstruct_<ssi, Direction::Z, Direction::X>(ssi_state, B(Component::X), idx);

            Ey(idx) = (aW * vxW * BzW + aE * vxE * BzE) - (aB * vzB * BxB + aT * vzT * BxT)
                      - (dE * BzE - dW * BzW) + (dT * BxT - dB * BxB);

            if constexpr (Hall)
            {
                auto [jxW, jxE] = reconstruct_<ssi, Direction::X, Direction::Z>(ssi_state, ct_state.template getJt<Direction::Z>()(Component::X), idx);
                auto [jzB, jzT] = reconstruct_<ssi, Direction::Z, Direction::X>(ssi_state, ct_state.template getJt<Direction::X>()(Component::Z), idx);
                auto [rhoW, rhoE] = reconstruct_<ssi, Direction::X, Direction::Z>(ssi_state, ct_state.template getRhot<Direction::Z>(), idx);
                auto [rhoB, rhoT] = reconstruct_<ssi, Direction::Z, Direction::X>(ssi_state, ct_state.template getRhot<Direction::X>(), idx);
                Ey(idx) += -(aW * jxW * BzW / rhoW + aE * jxE * BzE / rhoE)
                           + (aB * jzB * BxB / rhoB + aT * jzT * BxT / rhoT);
            }
        }
    }

    template<bool ssi>
    void EzEq_(auto& ct_state, auto const& ssi_state, auto& Ez, auto const& B,
              MeshIndex<dimension> idx) const
    {
        if constexpr (dimension == 1)
        {
            auto [ByL, ByR] = reconstruct_<ssi, Direction::X>(ssi_state, B(Component::Y), idx);

            auto FL = ByL * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Y)(idx);
            auto FR = ByR * ct_state.vt_x(Component::X)(idx)
                      - B(Component::X)(idx) * ct_state.vt_x(Component::Y)(idx);

            Ez(idx) = -(ct_state.aL_x(idx) * FL + ct_state.aR_x(idx) * FR)
                      + (ct_state.dR_x(idx) * ByR - ct_state.dL_x(idx) * ByL);

            if constexpr (Hall)
            {
                auto invRho  = 1.0 / ct_state.template getRhot<Direction::X>()(idx);
                auto JxB_z_L = ct_state.template getJt<Direction::X>()(Component::X)(idx)*ByL
                               - ct_state.template getJt<Direction::X>()(Component::Y)(idx)*B(
                                   Component::X)(idx);
                auto JxB_z_R = ct_state.template getJt<Direction::X>()(Component::X)(idx)*ByR
                               - ct_state.template getJt<Direction::X>()(Component::Y)(idx)*B(
                                   Component::X)(idx);

                auto HallL = -JxB_z_L * invRho;
                auto HallR = -JxB_z_R * invRho;

                auto F_By_x = ct_state.aL_x(idx) * HallL + ct_state.aR_x(idx) * HallR;

                Ez(idx) += -F_By_x;
            }
        }
        else if constexpr (dimension >= 2)
        {
            auto aW = 0.5
                      * (ct_state.aL_x(idx)
                         + ct_state.aL_x(layout_.template previous<Direction::Y>(idx)));
            auto aE = 0.5
                      * (ct_state.aR_x(idx)
                         + ct_state.aR_x(layout_.template previous<Direction::Y>(idx)));
            auto aS = 0.5
                      * (ct_state.aL_y(idx)
                         + ct_state.aL_y(layout_.template previous<Direction::X>(idx)));
            auto aN = 0.5
                      * (ct_state.aR_y(idx)
                         + ct_state.aR_y(layout_.template previous<Direction::X>(idx)));
            auto dW = 0.5
                      * (ct_state.dL_x(idx)
                         + ct_state.dL_x(layout_.template previous<Direction::Y>(idx)));
            auto dE = 0.5
                      * (ct_state.dR_x(idx)
                         + ct_state.dR_x(layout_.template previous<Direction::Y>(idx)));
            auto dS = 0.5
                      * (ct_state.dL_y(idx)
                         + ct_state.dL_y(layout_.template previous<Direction::X>(idx)));
            auto dN = 0.5
                      * (ct_state.dR_y(idx)
                         + ct_state.dR_y(layout_.template previous<Direction::X>(idx)));

            auto [vyS, vyN] = reconstruct_<ssi, Direction::Y, Direction::X>(ssi_state, ct_state.vt_x(Component::Y), idx);
            auto [vxW, vxE] = reconstruct_<ssi, Direction::X, Direction::Y>(ssi_state, ct_state.vt_y(Component::X), idx);

            auto [BxS, BxN] = reconstruct_<ssi, Direction::Y, Direction::X>(ssi_state, B(Component::X), idx);
            auto [ByW, ByE] = reconstruct_<ssi, Direction::X, Direction::Y>(ssi_state, B(Component::Y), idx);

            Ez(idx) = -(aW * vxW * ByW + aE * vxE * ByE) + (aS * vyS * BxS + aN * vyN * BxN)
                      + (dE * ByE - dW * ByW) - (dN * BxN - dS * BxS);

            if constexpr (Hall)
            {
                auto [jyS, jyN] = reconstruct_<ssi, Direction::Y, Direction::X>(ssi_state, ct_state.template getJt<Direction::X>()(Component::Y), idx);
                auto [jxW, jxE] = reconstruct_<ssi, Direction::X, Direction::Y>(ssi_state, ct_state.template getJt<Direction::Y>()(Component::X), idx);

                auto [rhoS, rhoN] = reconstruct_<ssi, Direction::Y, Direction::X>(ssi_state, ct_state.template getRhot<Direction::X>(), idx);
                auto [rhoW, rhoE] = reconstruct_<ssi, Direction::X, Direction::Y>(ssi_state, ct_state.template getRhot<Direction::Y>(), idx);

                Ez(idx) += (aW * jxW * ByW / rhoW + aE * jxE * ByE / rhoE)
                           - (aS * jyS * BxS / rhoS + aN * jyN * BxN / rhoN);
            }
        }
    }

    // Reconstructs a face quantity along recon_dir to the edge at idx. With shared smoothness, a
    // quantity located on faces normal to faceNormal takes, on each side, the mean betas of the
    // two cells sharing its face; without faceNormal it is cell-centred in the transverse plane.
    template<bool ssi, auto recon_dir, auto... faceNormal>
    auto reconstruct_(auto const& ssi_state, auto const& field, MeshIndex<dimension> idx) const
    {
        static_assert(sizeof...(faceNormal) <= 1);

        if constexpr (ssi)
        {
            auto const betasOf = [&](MeshIndex<dimension> const cell) {
                auto betas = ssi_state.template get<recon_dir>(cell);
                if constexpr (sizeof...(faceNormal) == 1)
                {
                    auto const other = ssi_state.template get<recon_dir>(
                        layout_.template previous<faceNormal...>(cell));
                    for (std::size_t m = 0; m < betas.size(); ++m)
                        betas[m] = 0.5 * (betas[m] + other[m]);
                }
                return betas;
            };

            return Reconstruction_t::template reconstruct<recon_dir>(
                field, idx, betasOf(layout_.template previous<recon_dir>(idx)), betasOf(idx));
        }
        else
            return Reconstruction_t::template reconstruct<recon_dir>(field, idx);
    }

    GridLayout layout_;
    bool const is_dissipative_;
};
} // namespace PHARE::core

#endif
