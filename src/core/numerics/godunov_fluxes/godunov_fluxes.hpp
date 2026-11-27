#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_HPP

#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/types.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/reconstructions/reconstructor.hpp"
#include "core/numerics/constrained_transport/hall_face_pair.hpp"
#include "core/numerics/reconstructions/weno3z_fallback.hpp"
#include "core/numerics/point_values_handler/point_value_converter.hpp"

#include "initializer/data_provider.hpp"

#include <limits>
#include <utility>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace PHARE::core
{
template<size_t dim>
constexpr auto getDirections()
{
    if constexpr (dim == 1)
    {
        return std::make_tuple(Direction::X);
    }
    else if constexpr (dim == 2)
    {
        return std::make_tuple(Direction::X, Direction::Y);
    }
    else if constexpr (dim == 3)
    {
        return std::make_tuple(Direction::X, Direction::Y, Direction::Z);
    }
}

template<auto direction, size_t dim>
auto getGrow(int const nghosts)
{
    Point<std::uint32_t, dim> p{};

    auto dir = static_cast<size_t>(direction);

    for (size_t i = 0; i < dim; ++i)
        if (i != dir)
            p[i] = nghosts;

    return p;
}

struct GodunovInfo : public OhmInfo
{
    double const gamma;

    GodunovInfo static FROM(initializer::PHAREDict const& dict)
    {
        return {{OhmInfo::FROM(dict)}, dict["heat_capacity_ratio"].template to<double>()};
    }
};


template<typename GridLayout, template<typename> typename Reconstruction, typename RiemannSolver,
         typename Equations>
class Godunov : public GodunovInfo
{
    using Super                                  = GodunovInfo;
    using Reconstruction_t                       = Reconstruction<GridLayout>;
    using Reconstructor_t                        = Reconstructor<Reconstruction_t>;
    constexpr static std::size_t projectionOrder = Reconstructor_t::pointValues ? 4 : 2;
    using RiemannSolver_t                        = RiemannSolver;
    constexpr static auto dimension              = GridLayout::dimension;

public:
    using Info_t      = Super;
    using Equations_t = Equations;

    template<typename T>
    using Rec = Reconstruction<T>;

    constexpr static auto Hall = Equations::hall;

    static_assert(GridLayout::options.field_ghost_width >= Reconstruction_t::nghosts + 1,
                  "MHD ghost width too small for the reconstruction stencil plus ampere's layer");

    explicit Godunov(GodunovInfo const& info, GridLayout const& layout)
        : Super{info}
        , layout_{layout}
        , is_resistive_{info.isResistive()}
        , is_hyper_resistive_{info.isHyperResistive()}
        , equations_{gamma}
        , face_pair_hall_{Hall && hallFacePairF12()}
        , riemann_{gamma, Hall && !info.isHyperResistive() && hallWhistlerFan()}
    {
        if (std::getenv("PHARE_O4_RECON_FALLBACK") != nullptr && dimension == 2)
        {
            auto const lo = layout_.AMRBox().lower;
            auto it       = pvMaskRegistry.find({int(lo[0]), int(lo[1])});
            if (it != pvMaskRegistry.end())
                fallbackMask_ = &it->second;
        }
    }

    template<typename State, typename Fluxes>
    void operator()(auto& ct_state, auto const& dissipative_electric_state, State& state,
                    Fluxes& fluxes)
    {
        constexpr auto directions = getDirections<dimension>();

        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            layout_.evalOnBiggerBox(
                fluxes.template expose_centering<direction>(),
                getGrow<direction, dimension>(Reconstruction_t::nghosts), [&](auto&... indices) {
                    auto&& [uL, uR] = reconstructSelected_<direction>(state, {indices...});
                    auto&& u = std::forward_as_tuple(uL, uR);

                    if constexpr (dimension == 2) // EXPERIMENT PHARE_O4_FACEP_DIAG
                        if (o4FaceProbe.on)
                            for (auto const* w : {&uL, &uR})
                                if (!(w->P > 0) || !(w->rho > 0))
                                    if (o4FaceProbe.n++ == 0)
                                    {
                                        auto const a = layout_.localToAMR(
                                            Point<int, dimension>{int(indices)...});
                                        o4FaceProbe.P   = w->P;
                                        o4FaceProbe.rho = w->rho;
                                        o4FaceProbe.dir = static_cast<int>(direction);
                                        o4FaceProbe.x   = a[0] * layout_.meshSize()[0];
                                        o4FaceProbe.y   = a[1] * layout_.meshSize()[1];
                                    }

                    if constexpr (Hall)
                        if (face_pair_hall_)
                        {
                            face_pair_hall_flux_<direction>(ct_state, fluxes, uL, uR, {indices...});
                            return;
                        }

                    if constexpr (Hall)
                    {
                        auto const& [jL, jR] = Reconstructor_t::template center_reconstruct<
                            direction,
                            GridLayout::implT::template edgeXToCellCenter<projectionOrder>,
                            GridLayout::implT::template edgeYToCellCenter<projectionOrder>,
                            GridLayout::implT::template edgeZToCellCenter<projectionOrder>>(
                            state.J, {indices...});

                        auto const& j = std::forward_as_tuple(jL, jR);

                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u),
                                                                          std::get<i>(j));
                        });

                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(
                                uL, uR, fL, fR, jL, jR, layout_.inverseMeshSize(direction));

                        ct_state.template save<direction>(riemann_.vt, riemann_.jt, riemann_.rhot,
                                                          riemann_.uct_coefs, {indices...});
                    }
                    else
                    {
                        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            return equations_.template compute<direction>(std::get<i>(u));
                        });

                        fluxes.template get_dir<direction>({indices...})
                            = riemann_.template solve<direction>(uL, uR, fL, fR);

                        ct_state.template save<direction>(riemann_.vt, riemann_.uct_coefs,
                                                          {indices...});
                    }
                });

            if (is_resistive_ || is_hyper_resistive_ || face_pair_hall_)
            {
                auto const dissipative = [&](auto&... indices) {
                    auto F              = fluxes.template get_dir<direction>({indices...});
                    auto const [Et, Bt] = transverse_on_face_<direction>(
                        dissipative_electric_state.E(), state.B, {indices...});
                    equations_.template dissipative_contributions<direction>(Et, Bt, F.B, F.Etot());
                };

                // point-value fluxes are averaged over their face afterwards, which reads one
                // more face on each transverse side
                if constexpr (Reconstructor_t::pointValues)
                    layout_.evalOnBiggerBox(fluxes.template expose_centering<direction>(),
                                            getGrow<direction, dimension>(1), dissipative);
                else
                    layout_.evalOnBox(fluxes.template expose_centering<direction>(), dissipative);
            }
        });
    }

    // EXPERIMENT PHARE_O4_RECON_FALLBACK: WENO3Z on faces adjacent to a troubled cell
    template<auto direction, typename State>
    auto reconstructSelected_(State const& S, MeshIndex<dimension> idx) const
    {
        if constexpr (Reconstructor_t::pointValues && dimension == 2)
            if (fallbackMask_)
            {
                constexpr auto d = static_cast<std::size_t>(direction);
                auto mk          = [&](int a, int b) {
                    if (a < 0 || b < 0 || a >= fallbackMask_->nx || b >= fallbackMask_->ny)
                        return 0.0;
                    return fallbackMask_->m[static_cast<std::size_t>(a * fallbackMask_->ny + b)];
                };
                int const i = int(idx[0]), j = int(idx[1]);
                double const tr = std::max(mk(i, j), d == 0 ? mk(i - 1, j) : mk(i, j - 1));
                if (tr > 0)
                {
                    ++nFallbackFaces_;
                    auto rec = [&](auto const& F) {
                        auto at = [&](int k) {
                            auto c = idx;
                            c[d] += k;
                            return F(c);
                        };
                        if (fallbackMode_ == 2) // first order: piecewise constant
                            return std::make_pair(at(-1), at(0));
                        return std::make_pair(o4fb::weno3zL(at(-2), at(-1), at(0)),
                                              o4fb::weno3zR(at(-1), at(0), at(1)));
                    };
                    auto crec = [&]<auto tdir>(auto const& U) {
                        auto at = [&](int k) {
                            auto c = idx;
                            c[d] += k;
                            return GridLayout::template project<
                                Reconstructor_t::template projection<tdir>>(U, c);
                        };
                        if (fallbackMode_ == 2) // first order: piecewise constant
                            return std::make_pair(at(-1), at(0));
                        return std::make_pair(o4fb::weno3zL(at(-2), at(-1), at(0)),
                                              o4fb::weno3zR(at(-1), at(0), at(1)));
                    };
                    auto [rhoL, rhoR] = rec(S.rho);
                    auto [VxL, VxR]   = rec(S.V(Component::X));
                    auto [VyL, VyR]   = rec(S.V(Component::Y));
                    auto [VzL, VzR]   = rec(S.V(Component::Z));
                    auto [PL, PR]     = rec(S.P);
                    PerIndexVector<double> BL, BR;
                    if constexpr (d == 0)
                    {
                        BL.x = BR.x = S.B(Component::X)(idx);
                        std::tie(BL.y, BR.y) = crec.template operator()<Direction::Y>(S.B(Component::Y));
                        std::tie(BL.z, BR.z) = crec.template operator()<Direction::Z>(S.B(Component::Z));
                    }
                    else
                    {
                        BL.y = BR.y = S.B(Component::Y)(idx);
                        std::tie(BL.x, BR.x) = crec.template operator()<Direction::X>(S.B(Component::X));
                        std::tie(BL.z, BR.z) = crec.template operator()<Direction::Z>(S.B(Component::Z));
                    }
                    auto hi = Reconstructor_t::template reconstruct<direction>(S, idx);
                    auto& [hL, hR] = hi;
                    hL.rho = rhoL; hR.rho = rhoR;
                    hL.V   = {VxL, VyL, VzL}; hR.V = {VxR, VyR, VzR};
                    hL.P   = PL; hR.P = PR;
                    hL.B   = BL; hR.B = BR;
                    return hi;
                }
            }
        return Reconstructor_t::template reconstruct<direction>(S, idx);
    }

    bool isResistive() const { return is_resistive_; }
    bool isHyperResistive() const { return is_hyper_resistive_; }

private:
    // Hall J stays out of the Riemann solver: the Hall field comes from the face pair on the
    // edges and enters the B and Etot fluxes with the dissipative EMF. The upwind whistler fan,
    // when on, still widens the magnetic subsystem's signal speed.
    template<auto direction>
    void face_pair_hall_flux_(auto& ct_state, auto& fluxes, auto& uL, auto& uR,
                              MeshIndex<dimension> const& idx)
    {
        auto const& [fL, fR] = for_N<2, for_N_R_mode::make_tuple>(
            [&](auto i) { return equations_.template compute<direction>(i == 0 ? uL : uR); });

        if constexpr (requires {
                          riemann_.template solve<direction>(uL, uR, fL, fR,
                                                             layout_.inverseMeshSize(direction));
                      })
            fluxes.template get_dir<direction>(idx) = riemann_.template solve<direction>(
                uL, uR, fL, fR, layout_.inverseMeshSize(direction));
        else
            throw std::runtime_error("the F12 face-pair Hall field needs Rusanov");

        ct_state.template save<direction>(riemann_.vt, riemann_.uct_coefs, idx);
    }

    template<auto direction>
    auto transverse_on_face_(auto const& E, auto const& B, MeshIndex<dimension> idx) const
    {
        using implT        = GridLayout::implT;
        auto constexpr nan = std::numeric_limits<double>::quiet_NaN();

        auto const& Ex = E(Component::X);
        auto const& Ey = E(Component::Y);
        auto const& Ez = E(Component::Z);
        auto const& Bx = B(Component::X);
        auto const& By = B(Component::Y);
        auto const& Bz = B(Component::Z);

        if constexpr (direction == Direction::X)
            return std::make_pair(
                PerIndexVector<double>{
                    nan,
                    GridLayout::template project<implT::template edgeYToFaceX<projectionOrder>>(
                        Ey, idx),
                    GridLayout::template project<implT::template edgeZToFaceX<projectionOrder>>(
                        Ez, idx)},
                PerIndexVector<double>{
                    nan,
                    GridLayout::template project<implT::template ByToFaceX<projectionOrder>>(By,
                                                                                             idx),
                    GridLayout::template project<implT::template BzToFaceX<projectionOrder>>(Bz,
                                                                                             idx)});
        else if constexpr (direction == Direction::Y)
            return std::make_pair(
                PerIndexVector<double>{
                    GridLayout::template project<implT::template edgeXToFaceY<projectionOrder>>(
                        Ex, idx),
                    nan,
                    GridLayout::template project<implT::template edgeZToFaceY<projectionOrder>>(
                        Ez, idx)},
                PerIndexVector<double>{
                    GridLayout::template project<implT::template BxToFaceY<projectionOrder>>(Bx,
                                                                                             idx),
                    nan,
                    GridLayout::template project<implT::template BzToFaceY<projectionOrder>>(Bz,
                                                                                             idx)});
        else
            return std::make_pair(
                PerIndexVector<double>{
                    GridLayout::template project<implT::template edgeXToFaceZ<projectionOrder>>(
                        Ex, idx),
                    GridLayout::template project<implT::template edgeYToFaceZ<projectionOrder>>(
                        Ey, idx),
                    nan},
                PerIndexVector<double>{
                    GridLayout::template project<implT::template BxToFaceZ<projectionOrder>>(Bx,
                                                                                             idx),
                    GridLayout::template project<implT::template ByToFaceZ<projectionOrder>>(By,
                                                                                             idx),
                    nan});
    }

    // EXPERIMENT PHARE_O4_RECON_FALLBACK: this patch's troubled mask (built this stage)
    PVMask const* fallbackMask_ = nullptr; // set in the constructor (needs layout_)
    int const fallbackMode_     = [] { // 1: WENO3Z, 2: piecewise constant
        auto const* e = std::getenv("PHARE_O4_RECON_FALLBACK");
        return e ? std::atoi(e) : 0;
    }();
    mutable std::size_t nFallbackFaces_ = 0;

    GridLayout layout_;
    bool const is_resistive_;
    bool const is_hyper_resistive_;
    bool const face_pair_hall_;
    Equations equations_;
    RiemannSolver_t riemann_;
};

} // namespace PHARE::core

#endif
