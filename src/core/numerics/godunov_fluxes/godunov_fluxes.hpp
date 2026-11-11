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
#include "core/numerics/godunov_fluxes/o4_facep_debug.hpp"

#include "initializer/data_provider.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <tuple>
#include <cstddef>
#include <cstdint>

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
        , riemann_{gamma, Hall && !info.isHyperResistive()}
    {
        riemann_.hydroEtot_      = o4Poynting_;
        riemann_.etotHydroSpeed_ = o4Poynting_ && std::getenv("PHARE_O4_ETOT_HYDRO_SPEED");
    }

    // EXPERIMENT PHARE_O4_POYNTING=1 (2D O4 Hall, with PHARE_O4_HALL_ZW): the Etot flux keeps only
    // its hydro part, the closure Poynting flux is added in compute_fluxes, and the face loop grows
    // so UCT can evaluate the EMF on a 2-layer band (O4/poynting P7)
    bool const o4Poynting_ = Reconstructor_t::pointValues && dimension == 2 && Hall
                             && std::getenv("PHARE_O4_POYNTING") != nullptr;

    template<typename State, typename Fluxes>
    void operator()(auto& ct_state, auto const& dissipative_electric_state, State& state,
                    Fluxes& fluxes)
    {
        constexpr auto directions = getDirections<dimension>();

        constexpr auto num_directions = std::tuple_size_v<std::decay_t<decltype(directions)>>;

        for_N<num_directions>([&](auto i) {
            constexpr Direction direction = std::get<i>(directions);

            auto grow = getGrow<direction, dimension>(Reconstruction_t::nghosts);
            if (o4Poynting_)
                for (std::size_t a = 0; a < dimension; ++a)
                    grow[a] += 2; // +2 transverse and +2 along the normal: the band's faces
            layout_.evalOnBiggerBox(
                fluxes.template expose_centering<direction>(), grow, [&](auto&... indices) {
                    auto&& [uL, uR]
                        = Reconstructor_t::template reconstruct<direction>(state, {indices...});
                    auto&& u = std::forward_as_tuple(uL, uR);

                    if (faceDiag_) // EXPERIMENT PHARE_O4_FACEP_DIAG: non-positive face states
                        for (auto const* w : {&uL, &uR})
                            if (!(w->P > 0) || !(w->rho > 0))
                            {
                                if (nBadFace_++ == 0)
                                {
                                    badP_   = w->P;
                                    badRho_ = w->rho;
                                    auto const a
                                        = layout_.localToAMR(Point<int, dimension>{int(indices)...});
                                    for (std::size_t k = 0; k < dimension; ++k)
                                        badX_[k] = a[k] * layout_.meshSize()[k];
                                    badDir_ = static_cast<int>(direction);
                                }
                                auto& fb = o4dbg::faceBad;
                                if constexpr (dimension == 2)
                                    if (o4dbg::dumpDir() && !fb.pending && !fb.done)
                                    {
                                        fb.pending = true;
                                        fb.dir     = static_cast<int>(direction);
                                        fb.side    = (w == &uL) ? 0 : 1;
                                        auto const a = layout_.localToAMR(
                                            Point<int, dimension>{int(indices)...});
                                        fb.amr = {a[0], a[1]};
                                        fb.P   = w->P;
                                        fb.rho = w->rho;
                                        fb.PL  = uL.P;
                                        fb.PR  = uR.P;
                                        MeshIndex<dimension> idx{indices...};
                                        for (int k = 0; k < 6; ++k)
                                        {
                                            auto c = idx;
                                            c[static_cast<std::size_t>(direction)] += k - 3;
                                            fb.Pst[k]   = state.P(c);
                                            fb.rhost[k] = state.rho(c);
                                        }
                                    }
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

                        auto [fL, fR] = for_N<2, for_N_R_mode::make_tuple>([&](auto i) {
                            if (o4Poynting_)
                            {
                                // hydro part of the Etot flux only: no Hall, no ideal Poynting
                                auto f        = equations_.template compute<direction>(std::get<i>(u));
                                auto const& w = std::get<i>(u);
                                constexpr auto d = static_cast<std::size_t>(direction);
                                auto const vn = d == 0 ? w.V.x : (d == 1 ? w.V.y : w.V.z);
                                auto const bn = d == 0 ? w.B.x : (d == 1 ? w.B.y : w.B.z);
                                auto const b2 = w.B.x * w.B.x + w.B.y * w.B.y + w.B.z * w.B.z;
                                auto const vb = w.V.x * w.B.x + w.V.y * w.B.y + w.V.z * w.B.z;
                                f.Etot() -= b2 * vn - bn * vb;
                                return f;
                            }
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

            if (is_resistive_ || is_hyper_resistive_)
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

    ~Godunov()
    {
        if (nBadFace_ > 0)
            std::printf("FACEP nbad=%zu first: dir=%d face@(%.3f,%.3f) P=%.4e rho=%.4e\n",
                        nBadFace_, badDir_, badX_[0], dimension > 1 ? badX_[1] : 0., badP_,
                        badRho_);
    }

    bool isResistive() const { return is_resistive_; }
    bool isHyperResistive() const { return is_hyper_resistive_; }

private:
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

    bool const faceDiag_ = std::getenv("PHARE_O4_FACEP_DIAG") != nullptr;
    std::size_t nBadFace_ = 0;
    double badP_ = 0, badRho_ = 0;
    std::array<double, 3> badX_{};
    int badDir_ = -1;

    GridLayout layout_;
    bool const is_resistive_;
    bool const is_hyper_resistive_;
    Equations equations_;
    RiemannSolver_t riemann_;
};

} // namespace PHARE::core

#endif
