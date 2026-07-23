#ifndef PHARE_CORE_NUMERICS_POINT_VALUE_HANDLER_HPP
#define PHARE_CORE_NUMERICS_POINT_VALUE_HANDLER_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"

#include <algorithm>
#include <cstdint>

namespace PHARE::core
{
enum class PointValueConversionMode { ToPointValue, ToAverage };

// Half-width of the average <-> point-value conversion stencil (the +-1 laplacian corrections
// below). Point values are LOCAL derived quantities: they are computed on the ghost box shrunk
// by this amount, so every value is computable from allocated average ghosts and no point-value
// refine schedule is needed. Consumers needing point values on ghost layers must stay within
// nbrGhosts - point_value_conversion_shrink (faces: no shrink in the component's normal
// direction, whose conversion is transverse-only).
inline constexpr std::uint32_t point_value_conversion_shrink = 1;

// Layout-taking stateless core operating on a PointValueState (passed by reference). Mirrors the
// State-helper / inline-transformer idiom of Godunov / UpwindConstrainedTransport / Ampere.
template<typename GridLayout>
class PointValueHandler
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    explicit PointValueHandler(GridLayout const& layout)
        : layout_{layout}
    {
    }

    // average -> point value of the model state into the point-value buffers.
    //
    // Evaluated on the ghost box shrunk by the conversion stencil (point_value_conversion_shrink)
    // so every point value is computable from allocated average ghosts — the point-value world has
    // no interlevel schedules (same discipline as the current J). With g allocated ghosts:
    //   - cell-centered quantities: full laplacian (+-1 isotropic) -> valid on g-1 layers;
    //   - face-centered B: transverse laplacian only -> valid on g-1 transverse layers, g layers
    //     in the component's normal direction (no correction there, no shrink).
    void operator()(auto& pv, auto const& state)
    {
        static constexpr auto toPointValue = PointValueConversionMode::ToPointValue;
        static constexpr auto shrink       = point_value_conversion_shrink;

        auto convert_cell = [&](auto const& src, auto& dst) {
            Point<std::uint32_t, dimension> amount;
            for (std::size_t i = 0; i < dimension; ++i)
                amount[i] = shrink;

            layout_.evalOnShrinkedGhostBox(src, amount, [&](auto&... args) mutable {
                cell_center_conversion_<toPointValue>(src, dst, {args...});
            });
        };

        convert_cell(state.rho, pv.rho);
        convert_cell(state.Etot, pv.Etot);
        for (int i = 0; i < 3; ++i)
        {
            convert_cell(state.rhoV(static_cast<core::Component>(i)),
                         pv.rhoV(static_cast<core::Component>(i)));
        }

        auto convert_face = [&]<auto dir_tag>(auto const& src, auto& dst) {
            Point<std::uint32_t, dimension> amount;
            for (std::size_t i = 0; i < dimension; ++i)
                amount[i] = (i == static_cast<std::size_t>(dir_tag)) ? 0u : shrink;

            layout_.evalOnShrinkedGhostBox(src, amount, [&](auto&... args) mutable {
                face_center_conversion_<dir_tag, toPointValue>(src, dst, {args...});
            });
        };

        convert_face.template operator()<Direction::X>(state.B(core::Component::X),
                                                       pv.B(core::Component::X));
        convert_face.template operator()<Direction::Y>(state.B(core::Component::Y),
                                                       pv.B(core::Component::Y));
        convert_face.template operator()<Direction::Z>(state.B(core::Component::Z),
                                                       pv.B(core::Component::Z));
    }

    // point value -> average of the point-value fluxes and electric field, written back in place.
    // tmpFluxes_/tmpE_ hold the point-value sources because the laplacian cannot be inplace.
    void point_value_fluxes_to_integral(auto& pv, auto& fluxes, auto& E)
    {
        static constexpr auto toAverage = PointValueConversionMode::ToAverage;

        pv.tmpE_.copyData(E);
        pv.tmpFluxes_.copyData(fluxes);

        auto convert_hydro_fluxes
            = [&]<auto dir_tag>(auto& src_rho_f, auto& src_rhov_f, auto& src_etot_f,
                                auto& dst_rho_f, auto& dst_rhov_f, auto& dst_etot_f) {
                  auto convert = [&](auto const& src, auto& dst) {
                      layout_.evalOnBox(dst, [&](auto&... args) mutable {
                          face_center_conversion_<dir_tag, toAverage>(src, dst, {args...});
                      });
                  };

                  convert(src_rho_f, dst_rho_f);
                  convert(src_etot_f, dst_etot_f);
                  for (int i = 0; i < 3; ++i)
                      convert(src_rhov_f(static_cast<core::Component>(i)),
                              dst_rhov_f(static_cast<core::Component>(i)));
              };

        convert_hydro_fluxes.template operator()<Direction::X>(
            pv.tmpFluxes_.rho_fx, pv.tmpFluxes_.rhoV_fx, pv.tmpFluxes_.Etot_fx, fluxes.rho_fx,
            fluxes.rhoV_fx, fluxes.Etot_fx);
        if constexpr (dimension >= 2)
        {
            convert_hydro_fluxes.template operator()<Direction::Y>(
                pv.tmpFluxes_.rho_fy, pv.tmpFluxes_.rhoV_fy, pv.tmpFluxes_.Etot_fy, fluxes.rho_fy,
                fluxes.rhoV_fy, fluxes.Etot_fy);

            if constexpr (dimension == 3)
                convert_hydro_fluxes.template operator()<Direction::Z>(
                    pv.tmpFluxes_.rho_fz, pv.tmpFluxes_.rhoV_fz, pv.tmpFluxes_.Etot_fz,
                    fluxes.rho_fz, fluxes.rhoV_fz, fluxes.Etot_fz);
        }

        auto convert_edge = [&]<auto dir_tag>(auto const& src, auto& dst) {
            layout_.evalOnBox(dst, [&](auto&... args) mutable {
                edge_center_conversion_<dir_tag, toAverage>(src, dst, {args...});
            });
        };

        convert_edge.template operator()<Direction::X>(pv.tmpE_(core::Component::X),
                                                       E(core::Component::X));
        convert_edge.template operator()<Direction::Y>(pv.tmpE_(core::Component::Y),
                                                       E(core::Component::Y));
        convert_edge.template operator()<Direction::Z>(pv.tmpE_(core::Component::Z),
                                                       E(core::Component::Z));
    }

private:
    template<PointValueConversionMode mode>
    void cell_center_conversion_(auto const& f, auto& fnew, MeshIndex<dimension> index) const
    {
        fnew(index) = getCellCentered<mode>(f, index);
    }

    template<auto direction, PointValueConversionMode mode>
    void face_center_conversion_(auto const& f, auto& fnew, MeshIndex<dimension> index) const
    {
        fnew(index) = getFaceCentered<direction, mode>(f, index);
    }

    template<auto direction, PointValueConversionMode mode>
    void edge_center_conversion_(auto const& f, auto& fnew, MeshIndex<dimension> index) const
    {
        fnew(index) = getEdgeCentered<direction, mode>(f, index);
    }

    template<PointValueConversionMode mode, typename Field>
    auto getCellCentered(Field const& f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl
            = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;

        auto const theta = limit_(index);
        return (theta != 0.0) ? f(index) + layout_.lapl(f, index) * wlapl : f(index);
    }

    template<auto direction, PointValueConversionMode mode, typename Field>
    auto getFaceCentered(Field const& f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl
            = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
        auto const theta = limit_face_<direction>(index);

        return (theta != 0.0)
                   ? f(index) + layout_.template tranverseLapl<direction>(f, index) * wlapl
                   : f(index);
    }

    template<auto direction, PointValueConversionMode mode, typename Field>
    auto getEdgeCentered(Field const& f, MeshIndex<dimension> index) const
    {
        constexpr bool has_direction = (dimension == 3)
                                       || (dimension == 2 && direction != Direction::Z)
                                       || (dimension == 1 && direction == Direction::X);

        if constexpr (!has_direction)
        {
            return f(index);
        }
        else
        {
            static constexpr auto wlapl
                = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
            auto const theta = limit_edge_<direction>(index);

            return (theta != 0.0)
                       ? f(index) + layout_.template directionalLapl<direction>(f, index) * wlapl
                       : f(index);
        }
    }

    template<auto direction>
    auto limit_face_(MeshIndex<dimension> index) const
    {
        constexpr bool has_next = (dimension == 3) || (dimension == 2 && direction != Direction::Z)
                                  || (dimension == 1 && direction == Direction::X);

        if constexpr (has_next)
        {
            return std::min(limit_(index), limit_(layout_.template next<direction>(index)));
        }
        else
        {
            return limit_(index);
        }
    }

    template<auto direction>
    auto limit_edge_(MeshIndex<dimension> index) const
    {
        if constexpr (dimension == 1)
        {
            static_assert(direction != Direction::Y && direction != Direction::Z
                          && "PointValueHandler::limit_edge_ forbidden direction in 1D");
            return limit_(index);
        }
        else if constexpr (dimension == 2)
        {
            static_assert(direction != Direction::Z
                          && "PointValueHandler::limit_edge_ forbidden direction in 2D");
            static constexpr auto perp_dir
                = (direction == Direction::X) ? Direction::Y : Direction::X;
            return std::min(limit_(index), limit_(layout_.template next<perp_dir>(index)));
        }
        else
        { // dimension == 3
            static constexpr auto d1 = (direction == Direction::X) ? Direction::Y : Direction::X;
            static constexpr auto d2 = (direction == Direction::Z) ? Direction::Y : Direction::Z;

            return std::min({limit_(index), limit_(layout_.template next<d1>(index)),
                             limit_(layout_.template next<d2>(index)),
                             limit_(layout_.template next<d1>(layout_.template next<d2>(index)))});
        }
    }

    auto limit_(MeshIndex<dimension>) const
    {
        return 1.; // not limiter yet
    }

    GridLayout layout_;
};
} // namespace PHARE::core

#endif
