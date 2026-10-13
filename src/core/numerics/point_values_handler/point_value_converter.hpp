#ifndef PHARE_CORE_NUMERICS_POINT_VALUE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_POINT_VALUE_CONVERTER_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"

#include <cstdint>

namespace PHARE::core
{
enum class PointValueConversionMode { ToPointValue, ToAverage };

inline constexpr std::uint32_t point_value_conversion_shrink = 1;

template<typename GridLayout>
class PointValueConverter
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    explicit PointValueConverter(GridLayout const& layout)
        : layout_{layout}
    {
    }

    void toPointValues(auto& pv, auto const& state)
    {
        static constexpr auto toPointValue = PointValueConversionMode::ToPointValue;
        static constexpr auto shrink       = point_value_conversion_shrink;

        auto convert_cell = [&](auto const& src, auto& dst) {
            Point<std::uint32_t, dimension> amount;
            for (std::size_t i = 0; i < dimension; ++i)
                amount[i] = shrink;

            layout_.evalOnShrinkedGhostBox(src, amount, [&](auto&... args) mutable {
                dst(args...) = getCellCentered<toPointValue>(src, {args...});
            });
        };

        convert_cell(state.rho, pv.rho);
        convert_cell(state.Etot, pv.Etot);
        for (int i = 0; i < 3; ++i)
            convert_cell(state.rhoV(static_cast<core::Component>(i)),
                         pv.rhoV(static_cast<core::Component>(i)));

        auto convert_face = [&]<auto dir_tag>(auto const& src, auto& dst) {
            Point<std::uint32_t, dimension> amount;
            for (std::size_t i = 0; i < dimension; ++i)
                amount[i] = (i == static_cast<std::size_t>(dir_tag)) ? 0u : shrink;

            layout_.evalOnShrinkedGhostBox(src, amount, [&](auto&... args) mutable {
                dst(args...) = getFaceCentered<dir_tag, toPointValue>(src, {args...});
            });
        };

        convert_face.template operator()<Direction::X>(state.B(core::Component::X),
                                                       pv.B(core::Component::X));
        convert_face.template operator()<Direction::Y>(state.B(core::Component::Y),
                                                       pv.B(core::Component::Y));
        convert_face.template operator()<Direction::Z>(state.B(core::Component::Z),
                                                       pv.B(core::Component::Z));
    }

    void fluxesToAverages(auto& pv, auto& fluxes, auto& E)
    {
        static constexpr auto toAverage = PointValueConversionMode::ToAverage;

        pv.tmpE.copyData(E);
        pv.tmpFluxes.copyData(fluxes);

        auto convert_hydro_fluxes
            = [&]<auto dir_tag>(auto& src_rho_f, auto& src_rhov_f, auto& src_etot_f,
                                auto& dst_rho_f, auto& dst_rhov_f, auto& dst_etot_f) {
                  auto convert = [&](auto const& src, auto& dst) {
                      layout_.evalOnBox(dst, [&](auto&... args) mutable {
                          dst(args...) = getFaceCentered<dir_tag, toAverage>(src, {args...});
                      });
                  };

                  convert(src_rho_f, dst_rho_f);
                  convert(src_etot_f, dst_etot_f);
                  for (int i = 0; i < 3; ++i)
                      convert(src_rhov_f(static_cast<core::Component>(i)),
                              dst_rhov_f(static_cast<core::Component>(i)));
              };

        convert_hydro_fluxes.template operator()<Direction::X>(
            pv.tmpFluxes.rho_fx, pv.tmpFluxes.rhoV_fx, pv.tmpFluxes.Etot_fx, fluxes.rho_fx,
            fluxes.rhoV_fx, fluxes.Etot_fx);
        if constexpr (dimension >= 2)
        {
            convert_hydro_fluxes.template operator()<Direction::Y>(
                pv.tmpFluxes.rho_fy, pv.tmpFluxes.rhoV_fy, pv.tmpFluxes.Etot_fy, fluxes.rho_fy,
                fluxes.rhoV_fy, fluxes.Etot_fy);

            if constexpr (dimension == 3)
                convert_hydro_fluxes.template operator()<Direction::Z>(
                    pv.tmpFluxes.rho_fz, pv.tmpFluxes.rhoV_fz, pv.tmpFluxes.Etot_fz, fluxes.rho_fz,
                    fluxes.rhoV_fz, fluxes.Etot_fz);
        }

        auto convert_edge = [&]<auto dir_tag>(auto const& src, auto& dst) {
            layout_.evalOnBox(dst, [&](auto&... args) mutable {
                dst(args...) = getEdgeCentered<dir_tag, toAverage>(src, {args...});
            });
        };

        convert_edge.template operator()<Direction::X>(pv.tmpE(core::Component::X),
                                                       E(core::Component::X));
        convert_edge.template operator()<Direction::Y>(pv.tmpE(core::Component::Y),
                                                       E(core::Component::Y));
        convert_edge.template operator()<Direction::Z>(pv.tmpE(core::Component::Z),
                                                       E(core::Component::Z));
    }

private:
    template<PointValueConversionMode mode, typename Field>
    auto getCellCentered(Field const& f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl
            = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
        return f(index) + layout_.lapl(f, index) * wlapl;
    }

    template<auto direction, PointValueConversionMode mode, typename Field>
    auto getFaceCentered(Field const& f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl
            = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
        return f(index) + layout_.template transverseLapl<direction>(f, index) * wlapl;
    }

    template<auto direction, PointValueConversionMode mode, typename Field>
    auto getEdgeCentered(Field const& f, MeshIndex<dimension> index) const
    {
        constexpr bool has_direction = (dimension == 3)
                                       || (dimension == 2 && direction != Direction::Z)
                                       || (dimension == 1 && direction == Direction::X);
        if constexpr (!has_direction)
            return f(index);
        else
        {
            static constexpr auto wlapl
                = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
            return f(index) + layout_.template directionalLapl<direction>(f, index) * wlapl;
        }
    }

    GridLayout layout_;
};
} // namespace PHARE::core

#endif
