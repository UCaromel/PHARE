#ifndef PHARE_CORE_NUMERICS_POINT_VALUE_HANDLER_HPP
#define PHARE_CORE_NUMERICS_POINT_VALUE_HANDLER_HPP

#include "core/data/grid/gridlayout_utils.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"

#include <strings.h>

namespace PHARE::core
{
template<typename GridLayout, typename MHDModel>
class PointValueHandler : public LayoutHolder<GridLayout>
{
    constexpr static auto dimension = GridLayout::dimension;
    using LayoutHolder<GridLayout>::layout_;

    using Field_t    = MHDModel::field_type;
    using VecField_t = MHDModel::vecfield_type;

    enum class ConversionMode { ToPointValue, ToAverage };

    // flux laplacian computations cannot be inplace.
    core::AllFluxes<Field_t, VecField_t> tmpFluxes_{
        {"pvh_rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
        {"pvh_rhoV_fx", MHDQuantity::Vector::VecFlux_x},
        {"pvh_B_fx", MHDQuantity::Vector::VecFlux_x},
        {"pvh_Etot_fx", MHDQuantity::Scalar::ScalarFlux_x},

        {"pvh_rho_fy", MHDQuantity::Scalar::ScalarFlux_y},
        {"pvh_rhoV_fy", MHDQuantity::Vector::VecFlux_y},
        {"pvh_B_fy", MHDQuantity::Vector::VecFlux_y},
        {"pvh_Etot_fy", MHDQuantity::Scalar::ScalarFlux_y},

        {"pvh_rho_fz", MHDQuantity::Scalar::ScalarFlux_z},
        {"pvh_rhoV_fz", MHDQuantity::Vector::VecFlux_z},
        {"pvh_B_fz", MHDQuantity::Vector::VecFlux_z},
        {"pvh_Etot_fz", MHDQuantity::Scalar::ScalarFlux_z}};

    VecField_t E_{"pvh_E", MHDQuantity::Vector::E};

public:
    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(rho);
        model.resourcesManager->registerResources(V);
        model.resourcesManager->registerResources(B);
        model.resourcesManager->registerResources(P);

        model.resourcesManager->registerResources(rhoV);
        model.resourcesManager->registerResources(Etot);

        model.resourcesManager->registerResources(J);

        model.resourcesManager->registerResources(tmpFluxes_);
        model.resourcesManager->registerResources(E_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(rho, patch, allocateTime);
        model.resourcesManager->allocate(V, patch, allocateTime);
        model.resourcesManager->allocate(B, patch, allocateTime);
        model.resourcesManager->allocate(P, patch, allocateTime);

        model.resourcesManager->allocate(rhoV, patch, allocateTime);
        model.resourcesManager->allocate(Etot, patch, allocateTime);

        model.resourcesManager->allocate(J, patch, allocateTime);

        model.resourcesManager->allocate(tmpFluxes_, patch, allocateTime);
        model.resourcesManager->allocate(E_, patch, allocateTime);
    }

    void fillMessengerInfo(auto& info) const
    {
        info.pointDensity  = rho.name();
        info.pointVelocity = V.name();
        info.pointMagnetic = B.name();
        info.pointPressure = P.name();

        info.pointCurrent = J.name();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, tmpFluxes_, E_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, tmpFluxes_, E_);
    }

    // here the V and P buffers are used for both primitive and conserved. The main reason is that
    // the pointwise conserved quantities are only computed to immediatly after get the primitives,
    // which are the ones used in the computations.
    Field_t rho{"point_value_rho", MHDQuantity::Scalar::rho};
    VecField_t V{"point_value_V", MHDQuantity::Vector::V};
    VecField_t B{"point_value_B", MHDQuantity::Vector::B};
    Field_t P{"point_value_P", MHDQuantity::Scalar::P};

    VecField_t J{"point_value_J", MHDQuantity::Vector::J};

    // not same buffers as primitive yet, but likely could be
    VecField_t rhoV{"point_value_rhoV", MHDQuantity::Vector::rhoV};
    Field_t Etot{"point_value_Etot", MHDQuantity::Scalar::Etot};

    void operator()(auto const& state)
    {
        static constexpr auto toPointValue = ConversionMode::ToPointValue;

        if (!this->hasLayout())
            throw std::runtime_error("Error - PointValueHandler - GridLayout not set");

        auto convert_cell = [&](auto const& src, auto& dst) {
            layout_->evalOnBox(src, [&](auto&... args) mutable {
                cell_center_conversion_<toPointValue>(src, dst, {args...});
            });
        };

        convert_cell(state.rho, rho);
        convert_cell(state.Etot, Etot);
        for (int i = 0; i < 3; ++i)
        {
            convert_cell(state.rhoV(static_cast<core::Component>(i)),
                         rhoV(static_cast<core::Component>(i)));
        }

        auto convert_face = [&]<auto dir_tag>(auto const& src, auto& dst) {
            layout_->evalOnBox(src, [&](auto&... args) mutable {
                face_center_conversion_<dir_tag, toPointValue>(src, dst, {args...});
            });
        };

        convert_face.template operator()<Direction::X>(state.B(core::Component::X),
                                                       B(core::Component::X));
        convert_face.template operator()<Direction::Y>(state.B(core::Component::Y),
                                                       B(core::Component::Y));
        convert_face.template operator()<Direction::Z>(state.B(core::Component::Z),
                                                       B(core::Component::Z));
    }

    void point_value_fluxes_to_integral(auto& fluxes, auto& E)
    {
        static constexpr auto toAverage = ConversionMode::ToAverage;

        E_.copyData(E);
        tmpFluxes_.copyData(fluxes);

        auto convert_hydro_fluxes
            = [&]<auto dir_tag>(auto& src_rho_f, auto& src_rhov_f, auto& src_etot_f,
                                auto& dst_rho_f, auto& dst_rhov_f, auto& dst_etot_f) {
                  auto convert = [&](auto const& src, auto& dst) {
                      layout_->evalOnBox(dst, [&](auto&... args) mutable {
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
            tmpFluxes_.rho_fx, tmpFluxes_.rhoV_fx, tmpFluxes_.Etot_fx, fluxes.rho_fx,
            fluxes.rhoV_fx, fluxes.Etot_fx);
        if constexpr (dimension >= 2)
        {
            convert_hydro_fluxes.template operator()<Direction::Y>(
                tmpFluxes_.rho_fy, tmpFluxes_.rhoV_fy, tmpFluxes_.Etot_fy, fluxes.rho_fy,
                fluxes.rhoV_fy, fluxes.Etot_fy);

            if constexpr (dimension == 3)
                convert_hydro_fluxes.template operator()<Direction::Z>(
                    tmpFluxes_.rho_fz, tmpFluxes_.rhoV_fz, tmpFluxes_.Etot_fz, fluxes.rho_fz,
                    fluxes.rhoV_fz, fluxes.Etot_fz);
        }

        auto convert_edge = [&]<auto dir_tag>(auto const& src, auto& dst) {
            layout_->evalOnBox(dst, [&](auto&... args) mutable {
                edge_center_conversion_<dir_tag, toAverage>(src, dst, {args...});
            });
        };

        convert_edge.template operator()<Direction::X>(E_(core::Component::X),
                                                       E(core::Component::X));
        convert_edge.template operator()<Direction::Y>(E_(core::Component::Y),
                                                       E(core::Component::Y));
        convert_edge.template operator()<Direction::Z>(E_(core::Component::Z),
                                                       E(core::Component::Z));
    }

    // tbd if we want a cell centered version to save the cost of projection (used in the conversion
    // to primitive and the godunov reconstruction step, also possibly better to have less ghosts,
    // because godunov need to handle centering and reconstruction also on ghosts, which requires
    // additionnal points because of the centering stencil. That being said this might be a bad
    // idea, as well would need to fill ghosts in a way that conserves divB on this cell centered B,
    // which could be tricky. also possible to actually just compute it on a reduced ghost box.

    //  VecField_t B_cc;

private:
    template<ConversionMode mode>
    auto cell_center_conversion_(Field_t const f, Field_t fnew, MeshIndex<dimension> index)
    {
        fnew(index) = get_cell_center_<mode>(f, index);
    }

    template<ConversionMode mode>
    auto cell_center_conversion_(Field_t const f, Field_t fnew, MeshIndex<dimension> index) const
    {
        fnew(index) = get_cell_center_<mode>(f, index);
    }

    template<ConversionMode mode>
    auto get_cell_center_(Field_t const f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl = (mode == ConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;

        auto const theta = limit_(index);

        return (theta != 0.0) ? f(index) + layout_->lapl(f, index) * wlapl : f(index);
    }

    template<auto direction, ConversionMode mode>
    auto face_center_conversion_(Field_t const f, Field_t fnew, MeshIndex<dimension> index)
    {
        fnew(index) = get_face_center_<direction, mode>(f, index);
    }

    template<auto direction, ConversionMode mode>
    auto face_center_conversion_(Field_t const f, Field_t fnew, MeshIndex<dimension> index) const
    {
        fnew(index) = get_face_center_<direction, mode>(f, index);
    }

    template<auto direction, ConversionMode mode>
    auto get_face_center_(Field_t const f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl = (mode == ConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
        auto const theta            = limit_face_<direction>(index);

        return (theta != 0.0)
                   ? f(index) + layout_->template tranverseLapl<direction>(f, index) * wlapl
                   : f(index);
    }

    template<auto direction, ConversionMode mode>
    auto edge_center_conversion_(Field_t const f, Field_t fnew, MeshIndex<dimension> index)
    {
        fnew(index) = get_edge_center_<direction, mode>(f, index);
    }

    template<auto direction, ConversionMode mode>
    auto edge_center_conversion_(Field_t const f, Field_t fnew, MeshIndex<dimension> index) const
    {
        fnew(index) = get_edge_center_<direction, mode>(f, index);
    }

    template<auto direction, ConversionMode mode>
    auto get_edge_center_(Field_t const f, MeshIndex<dimension> index) const
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
                = (mode == ConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
            auto const theta = limit_edge_<direction>(index);

            return (theta != 0.0)
                       ? f(index) + layout_->template directionalLapl<direction>(f, index) * wlapl
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
            return std::min(limit_(index), limit_(layout_->template next<direction>(index)));
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
            return std::min(limit_(index), limit_(layout_->template next<perp_dir>(index)));
        }
        else
        { // dimension == 3
            static constexpr auto d1 = (direction == Direction::X) ? Direction::Y : Direction::X;
            static constexpr auto d2 = (direction == Direction::Z) ? Direction::Y : Direction::Z;

            return std::min(
                {limit_(index), limit_(layout_->template next<d1>(index)),
                 limit_(layout_->template next<d2>(index)),
                 limit_(layout_->template next<d1>(layout_->template next<d2>(index)))});
        }
    }

    auto limit_(MeshIndex<dimension> index) const
    {
        return 1.; // not limiter yet
    }
};
} // namespace PHARE::core

#endif
