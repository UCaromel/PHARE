#ifndef PHARE_CORE_NUMERICS_POINT_VALUE_HANDLER_UTILS_HPP
#define PHARE_CORE_NUMERICS_POINT_VALUE_HANDLER_UTILS_HPP

#include "core/data/field/field.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"

#include <tuple>

namespace PHARE::core
{
// Bare resource container for the point-value flux pipeline, mirroring GodunovState /
// UpwindConstrainedTransportState. Held by ComputeFluxes and passed by reference to the
// layout-taking PointValueHandler core.
template<typename VecField>
class PointValueState
{
    using Field = typename VecField::field_type;

public:
    PointValueState() = default;

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, tmpFluxes_, tmpE_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, tmpFluxes_, tmpE_);
    }

    // no fillMessengerInfo: point-value quantities are local derived quantities, computed on
    // shrunk ghost boxes from average ghosts — they are never communicated (no refine schedules).

    // here the V and P buffers are used for both primitive and conserved. The main reason is that
    // the pointwise conserved quantities are only computed immediately to get the primitives, which
    // are the ones used in the computations.
    Field rho{"point_value_rho", MHDQuantity::Scalar::rho};
    VecField V{"point_value_V", MHDQuantity::Vector::V};
    VecField B{"point_value_B", MHDQuantity::Vector::B};
    Field P{"point_value_P", MHDQuantity::Scalar::P};

    VecField J{"point_value_J", MHDQuantity::Vector::J};

    // not same buffers as primitive yet, but likely could be
    VecField rhoV{"point_value_rhoV", MHDQuantity::Vector::rhoV};
    Field Etot{"point_value_Etot", MHDQuantity::Scalar::Etot};

    // flux/electric laplacian computations cannot be inplace; tmpFluxes_/tmpE_ are scratch holding
    // the point-value sources while the average is written back into the passed buffers.
    AllFluxes<Field, VecField> tmpFluxes_{
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

    VecField tmpE_{"pvh_E", MHDQuantity::Vector::E};
};

} // namespace PHARE::core

#endif
