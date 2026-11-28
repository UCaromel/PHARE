#ifndef PHARE_CORE_NUMERICS_POINT_VALUE_STATE_HPP
#define PHARE_CORE_NUMERICS_POINT_VALUE_STATE_HPP

#include "core/data/field/field.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"

#include <tuple>

namespace PHARE::core
{
template<typename VecField>
class PointValueState
{
    using Field = typename VecField::field_type;

public:
    PointValueState() = default;

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, tmpFluxes, tmpE, troubled, sensed,
                                     magneticEnergy, hydroEnergy);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(rho, V, B, P, rhoV, Etot, J, tmpFluxes, tmpE, troubled, sensed,
                                     magneticEnergy, hydroEnergy);
    }

    Field rho{"point_value_rho", MHDQuantity::Scalar::rho};
    VecField V{"point_value_V", MHDQuantity::Vector::V};
    VecField B{"point_value_B", MHDQuantity::Vector::B};
    Field P{"point_value_P", MHDQuantity::Scalar::P};
    VecField J{"point_value_J", MHDQuantity::Vector::J};
    VecField rhoV{"point_value_rhoV", MHDQuantity::Vector::rhoV};
    Field Etot{"point_value_Etot", MHDQuantity::Scalar::Etot};

    // scratch copies for the flux and EMF conversion back to averages: its stencil reads
    // neighbours, so it cannot work in place
    AllFluxes<Field, VecField> tmpFluxes{{"pvh_rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
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

    VecField tmpE{"pvh_E", MHDQuantity::Vector::E};

    // Conversion limiter (TroubledCellDetector): 1 in the troubled cells, 0 elsewhere, and the
    // detector's scratch. They are cell-centred flags, declared as densities so that a physical
    // boundary mirrors them as it does rho.
    Field troubled{"point_value_troubled", MHDQuantity::Scalar::rho};
    Field sensed{"point_value_sensed", MHDQuantity::Scalar::rho};

    // scratch of the limiter's Hydro form: |B|^2 / 2 at the cell centre and Etot minus its average
    Field magneticEnergy{"point_value_magnetic_energy", MHDQuantity::Scalar::rho};
    Field hydroEnergy{"point_value_hydro_energy", MHDQuantity::Scalar::rho};
};

} // namespace PHARE::core

#endif
