#pragma once

#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "amr/physical_models/mhd_model.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/types/amr_types.hpp"
#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_impl.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/numerics/MHD_equations/MHD_equations.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/span.hpp"

#include "tests/amr/solvers/periodic_hierarchy.hpp"
#include "tests/core/data/mhd_state/init_functions.hpp"
#include "tests/amr/solvers/convergence/exact_solutions.hpp"


using namespace PHARE::core;

inline double convergenceOrder(double err_coarse, double err_fine, double ratio = 2.0)
{
    return std::log(err_coarse / err_fine) / std::log(ratio);
}

template<std::size_t dim>
using InitFn = PHARE::initializer::InitFunction<dim>;

template<typename Layout, typename Field>
inline void fillUsableField(Layout const& layout, Field& field, auto&& fn)
{
    auto cent = layout.centering(field.physicalQuantity());
    for (auto i = layout.ghostStartIndex(cent[0], Direction::X);
         i <= layout.ghostEndIndex(cent[0], Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], Direction::Y);
             j <= layout.ghostEndIndex(cent[1], Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], Direction::Z); ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                field(i, j, kk) = fn(c[0], c[1], c[2]);
            }
}

template<typename Layout, typename Field>
inline void periodicFillGhosts(Layout const& layout, Field& field)
{
    auto cent = layout.centering(field.physicalQuantity());

    auto psiX = layout.physicalStartIndex(cent[0], Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], Direction::Z);

    auto nX = peiX - psiX + 1;
    auto nY = peiY - psiY + 1;
    auto nZ = peiZ - psiZ + 1;

    auto wrap = [](int idx, int lo, int n) {
        int r = (idx - lo) % n;
        if (r < 0)
            r += n;
        return lo + r;
    };

    for (auto i = layout.ghostStartIndex(cent[0], Direction::X);
         i <= layout.ghostEndIndex(cent[0], Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], Direction::Y);
             j <= layout.ghostEndIndex(cent[1], Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], Direction::Z); ++kk)
            {
                auto iw = wrap(static_cast<int>(i), static_cast<int>(psiX), static_cast<int>(nX));
                auto jw = wrap(static_cast<int>(j), static_cast<int>(psiY), static_cast<int>(nY));
                auto kw = wrap(static_cast<int>(kk), static_cast<int>(psiZ), static_cast<int>(nZ));
                field(i, j, kk) = field(iw, jw, kw);
            }
}

template<typename Layout, typename VecField>
inline void periodicFillGhostsVec(Layout const& layout, VecField& vecField)
{
    periodicFillGhosts(layout, vecField(Component::X));
    periodicFillGhosts(layout, vecField(Component::Y));
    periodicFillGhosts(layout, vecField(Component::Z));
}

template<typename Layout, typename Fluxes>
inline void fillFluxGhosts(Layout const& layout, Fluxes& fluxes)
{
    periodicFillGhosts(layout, fluxes.rho_fx);
    periodicFillGhostsVec(layout, fluxes.rhoV_fx);
    periodicFillGhosts(layout, fluxes.Etot_fx);
    periodicFillGhosts(layout, fluxes.B_fx(Component::X));
    periodicFillGhosts(layout, fluxes.B_fx(Component::Y));
    periodicFillGhosts(layout, fluxes.B_fx(Component::Z));

    periodicFillGhosts(layout, fluxes.rho_fy);
    periodicFillGhostsVec(layout, fluxes.rhoV_fy);
    periodicFillGhosts(layout, fluxes.Etot_fy);
    periodicFillGhosts(layout, fluxes.B_fy(Component::X));
    periodicFillGhosts(layout, fluxes.B_fy(Component::Y));
    periodicFillGhosts(layout, fluxes.B_fy(Component::Z));

    periodicFillGhosts(layout, fluxes.rho_fz);
    periodicFillGhostsVec(layout, fluxes.rhoV_fz);
    periodicFillGhosts(layout, fluxes.Etot_fz);
    periodicFillGhosts(layout, fluxes.B_fz(Component::X));
    periodicFillGhosts(layout, fluxes.B_fz(Component::Y));
    periodicFillGhosts(layout, fluxes.B_fz(Component::Z));
}


// The oracle lives in exact_solutions.hpp, which carries the same formulas at the same k with
// flux()/electric() additionally able to drop the Hall terms. Pulled into the global namespace
// because this header's users refer to it unqualified.
using PHARE::test::ExactHall3D;

inline PHARE::initializer::PHAREDict makeHall3DMHDModelDict()
{
    using namespace PHARE::initializer;

    auto scalar_from_xyz = [](auto&& fn) -> InitFn<3> {
        return [f = std::forward<decltype(fn)>(fn)](PHARE::core::Span<double const> const& x,
                                                    PHARE::core::Span<double const> const& y,
                                                    PHARE::core::Span<double const> const& z) {
            std::vector<double> vals(x.size());
            for (std::size_t i = 0; i < vals.size(); ++i)
                vals[i] = f(x[i], y[i], z[i]);
            return std::make_shared<PHARE::core::VectorSpan<double>>(std::move(vals));
        };
    };

    auto const momentum = [](auto&& v) {
        return [v](double x, double y, double z) { return ExactHall3D::rho(x, y, z) * v(x, y, z); };
    };

    PHARE::initializer::PHAREDict state;
    state["name"]                                   = std::string{"hall_state"};
    state["density"]["initializer"]                 = scalar_from_xyz(ExactHall3D::rho);
    state["rhoV"]["initializer"]["x_component"]     = scalar_from_xyz(momentum(ExactHall3D::vx));
    state["rhoV"]["initializer"]["y_component"]     = scalar_from_xyz(momentum(ExactHall3D::vy));
    state["rhoV"]["initializer"]["z_component"]     = scalar_from_xyz(momentum(ExactHall3D::vz));
    state["magnetic"]["initializer"]["x_component"] = scalar_from_xyz(ExactHall3D::bx);
    state["magnetic"]["initializer"]["y_component"] = scalar_from_xyz(ExactHall3D::by);
    state["magnetic"]["initializer"]["z_component"] = scalar_from_xyz(ExactHall3D::bz);
    state["Etot"]["initializer"]                    = scalar_from_xyz(ExactHall3D::etot);

    PHARE::initializer::PHAREDict model;
    model["mhd_state"] = state;
    return model;
}

inline PHARE::initializer::PHAREDict makeHall3DComputeFluxDict()
{
    PHARE::initializer::PHAREDict dict;

    dict["fv_method"]["heat_capacity_ratio"] = 1.4;
    dict["fv_method"]["resistivity"]         = 0.0;
    dict["fv_method"]["hyper_resistivity"]   = 0.0;

    dict["constrained_transport"]["resistivity"]       = 0.0;
    dict["constrained_transport"]["hyper_resistivity"] = 0.0;

    dict["to_primitive"]["heat_capacity_ratio"] = 1.4;
    return dict;
}

template<typename MHDModelT>
struct HallFVMethod3D
{
    // Godunov takes a one-parameter Reconstruction template, as MHDResolver::Reconstruction_t
    template<typename GridLayoutT>
    using Reconstruction = WENOZReconstruction<GridLayoutT, void, true>;

    template<typename GridLayoutT>
    using type = Godunov<GridLayoutT, Reconstruction, Rusanov<true>, MHDEquations<true>>;
};
