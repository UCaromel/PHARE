#pragma once

#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/BoxLevel.h>
#include <SAMRAI/hier/BoxLevelConnectorUtils.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/tbox/MemoryDatabase.h>
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

#include "tests/core/data/mhd_state/init_functions.hpp"
#include "tests/core/numerics/convergence/exact_solutions.hpp"


using namespace PHARE::core;

constexpr double k = 2.0 * M_PI;

inline double f(double x) { return std::sin(k * x); }
inline double df(double x) { return k * std::cos(k * x); }
inline double f3d(double x, double y, double z)
{
    return std::sin(k * x) * std::sin(k * y) * std::sin(k * z);
}
inline double d2f3d_dx2(double x, double y, double z) { return -k * k * f3d(x, y, z); }

inline double Bx_func(double x, double y, double z) { return std::sin(k * y) * std::sin(k * z); }
inline double By_func(double x, double y, double z) { return std::sin(k * x) * std::sin(k * z); }
inline double Bz_func(double x, double y, double z) { return std::sin(k * x) * std::sin(k * y); }

inline double Jx_exact(double x, double y, double z)
{
    return k * std::cos(k * y) * std::sin(k * x) - k * std::cos(k * z) * std::sin(k * x);
}
inline double Jy_exact(double x, double y, double z)
{
    return k * std::cos(k * z) * std::sin(k * y) - k * std::cos(k * x) * std::sin(k * y);
}
inline double Jz_exact(double x, double y, double z)
{
    return k * std::cos(k * x) * std::sin(k * z) - k * std::cos(k * y) * std::sin(k * z);
}

inline double convergenceOrder(double err_coarse, double err_fine, double ratio = 2.0)
{
    return std::log(err_coarse / err_fine) / std::log(ratio);
}

template<std::size_t dim>
using InitFn = PHARE::initializer::InitFunction<dim>;

inline std::shared_ptr<PHARE::core::Span<double>> constantOne(std::vector<double> const& x)
{
    return std::make_shared<PHARE::core::VectorSpan<double>>(x.size(), 1.0);
}

inline std::shared_ptr<PHARE::core::Span<double>> constantZero(std::vector<double> const& x)
{
    return std::make_shared<PHARE::core::VectorSpan<double>>(x.size(), 0.0);
}

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


template<typename Layout, typename ResourcesManagerT>
struct Hall3DPeriodicGhostFiller
{
    using level_t = PHARE::amr::SAMRAI_Types::level_t;

    Hall3DPeriodicGhostFiller(Layout const& layout, ResourcesManagerT& resourcesManager)
        : layout_{layout}
        , resourcesManager_{resourcesManager}
    {
    }

    // no point-value fill methods: point-value quantities are local derived quantities computed
    // on shrunk ghost boxes from the (periodically filled) average ghosts — ComputeFluxes only
    // needs the electric ghost fill from its `bc`.

    template<typename VecField>
    void fillElectricGhosts(VecField& E, level_t const& level, double) const
    {
        for (auto const& patch : level)
        {
            auto guard = resourcesManager_.setOnPatch(*patch, E);
            periodicFillGhostsVec(layout_, E);
        }
    }

private:
    Layout const& layout_;
    ResourcesManagerT& resourcesManager_;
};

// The oracle lives in exact_solutions.hpp, which carries the same formulas at the same k with
// flux()/electric() additionally able to drop the Hall terms. Pulled into the global namespace
// because this header's users refer to it unqualified.
using PHARE::test::ExactHall3D;

inline PHARE::initializer::PHAREDict makeHall3DMHDModelDict()
{
    using namespace PHARE::initializer;

    auto scalar_from_xyz = [](auto&& fn) -> InitFn<3> {
        return [f = std::forward<decltype(fn)>(fn)](std::vector<double> const& x,
                                                    std::vector<double> const& y,
                                                    std::vector<double> const& z) {
            std::vector<double> vals(x.size());
            for (std::size_t i = 0; i < vals.size(); ++i)
                vals[i] = f(x[i], y[i], z[i]);
            return std::make_shared<PHARE::core::VectorSpan<double>>(std::move(vals));
        };
    };

    double const gamma = 1.4;

    InitFn<3> rho = scalar_from_xyz(ExactHall3D::rho);
    InitFn<3> vx  = scalar_from_xyz(ExactHall3D::vx);
    InitFn<3> vy  = scalar_from_xyz(ExactHall3D::vy);
    InitFn<3> vz  = scalar_from_xyz(ExactHall3D::vz);
    InitFn<3> bx  = scalar_from_xyz(ExactHall3D::bx);
    InitFn<3> by  = scalar_from_xyz(ExactHall3D::by);
    InitFn<3> bz  = scalar_from_xyz(ExactHall3D::bz);
    InitFn<3> p   = scalar_from_xyz(ExactHall3D::pressure);


    PHARE::initializer::PHAREDict state;
    state["name"] = std::string{"hall_state"};
    state["density"]["initializer"] = rho;
    state["rhoV"]["initializer"]["x_component"] = mulInit<3>(rho, vx);
    state["rhoV"]["initializer"]["y_component"] = mulInit<3>(rho, vy);
    state["rhoV"]["initializer"]["z_component"] = mulInit<3>(rho, vz);
    state["magnetic"]["initializer"]["x_component"] = bx;
    state["magnetic"]["initializer"]["y_component"] = by;
    state["magnetic"]["initializer"]["z_component"] = bz;
    state["Etot"]["initializer"] = etotInit<3>(gamma, rho, vx, vy, vz, bx, by, bz, p);

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
    dict["fv_method"]["hyper_mode"]          = std::string{"constant"};

    dict["constrained_transport"]["resistivity"]       = 0.0;
    dict["constrained_transport"]["hyper_resistivity"] = 0.0;
    dict["constrained_transport"]["hyper_mode"]        = std::string{"constant"};

    dict["to_primitive"]["heat_capacity_ratio"]    = 1.4;
    dict["to_conservative"]["heat_capacity_ratio"] = 1.4;
    return dict;
}

inline std::shared_ptr<SAMRAI::hier::PatchHierarchy> makePeriodicHierarchy3D(int nCells)
{
    auto const dim = SAMRAI::tbox::Dimension{3};

    auto geomDB = std::make_shared<SAMRAI::tbox::MemoryDatabase>("Hall3DGeomDB");
    int lower[3] = {0, 0, 0};
    int upper[3] = {nCells - 1, nCells - 1, nCells - 1};
    std::vector<SAMRAI::tbox::DatabaseBox> dbBoxes;
    dbBoxes.emplace_back(dim, lower, upper);
    geomDB->putDatabaseBoxVector("domain_boxes", dbBoxes);

    double x_lo[3] = {0.0, 0.0, 0.0};
    double x_up[3] = {1.0, 1.0, 1.0};
    geomDB->putDoubleArray("x_lo", x_lo, 3);
    geomDB->putDoubleArray("x_up", x_up, 3);

    int periodicity[3] = {1, 1, 1};
    geomDB->putIntegerArray("periodic_dimension", periodicity, 3);

    auto hierDB = std::make_shared<SAMRAI::tbox::MemoryDatabase>("Hall3DHierarchyDB");
    hierDB->putInteger("max_levels", 1);

    auto geometry = std::make_shared<SAMRAI::geom::CartesianGridGeometry>(dim, "Hall3DGeom", geomDB);
    auto hierarchy = std::make_shared<SAMRAI::hier::PatchHierarchy>("Hall3DHierarchy", geometry, hierDB);

    SAMRAI::hier::Box domain{dim};
    static int boxCounter = 0;
    auto const rank = SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld().getRank();
    auto const gid  = SAMRAI::hier::GlobalId{SAMRAI::hier::LocalId{boxCounter++}, rank};
    domain.setBlockId(SAMRAI::hier::BlockId{0});
    domain.setId(SAMRAI::hier::BoxId{gid});
    domain.setLower(SAMRAI::hier::Index{std::vector<int>{0, 0, 0}});
    domain.setUpper(SAMRAI::hier::Index{std::vector<int>{nCells - 1, nCells - 1, nCells - 1}});

    SAMRAI::hier::BoxContainer levelBoxes;
    levelBoxes.push_back(domain);
    SAMRAI::hier::BoxLevel level0{levelBoxes, SAMRAI::hier::IntVector::getOne(dim), geometry};

    // A hand-built BoxLevel carries no periodic image boxes: "periodic_dimension" only sets up
    // the grid geometry's shift catalog, and the sole place SAMRAI populates the coarsest
    // level's images from it is mesh::GriddingAlgorithm::makeCoarsestLevel(), through
    // hier::BoxLevelConnectorUtils::addPeriodicImages() -- the path a real hierarchy takes and
    // this fixture bypasses. Without the images, every dst<->src Connector a RefineSchedule
    // builds on this level sees one isolated patch, so the entire ghost layer comes back
    // unfilled and RefineSchedule escalates it to a coarser level that does not exist.
    // Add them here, exactly as makeCoarsestLevel() does, so periodicity is real for
    // communication and not only for the geometry.
    SAMRAI::hier::BoxLevelConnectorUtils blcu;
    blcu.addPeriodicImages(level0, geometry->getDomainSearchTree(),
                           SAMRAI::hier::IntVector::max(
                               hierarchy->getRequiredConnectorWidth(0, 0, true),
                               SAMRAI::hier::IntVector::getOne(dim)));

    hierarchy->makeNewPatchLevel(0, level0);

    return hierarchy;
}

template<typename MHDModelT>
struct HallFVMethod3D
{
    // Godunov takes a Reconstruction template of exactly one parameter.
    // PointValueWENOZReconstruction carries a defaulted SlopeLimiter, and binding a
    // two-parameter template to a one-parameter template template parameter is P0522 relaxed
    // matching -- applied by default by GCC, not by Clang. Wrap it down to one parameter, the
    // same way the solver does in MHDResolver::Reconstruction_t.
    template<typename GridLayoutT>
    using Reconstruction = PointValueWENOZReconstruction<GridLayoutT>;

    template<typename GridLayoutT>
    using type
        = Godunov<GridLayoutT, Reconstruction, Rusanov<true>, MHDEquations<true, false, false>>;
};
