#pragma once

#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/BoxLevel.h>
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
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/numerics/MHD_equations/MHD_equations.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/span.hpp"

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

template<typename Layout>
inline double l2FluxError(Layout const& layout, auto const& field, auto const& exactFn)
{
    auto cent = layout.centering(field.physicalQuantity());
    auto psiX = layout.physicalStartIndex(cent[0], Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], Direction::Z);

    double err = 0.0;
    std::size_t count = 0;
    constexpr int margin = 6;

    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                auto diff = field(i, j, kk) - exactFn(c[0], c[1], c[2]);
                err += diff * diff;
                ++count;
            }
    return std::sqrt(err / static_cast<double>(count));
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

    template<typename VecField>
    void fillCurrentPointGhosts(VecField& J, level_t const& level, double) const
    {
        for (auto const& patch : level)
        {
            auto guard = resourcesManager_.setOnPatch(*patch, J);
            periodicFillGhostsVec(layout_, J);
        }
    }

    template<typename VecField>
    void fillMagneticPointGhosts(VecField& B, level_t const& level, double) const
    {
        for (auto const& patch : level)
        {
            auto guard = resourcesManager_.setOnPatch(*patch, B);
            periodicFillGhostsVec(layout_, B);
        }
    }

    template<typename State>
    void fillPrimitivePointGhosts(State& state, level_t const& level, double) const
    {
        for (auto const& patch : level)
        {
            auto guard = resourcesManager_.setOnPatch(*patch, state);
            periodicFillGhosts(layout_, state.rho);
            periodicFillGhostsVec(layout_, state.V);
            periodicFillGhosts(layout_, state.P);
            periodicFillGhostsVec(layout_, state.J);
            periodicFillGhostsVec(layout_, state.B);
            periodicFillGhostsVec(layout_, state.rhoV);
            periodicFillGhosts(layout_, state.Etot);
        }
    }

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

struct ExactHall3D
{
    static double rho(double x, double y, double z)
    {
        return 1.5 + 0.1 * std::sin(k * x) + 0.07 * std::cos(k * y) + 0.05 * std::sin(k * z);
    }

    static double vx(double x, double y, double z)
    {
        return 0.2 * std::sin(k * x) + 0.1 * std::cos(k * y) + 0.03 * std::sin(k * z);
    }

    static double vy(double x, double y, double z)
    {
        return -0.15 * std::cos(k * y) + 0.08 * std::sin(k * z) + 0.02 * std::cos(k * x);
    }

    static double vz(double x, double y, double z)
    {
        return 0.12 * std::sin(k * z) + 0.06 * std::cos(k * x) + 0.02 * std::sin(k * y);
    }

    static double bx(double, double y, double z) { return 0.3 + 0.1 * std::sin(k * y) * std::sin(k * z); }
    static double by(double x, double, double z) { return 0.2 + 0.1 * std::sin(k * x) * std::sin(k * z); }
    static double bz(double x, double y, double) { return -0.1 + 0.1 * std::sin(k * x) * std::sin(k * y); }

    static double pressure(double x, double y, double z)
    {
        return 1.2 + 0.1 * std::cos(k * (x + y)) + 0.05 * std::sin(k * z);
    }

    static std::array<double, 3> current(double x, double y, double z)
    {
        auto const jx = 0.1 * k * std::sin(k * x) * (std::cos(k * y) - std::cos(k * z));
        auto const jy = 0.1 * k * std::sin(k * y) * (std::cos(k * z) - std::cos(k * x));
        auto const jz = 0.1 * k * std::sin(k * z) * (std::cos(k * x) - std::cos(k * y));
        return {jx, jy, jz};
    }

    static double etot(double x, double y, double z)
    {
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);
        auto const p  = pressure(x, y, z);
        constexpr double gamma = 1.4;
        return p / (gamma - 1.0) + 0.5 * r * (vX * vX + vY * vY + vZ * vZ)
               + 0.5 * (bX * bX + bY * bY + bZ * bZ);
    }

    static auto flux(Direction dir, double x, double y, double z)
    {
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);
        auto const p  = pressure(x, y, z);
        auto const j  = current(x, y, z);
        auto const jX = j[0];
        auto const jY = j[1];
        auto const jZ = j[2];
        auto const eT = etot(x, y, z);

        auto const gp = p + 0.5 * (bX * bX + bY * bY + bZ * bZ);

        double frho = 0.0, frhoVx = 0.0, frhoVy = 0.0, frhoVz = 0.0, fetot = 0.0;
        if (dir == Direction::X)
        {
            frho   = r * vX;
            frhoVx = r * vX * vX + gp - bX * bX;
            frhoVy = r * vX * vY - bX * bY;
            frhoVz = r * vX * vZ - bX * bZ;
            fetot  = (eT + gp) * vX - bX * (vX * bX + vY * bY + vZ * bZ);
        }
        else if (dir == Direction::Y)
        {
            frho   = r * vY;
            frhoVx = r * vY * vX - bY * bX;
            frhoVy = r * vY * vY + gp - bY * bY;
            frhoVz = r * vY * vZ - bY * bZ;
            fetot  = (eT + gp) * vY - bY * (vX * bX + vY * bY + vZ * bZ);
        }
        else
        {
            frho   = r * vZ;
            frhoVx = r * vZ * vX - bZ * bX;
            frhoVy = r * vZ * vY - bZ * bY;
            frhoVz = r * vZ * vZ + gp - bZ * bZ;
            fetot  = (eT + gp) * vZ - bZ * (vX * bX + vY * bY + vZ * bZ);
        }

        auto const bdotJ = bX * jX + bY * jY + bZ * jZ;
        auto const bdotB = bX * bX + bY * bY + bZ * bZ;
        if (dir == Direction::X)
            fetot += (bdotJ * bX - bdotB * jX) / r;
        else if (dir == Direction::Y)
            fetot += (bdotJ * bY - bdotB * jY) / r;
        else
            fetot += (bdotJ * bZ - bdotB * jZ) / r;

        return std::array<double, 5>{frho, frhoVx, frhoVy, frhoVz, fetot};
    }

    static auto electric(double x, double y, double z)
    {
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);
        auto const j  = current(x, y, z);
        auto const jX = j[0];
        auto const jY = j[1];
        auto const jZ = j[2];

        auto const ex = -(vY * bZ - vZ * bY) + (jY * bZ - jZ * bY) / r;
        auto const ey = -(vZ * bX - vX * bZ) + (jZ * bX - jX * bZ) / r;
        auto const ez = -(vX * bY - vY * bX) + (jX * bY - jY * bX) / r;

        return std::array<double, 3>{ex, ey, ez};
    }
};

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

    PHARE::initializer::PHAREDict state;
    state["name"] = std::string{"hall_state"};
    state["density"]["initializer"] = scalar_from_xyz(ExactHall3D::rho);
    state["velocity"]["initializer"]["x_component"] = scalar_from_xyz(ExactHall3D::vx);
    state["velocity"]["initializer"]["y_component"] = scalar_from_xyz(ExactHall3D::vy);
    state["velocity"]["initializer"]["z_component"] = scalar_from_xyz(ExactHall3D::vz);
    state["magnetic"]["initializer"]["x_component"] = scalar_from_xyz(ExactHall3D::bx);
    state["magnetic"]["initializer"]["y_component"] = scalar_from_xyz(ExactHall3D::by);
    state["magnetic"]["initializer"]["z_component"] = scalar_from_xyz(ExactHall3D::bz);
    state["pressure"]["initializer"] = scalar_from_xyz(ExactHall3D::pressure);
    state["to_conservative_init"]["heat_capacity_ratio"] = 1.4;

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
    hierarchy->makeNewPatchLevel(0, level0);

    return hierarchy;
}

template<typename MHDModelT>
struct HallFVMethod3D
{
    template<typename GridLayoutT>
    using type = Godunov<GridLayoutT, MHDModelT, WENOZReconstruction, Rusanov<true>,
                         MHDEquations<true, false, false>>;
};
