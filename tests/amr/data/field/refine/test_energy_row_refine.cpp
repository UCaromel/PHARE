#include "phare_mpi.hpp"
#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"

#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee.hpp"
#include "core/numerics/refinement/slope_rules.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"

#include "amr/data/field/refine/energy_row_refiner.hpp"
#include "amr/data/field/refine/limited_field_refiner.hpp"

#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/SAMRAIManager.h>

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>


using namespace PHARE::core;
using namespace PHARE::amr;


namespace
{
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

template<std::size_t dim>
using GridLayout_t =
    typename PHARE::core::PHARE_Types<PHARE::SimOpts{dim, 1}>::Hybrid::GridLayout_t;

template<std::size_t dim>
using Grid_t = Grid<NdArrayVector<dim>, HybridQuantity::Scalar>;

template<std::size_t dim>
using Point_t = Point<int, dim>;

template<std::size_t dim>
using IBox = PHARE::core::Box<int, dim>;

constexpr std::array allLimiters{FieldRefinementLimiter::None, FieldRefinementLimiter::VanLeer,
                                 FieldRefinementLimiter::MC,   FieldRefinementLimiter::MCGate,
                                 FieldRefinementLimiter::AO5Z, FieldRefinementLimiter::MP};

std::string nameOf(FieldRefinementLimiter const limiter)
{
    using enum FieldRefinementLimiter;
    switch (limiter)
    {
        case None: return "none";
        case VanLeer: return "vanleer";
        case MC: return "mc";
        case MCGate: return "mc_gate";
        case AO5Z: return "ao5z";
        case MP: return "mp";
    }
    return "?";
}

template<std::size_t dim>
SAMRAI::hier::Box samraiBox(IBox<dim> const& box)
{
    SAMRAI::tbox::Dimension d{static_cast<unsigned short>(dim)};
    SAMRAI::hier::Index lo{d, 0}, up{d, 0};
    for (std::size_t k = 0; k < dim; ++k)
    {
        lo(k) = box.lower[k];
        up(k) = box.upper[k];
    }
    return SAMRAI::hier::Box{lo, up, SAMRAI::hier::BlockId{0}};
}

SAMRAI::hier::IntVector ratio2(std::size_t dim)
{
    return SAMRAI::hier::IntVector{SAMRAI::tbox::Dimension{static_cast<unsigned short>(dim)}, 2};
}

template<std::size_t dim>
IBox<dim> cube(int lo, int up)
{
    Point_t<dim> l, u;
    for (std::size_t d = 0; d < dim; ++d)
    {
        l[d] = lo;
        u[d] = up;
    }
    return {l, u};
}

template<std::size_t dim>
std::array<std::uint32_t, dim> cubeShape(std::uint32_t n)
{
    std::array<std::uint32_t, dim> shape{};
    shape.fill(n);
    return shape;
}

template<std::size_t dim>
constexpr int nCoarse = dim == 1 ? 24 : (dim == 2 ? 12 : 8);

template<std::size_t dim>
Grid_t<dim> coarseGrid(std::string const& name)
{
    return Grid_t<dim>{name, HybridQuantity::Scalar::rho, cubeShape<dim>(nCoarse<dim>)};
}

template<std::size_t dim>
Grid_t<dim> fineGrid()
{
    return Grid_t<dim>{"fine", HybridQuantity::Scalar::rho, cubeShape<dim>(2 * nCoarse<dim>), NaN};
}

// fine region whose anchors are the coarse cells 2 .. n-3: every 5-point line inside
template<std::size_t dim>
IBox<dim> interiorFineRegion()
{
    return cube<dim>(4, 2 * (nCoarse<dim> - 3) + 1);
}

template<std::size_t dim>
std::array<QtyCentering, dim> allDual()
{
    std::array<QtyCentering, dim> c;
    c.fill(QtyCentering::dual);
    return c;
}

double kinetic(double rho, double mx, double my, double mz)
{
    return (mx * mx + my * my + mz * mz) / (2. * rho);
}


// a hydro state on the coarse grid (AMR box = local box, origin 0)
template<std::size_t dim>
struct CoarseState
{
    Grid_t<dim> rho = coarseGrid<dim>("rho");
    Grid_t<dim> mx  = coarseGrid<dim>("mx");
    Grid_t<dim> my  = coarseGrid<dim>("my");
    Grid_t<dim> mz  = coarseGrid<dim>("mz");
    Grid_t<dim> E   = coarseGrid<dim>("E");

    double q(Point_t<dim> const& I) const { return E(I) - kinetic(rho(I), mx(I), my(I), mz(I)); }

    // the energy-row kernel of `limiter` on `region` into `fine`
    void energyRow(FieldRefinementLimiter limiter, IBox<dim> const& region, Grid_t<dim>& fine) const
    {
        LimiterSettings settings{};
        settings.limiter   = limiter;
        settings.energyRow = true;
        auto const kernel  = makeEnergyRowKernel<GridLayout_t<dim>, Grid_t<dim>>(
            FieldRefinementOrder::Linear, settings);
        kernel->refineBox(rho,
                          std::array<Grid_t<dim> const*, 3>{std::addressof(mx), std::addressof(my),
                                                            std::addressof(mz)},
                          E, fine, samraiBox(region), samraiBox(cube<dim>(0, 2 * nCoarse<dim> - 1)),
                          samraiBox(cube<dim>(0, nCoarse<dim> - 1)), ratio2(dim));
    }
};

template<std::size_t dim>
auto reader(Grid_t<dim> const& u)
{
    return [&u](Point_t<dim> const& p) { return u(p); };
}

template<std::size_t dim>
IBox<dim> childrenOf(Point_t<dim> const& I)
{
    return {I * 2, I * 2 + 1};
}

template<std::size_t dim>
IBox<dim> anchorsOf(IBox<dim> const& fineRegion)
{
    return {toCoarseIndex<dim>(fineRegion.lower), toCoarseIndex<dim>(fineRegion.upper)};
}

// the limited per-variable fill of `limiter` (product mode)
template<std::size_t dim, typename CoarseFn>
auto limitedRefine(FieldRefinementLimiter limiter, CoarseFn const& coarse, IBox<dim> const& region)
{
    return withRule(limiter, [&]<typename Rule>(std::type_identity<Rule>) {
        return PHARE::amr::detail::LimitedFill<dim, 2, Rule>::refineRegion(
            coarse, region, allDual<dim>(), {}, nullptr);
    });
}

} // namespace



// (a) the 2^dim children mean back to the coarse Etot, every rule
template<std::size_t dim>
void energyRowConservesEtot()
{
    CoarseState<dim> s;
    std::mt19937 gen{17 + dim};
    std::uniform_real_distribution<double> pos{0.5, 2.}, sgn{-3., 3.};
    for_each_point(cube<dim>(0, nCoarse<dim> - 1), [&](auto const& p) {
        s.rho(p) = pos(gen) + (p[0] >= nCoarse<dim> / 2 ? 4. : 0.);
        s.mx(p)  = sgn(gen);
        s.my(p)  = sgn(gen);
        s.mz(p)  = sgn(gen);
        s.E(p)   = pos(gen) + kinetic(s.rho(p), s.mx(p), s.my(p), s.mz(p));
    });
    double scale = 0.;
    for (auto const v : s.E)
        scale = std::max(scale, std::abs(v));

    auto const region = interiorFineRegion<dim>();
    for (auto const limiter : allLimiters)
    {
        auto fine = fineGrid<dim>();
        s.energyRow(limiter, region, fine);

        double worst = 0.;
        for_each_point(anchorsOf<dim>(region), [&](auto const& anchor) {
            Point_t<dim> const I{anchor};
            double sum = 0.;
            for_each_point(childrenOf<dim>(I), [&](auto const& f) {
                ASSERT_FALSE(std::isnan(fine(f)));
                sum += fine(f);
            });
            worst = std::max(worst, std::abs(sum / (1 << dim) - s.E(I)));
        });
        EXPECT_LE(worst, 1e-13 * scale) << "dim " << dim << " rule " << nameOf(limiter);
    }
}

TEST(energyRow, childrenConserveEtotForEveryRule1D)
{
    energyRowConservesEtot<1>();
}
TEST(energyRow, childrenConserveEtotForEveryRule2D)
{
    energyRowConservesEtot<2>();
}
TEST(energyRow, childrenConserveEtotForEveryRule3D)
{
    energyRowConservesEtot<3>();
}



// (b) uniform density and velocity: the kinetic energy is uniform, so E_k - E_0 = q_k - q_0
template<std::size_t dim>
void uniformFlowFollowsQ()
{
    CoarseState<dim> s;
    std::mt19937 gen{5 + dim};
    std::uniform_real_distribution<double> pres{0.1, 3.};
    double const rho = 1.7, vx = 0.6, vy = -1.3, vz = 2.1;
    for_each_point(cube<dim>(0, nCoarse<dim> - 1), [&](auto const& p) {
        s.rho(p) = rho;
        s.mx(p)  = rho * vx;
        s.my(p)  = rho * vy;
        s.mz(p)  = rho * vz;
        s.E(p)   = pres(gen) + kinetic(rho, rho * vx, rho * vy, rho * vz);
    });
    double scale = 0.;
    for (auto const v : s.E)
        scale = std::max(scale, std::abs(v));

    auto const region = interiorFineRegion<dim>();
    for (auto const limiter : allLimiters)
    {
        auto fine = fineGrid<dim>();
        s.energyRow(limiter, region, fine);

        auto const qF
            = limitedRefine<dim>(limiter, [&](Point_t<dim> const& I) { return s.q(I); }, region);

        double worst = 0.;
        for_each_point(region, [&](auto const& f) {
            Point_t<dim> const fp{f};
            auto const I = toCoarseIndex<dim>(fp);
            worst        = std::max(worst, std::abs((fine(fp) - s.E(I)) - (qF(fp) - s.q(I))));
        });
        EXPECT_LE(worst, 1e-13 * scale) << "dim " << dim << " rule " << nameOf(limiter);
    }
}

TEST(energyRow, uniformFlowChildrenFollowQ1D)
{
    uniformFlowFollowsQ<1>();
}
TEST(energyRow, uniformFlowChildrenFollowQ2D)
{
    uniformFlowFollowsQ<2>();
}
TEST(energyRow, uniformFlowChildrenFollowQ3D)
{
    uniformFlowFollowsQ<3>();
}



// (c) only the overlap's unset fine cells are written, and they are the children of the whole
// coarse cells: an overlap cutting coarse cells in half gives the same values as the full one
TEST(energyRow, writesUnsetCellsOfTheOverlapOnly2D)
{
    constexpr std::size_t dim = 2;
    constexpr double kept     = 42.;

    CoarseState<dim> s;
    std::mt19937 gen{29};
    std::uniform_real_distribution<double> pos{0.5, 2.}, sgn{-3., 3.};
    for_each_point(cube<dim>(0, nCoarse<dim> - 1), [&](auto const& p) {
        s.rho(p) = pos(gen);
        s.mx(p)  = sgn(gen);
        s.my(p)  = sgn(gen);
        s.mz(p)  = sgn(gen);
        s.E(p)   = pos(gen) + kinetic(s.rho(p), s.mx(p), s.my(p), s.mz(p));
    });

    auto full = fineGrid<dim>();
    s.energyRow(FieldRefinementLimiter::VanLeer, interiorFineRegion<dim>(), full);

    // odd lower and even upper bounds: every edge cuts coarse cells
    IBox<dim> const overlap = cube<dim>(5, 2 * (nCoarse<dim> - 3));
    auto part               = fineGrid<dim>();
    for_each_point(overlap, [&](auto const& f) {
        if ((f[0] + 2 * f[1]) % 5 == 0)
            part(f) = kept;
    });
    s.energyRow(FieldRefinementLimiter::VanLeer, overlap, part);

    std::size_t nKept = 0, nWritten = 0;
    for_each_point(cube<dim>(0, 2 * nCoarse<dim> - 1), [&](auto const& f) {
        Point_t<dim> const fp{f};
        if (!isIn(fp, overlap))
            EXPECT_TRUE(std::isnan(part(fp)));
        else if ((f[0] + 2 * f[1]) % 5 == 0)
        {
            EXPECT_EQ(part(fp), kept);
            ++nKept;
        }
        else
        {
            EXPECT_EQ(part(fp), full(fp));
            ++nWritten;
        }
    });
    EXPECT_GT(nKept, 0u);
    EXPECT_GT(nWritten, 0u);
}



// (d) rotor-like cells, kinetic energy >> pressure across a density and velocity jump: van Leer
// with the energy row keeps the children's pressure positive. The per-variable Etot of the same
// rule is printed for comparison (not asserted).
template<std::size_t dim>
void rotorChildrenPressure()
{
    constexpr int n    = nCoarse<dim>;
    double const p0    = 0.1;
    double const r0    = 0.1;
    double const r1    = 0.115;
    auto const centre  = [](int i) { return (i + 0.5) / n - 0.5; };
    auto const profile = [&](double r) { return std::clamp((r1 - r) / (r1 - r0), 0., 1.); };

    CoarseState<dim> s;
    double maxKE = 0.;
    for_each_point(cube<dim>(0, n - 1), [&](auto const& p) {
        double const x = centre(p[0]);
        double vx = 0., vy = 0.;
        double r = std::abs(x);
        if constexpr (dim == 1)
            vy = 2. * (x < 0 ? -1. : 1.) * profile(r);
        else
        {
            double const y = centre(p[1]);
            r              = std::sqrt(x * x + y * y);
            double const w = 20. * profile(r);
            vx             = -w * y;
            vy             = w * x;
        }
        double const rho = 1. + 9. * profile(r);
        s.rho(p)         = rho;
        s.mx(p)          = rho * vx;
        s.my(p)          = rho * vy;
        s.mz(p)          = 0.;
        double const ke  = kinetic(rho, rho * vx, rho * vy, 0.);
        s.E(p)           = p0 + ke;
        maxKE            = std::max(maxKE, ke);
    });
    ASSERT_GT(maxKE, 10. * p0);

    // anchors 1 .. n-2: the 3-point lines of van Leer
    auto const region = cube<dim>(2, 2 * (n - 2) + 1);
    auto const vl     = FieldRefinementLimiter::VanLeer;

    auto const rhoF = limitedRefine<dim>(vl, reader(s.rho), region);
    auto const mxF  = limitedRefine<dim>(vl, reader(s.mx), region);
    auto const myF  = limitedRefine<dim>(vl, reader(s.my), region);
    auto const mzF  = limitedRefine<dim>(vl, reader(s.mz), region);
    auto const EF   = limitedRefine<dim>(vl, reader(s.E), region);

    auto fine = fineGrid<dim>();
    s.energyRow(vl, region, fine);

    double minEnergyRow = std::numeric_limits<double>::max();
    double minPlain     = std::numeric_limits<double>::max();
    for_each_point(region, [&](auto const& f) {
        Point_t<dim> const fp{f};
        double const ke = kinetic(rhoF(fp), mxF(fp), myF(fp), mzF(fp));
        minEnergyRow    = std::min(minEnergyRow, fine(fp) - ke);
        minPlain        = std::min(minPlain, EF(fp) - ke);
    });

    std::cout << "rotor " << dim << "D vanleer, min (Etot - KE) of the children: energy row "
              << minEnergyRow << ", per-variable Etot " << minPlain << " (coarse p = " << p0
              << ")\n";
    EXPECT_GT(minEnergyRow, 0.) << "dim " << dim;
}

TEST(energyRow, rotorVanLeerKeepsChildPressurePositive1D)
{
    rotorChildrenPressure<1>();
}
TEST(energyRow, rotorVanLeerKeepsChildPressurePositive2D)
{
    rotorChildrenPressure<2>();
}
TEST(energyRow, rotorVanLeerKeepsChildPressurePositive3D)
{
    rotorChildrenPressure<3>();
}



int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();

    int testResult = RUN_ALL_TESTS();

    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();

    return testResult;
}
