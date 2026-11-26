
#include "phare_mpi.hpp"
#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"

#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee.hpp"
#include "core/numerics/refinement/slope_rules.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"

#include "amr/data/field/refine/limited_field_refiner.hpp"
#include "amr/data/field/refine/refine_probe.hpp"

#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/SAMRAIManager.h>

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
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

template<std::size_t dim, typename Rule>
using Fill_t = PHARE::amr::detail::LimitedFill<dim, 2, Rule>;

using AllRules = std::tuple<refinement::CentralRule, refinement::VanLeerRule, refinement::MCRule,
                            refinement::MCGateRule, refinement::AO5ZRule, refinement::MPRule>;

template<typename Fn>
void forEachRule(Fn&& fn)
{
    std::apply([&](auto... rule) { (fn(rule), ...); }, AllRules{});
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

// coarse cells per direction, so that every 5-point line of the tested anchors is inside
template<std::size_t dim>
constexpr int nCoarse = dim == 1 ? 24 : (dim == 2 ? 12 : 8);

// random positive values plus a step of height 3 across the middle of direction 0: smooth,
// noisy and discontinuous lines for every rule
template<std::size_t dim>
Grid_t<dim> coarseData(unsigned seed)
{
    constexpr int n = nCoarse<dim>;
    Grid_t<dim> u{"coarse", HybridQuantity::Scalar::rho, cubeShape<dim>(n)};
    std::mt19937 gen{seed};
    std::uniform_real_distribution<double> dist{0.5, 2.};
    for_each_point(cube<dim>(0, n - 1),
                   [&](auto const& p) { u(p) = dist(gen) + (p[0] >= n / 2 ? 3. : 0.); });
    return u;
}

template<std::size_t dim>
auto reader(Grid_t<dim> const& u)
{
    return [&u](Point_t<dim> const& p) { return u(p); };
}

template<std::size_t dim>
double maxAbs(Grid_t<dim> const& u)
{
    double m = 0.;
    for (auto const v : u)
        m = std::max(m, std::abs(v));
    return m;
}

// fine region whose anchors are the coarse cells 2 .. n-3 of every direction
template<std::size_t dim>
IBox<dim> interiorFineRegion()
{
    return cube<dim>(4, 2 * (nCoarse<dim> - 3) + 1);
}

template<std::size_t dim>
std::array<QtyCentering, dim> centeringOf(unsigned mask)
{
    std::array<QtyCentering, dim> c{};
    for (std::size_t d = 0; d < dim; ++d)
        c[d] = ((mask >> d) & 1u) ? QtyCentering::primal : QtyCentering::dual;
    return c;
}

template<std::size_t dim>
auto allDual()
{
    return centeringOf<dim>(0u);
}

template<std::size_t dim>
Point_t<dim> toCoarse(Point_t<dim> const& fine)
{
    return toCoarseIndex<dim>(fine);
}

constexpr std::array<RefinementSweepOrders, 3> allOrderSets{
    RefinementSweepOrders::All, RefinementSweepOrders::Cyclic, RefinementSweepOrders::Single};

template<typename Kernel>
bool isA(auto const& kernel)
{
    return dynamic_cast<Kernel const*>(kernel.get()) != nullptr;
}

} // namespace



// (a) with the central rule the product mode is master's table, every centering
template<std::size_t dim>
void noneProductEqualsCompositeFill()
{
    auto const coarse = coarseData<dim>(11 + dim);
    auto const scale  = maxAbs(coarse);
    auto const region = interiorFineRegion<dim>();
    constexpr int nf  = 2 * nCoarse<dim>;

    using Composite = PHARE::amr::detail::CompositeFill<GridLayout_t<dim>, Grid_t<dim>, 2>;
    using Limited   = Fill_t<dim, refinement::CentralRule>;

    for (unsigned mask = 0; mask < (1u << dim); ++mask)
    {
        auto const centering = centeringOf<dim>(mask);

        Grid_t<dim> master{"fine", HybridQuantity::Scalar::rho, cubeShape<dim>(nf), NaN};
        Composite::fill(coarse, master, samraiBox(region), centering,
                        samraiBox(cube<dim>(0, nf - 1)), samraiBox(cube<dim>(0, nCoarse<dim> - 1)),
                        ratio2(dim));

        auto const block = Limited::refineRegion(reader(coarse), region, centering, {}, nullptr);

        std::size_t compared = 0;
        for_each_point(region, [&](auto const& f) {
            ASSERT_FALSE(std::isnan(master(f)));
            EXPECT_LE(std::abs(block(f) - master(f)), 1e-14 * scale)
                << "dim " << dim << " centering mask " << mask;
            ++compared;
        });
        EXPECT_EQ(compared, region.size());
    }
}

TEST(limitedFillProduct, centralRuleEqualsMastersTable1D)
{
    noneProductEqualsCompositeFill<1>();
}
TEST(limitedFillProduct, centralRuleEqualsMastersTable2D)
{
    noneProductEqualsCompositeFill<2>();
}
TEST(limitedFillProduct, centralRuleEqualsMastersTable3D)
{
    noneProductEqualsCompositeFill<3>();
}



// (b) the 2^dim children of a coarse cell mean back to it, every rule
template<std::size_t dim>
void childrenConserveTheCoarseCell()
{
    auto const coarse = coarseData<dim>(23 + dim);
    auto const scale  = maxAbs(coarse);
    auto const region = interiorFineRegion<dim>();

    forEachRule([&]<typename Rule>(Rule) {
        auto const block
            = Fill_t<dim, Rule>::refineRegion(reader(coarse), region, allDual<dim>(), {}, nullptr);

        IBox<dim> const anchors{toCoarse<dim>(region.lower), toCoarse<dim>(region.upper)};
        double worst = 0.;
        for_each_point(anchors, [&](auto const& I) {
            double sum = 0.;
            for_each_point(IBox<dim>{Point_t<dim>{I} * 2, Point_t<dim>{I} * 2 + 1},
                           [&](auto const& f) { sum += block(f); });
            worst = std::max(worst, std::abs(sum / (1 << dim) - coarse(I)));
        });
        EXPECT_LE(worst, 1e-13 * scale) << "dim " << dim << " width " << Rule::width;
    });
}

TEST(limitedFillProduct, childrenConserveTheCoarseCellForEveryRule1D)
{
    childrenConserveTheCoarseCell<1>();
}
TEST(limitedFillProduct, childrenConserveTheCoarseCellForEveryRule2D)
{
    childrenConserveTheCoarseCell<2>();
}
TEST(limitedFillProduct, childrenConserveTheCoarseCellForEveryRule3D)
{
    childrenConserveTheCoarseCell<3>();
}



// (c) 1D: the children are u0 -+ s/4 with s the rule's slope of the anchor's line
TEST(limitedFillProduct, childrenAreTheRuleSlopeLadder1D)
{
    auto const coarse = coarseData<1>(5);
    auto const scale  = maxAbs(coarse);
    auto const region = interiorFineRegion<1>();

    forEachRule([&]<typename Rule>(Rule) {
        constexpr int w = Rule::width;
        auto const block
            = Fill_t<1, Rule>::refineRegion(reader(coarse), region, allDual<1>(), {}, nullptr);

        std::size_t limited = 0;
        for (int I = region.lower[0] / 2; I <= region.upper[0] / 2; ++I)
        {
            std::array<double, 2 * w + 1> line;
            for (int k = -w; k <= w; ++k)
                line[k + w] = coarse(I + k);
            double const c = (line[w + 1] - line[w - 1]) / 2.;
            double const s = Rule::slope(line, c, Rule::defaultParam);
            limited += s != c;

            EXPECT_NEAR(block(Point_t<1>{2 * I}), coarse(I) - s / 4., 1e-14 * scale);
            EXPECT_NEAR(block(Point_t<1>{2 * I + 1}), coarse(I) + s / 4., 1e-14 * scale);
        }
        if constexpr (!std::is_same_v<Rule, refinement::CentralRule>)
            EXPECT_GT(limited, 0u) << "the data must exercise the limiter";
    });
}



// (d) a non-finite value beyond master's reach (offset +-2) is an error, within it (+-1) it
// propagates as in master
TEST(limitedFillProduct, nonFiniteFivePointReadThrowsAndThreePointReadPropagates)
{
    auto run = [&]<typename Rule>(Rule, int const offset, RefinementTensorMode const tensor) {
        auto coarse      = coarseData<1>(7);
        constexpr int J  = 12;
        coarse(J)        = NaN;
        int const anchor = J - offset;
        IBox<1> const fr = {Point_t<1>{2 * anchor}, Point_t<1>{2 * anchor + 1}};
        return Fill_t<1, Rule>::refineRegion(reader(coarse), fr, allDual<1>(), {.tensor = tensor},
                                             nullptr);
    };

    for (auto const tensor : {RefinementTensorMode::Product, RefinementTensorMode::Sweeps})
    {
        for (int const offset : {-2, 2})
        {
            EXPECT_THROW(run(refinement::AO5ZRule{}, offset, tensor), std::runtime_error);
            EXPECT_THROW(run(refinement::MPRule{}, offset, tensor), std::runtime_error);
        }
        for (int const offset : {-1, 1})
        {
            forEachRule([&]<typename Rule>(Rule) {
                EXPECT_NO_THROW({
                    auto const block = run(Rule{}, offset, tensor);
                    EXPECT_TRUE(std::isnan(block(block.box.lower)));
                    EXPECT_TRUE(std::isnan(block(block.box.upper)));
                });
            });
        }
    }
}



// (e) through the kernel with real fields, non-zero box origins: already-set fine values are
// kept, the others are the conservative children, and the probe counts one row per anchor and
// dual direction
TEST(limitedFieldRefiner2D, vanLeerWritesUnsetChildrenOnlyAndConserves)
{
    constexpr std::size_t dim = 2;
    constexpr int n           = nCoarse<dim>;
    constexpr double kept     = 42.;

    auto const coarse = coarseData<dim>(3);
    auto const scale  = maxAbs(coarse);

    // AMR boxes: the coarse field starts at -2, the fine one at -4 (same physical origin)
    auto const sourceFieldBox = cube<dim>(-2, n - 3);
    auto const destFieldBox   = cube<dim>(-4, 2 * n - 5);
    auto const region         = cube<dim>(0, 2 * (n - 5) + 1); // anchors 0 .. n-5 in AMR

    Grid_t<dim> fine{"fine", HybridQuantity::Scalar::rho, cubeShape<dim>(2 * n), NaN};
    auto const toLocalFine = [&](Point_t<dim> const& f) { return f - destFieldBox.lower; };
    for_each_point(region, [&](auto const& f) {
        if ((f[0] + 3 * f[1]) % 7 == 0)
            fine(toLocalFine(f)) = kept;
    });

    LimiterSettings settings{};
    settings.limiter = FieldRefinementLimiter::VanLeer;
    auto probe       = std::make_shared<RefineProbe>();

    auto const kernel = makeRefineKernel<GridLayout_t<dim>, Grid_t<dim>>(
        FieldRefinementOrder::Linear, settings, probe);
    using VanLeerKernel
        = LimitedFieldRefiner<GridLayout_t<dim>, Grid_t<dim>, 2, refinement::VanLeerRule>;
    ASSERT_TRUE(isA<VanLeerKernel>(kernel));
    EXPECT_EQ(kernel->coarseStencilWidth(), 1);

    kernel->refineBox(coarse, fine, samraiBox(region), allDual<dim>(), samraiBox(destFieldBox),
                      samraiBox(sourceFieldBox), ratio2(dim));

    // the same children from refineRegion on the AMR coarse reader
    auto const coarseAMR = [&](Point_t<dim> const& I) { return coarse(I - sourceFieldBox.lower); };
    auto const block     = Fill_t<dim, refinement::VanLeerRule>::refineRegion(
        coarseAMR, region, allDual<dim>(), {}, nullptr);

    std::size_t nKept = 0;
    for_each_point(region, [&](auto const& f) {
        double const v = fine(toLocalFine(f));
        if ((f[0] + 3 * f[1]) % 7 == 0)
        {
            EXPECT_EQ(v, kept);
            ++nKept;
        }
        else
            EXPECT_EQ(v, block(f));
    });
    EXPECT_GT(nKept, 0u);

    IBox<dim> const anchors{toCoarse<dim>(region.lower), toCoarse<dim>(region.upper)};
    for_each_point(anchors, [&](auto const& I) {
        double sum = 0.;
        for_each_point(IBox<dim>{Point_t<dim>{I} * 2, Point_t<dim>{I} * 2 + 1},
                       [&](auto const& f) { sum += block(f); });
        EXPECT_LE(std::abs(sum / 4. - coarseAMR(I)), 1e-13 * scale);
    });

    EXPECT_EQ(probe->rows.load(), dim * anchors.size());
    EXPECT_GT(probe->limitedRows.load(), 0u);

    // nothing outside the region is written
    std::size_t written = 0;
    for (auto const v : fine)
        written += !std::isnan(v);
    EXPECT_EQ(written, region.size());
}



// (f) with the central rule every sweep is linear: sweeps equal the product mode and master's
// table, every centering and order set
template<std::size_t dim>
void noneSweepsEqualCompositeFill()
{
    auto const coarse = coarseData<dim>(31 + dim);
    auto const scale  = maxAbs(coarse);
    auto const region = interiorFineRegion<dim>();
    constexpr int nf  = 2 * nCoarse<dim>;

    using Composite = PHARE::amr::detail::CompositeFill<GridLayout_t<dim>, Grid_t<dim>, 2>;
    using Limited   = Fill_t<dim, refinement::CentralRule>;

    for (unsigned mask = 0; mask < (1u << dim); ++mask)
    {
        auto const centering = centeringOf<dim>(mask);

        Grid_t<dim> master{"fine", HybridQuantity::Scalar::rho, cubeShape<dim>(nf), NaN};
        Composite::fill(coarse, master, samraiBox(region), centering,
                        samraiBox(cube<dim>(0, nf - 1)), samraiBox(cube<dim>(0, nCoarse<dim> - 1)),
                        ratio2(dim));
        auto const product = Limited::refineRegion(reader(coarse), region, centering, {}, nullptr);

        for (auto const orders : allOrderSets)
        {
            auto const sweeps = Limited::refineRegion(
                reader(coarse), region, centering,
                {.tensor = RefinementTensorMode::Sweeps, .sweepOrders = orders}, nullptr);

            for_each_point(region, [&](auto const& f) {
                ASSERT_FALSE(std::isnan(master(f)));
                EXPECT_LE(std::abs(sweeps(f) - master(f)), 1e-14 * scale)
                    << "dim " << dim << " centering mask " << mask;
                EXPECT_LE(std::abs(sweeps(f) - product(f)), 1e-14 * scale)
                    << "dim " << dim << " centering mask " << mask;
            });
        }
    }
}

TEST(limitedFillSweeps, centralRuleEqualsProductAndMastersTable1D)
{
    noneSweepsEqualCompositeFill<1>();
}
TEST(limitedFillSweeps, centralRuleEqualsProductAndMastersTable2D)
{
    noneSweepsEqualCompositeFill<2>();
}
TEST(limitedFillSweeps, centralRuleEqualsProductAndMastersTable3D)
{
    noneSweepsEqualCompositeFill<3>();
}



// (g) every sweep conserves, so does their mean: every rule and order set
template<std::size_t dim>
void sweepsConserveTheCoarseCell()
{
    auto const coarse = coarseData<dim>(41 + dim);
    auto const scale  = maxAbs(coarse);
    auto const region = interiorFineRegion<dim>();
    IBox<dim> const anchors{toCoarse<dim>(region.lower), toCoarse<dim>(region.upper)};

    forEachRule([&]<typename Rule>(Rule) {
        for (auto const orders : allOrderSets)
        {
            auto const block = Fill_t<dim, Rule>::refineRegion(
                reader(coarse), region, allDual<dim>(),
                {.tensor = RefinementTensorMode::Sweeps, .sweepOrders = orders}, nullptr);

            double worst = 0.;
            for_each_point(anchors, [&](auto const& I) {
                double sum = 0.;
                for_each_point(IBox<dim>{Point_t<dim>{I} * 2, Point_t<dim>{I} * 2 + 1},
                               [&](auto const& f) { sum += block(f); });
                worst = std::max(worst, std::abs(sum / (1 << dim) - coarse(I)));
            });
            EXPECT_LE(worst, 1e-13 * scale) << "dim " << dim << " width " << Rule::width
                                            << " orders " << static_cast<int>(orders);
        }
    });
}

TEST(limitedFillSweeps, childrenConserveTheCoarseCellForEveryRuleAndOrderSet2D)
{
    sweepsConserveTheCoarseCell<2>();
}
TEST(limitedFillSweeps, childrenConserveTheCoarseCellForEveryRuleAndOrderSet3D)
{
    sweepsConserveTheCoarseCell<3>();
}



// (h) averaged over every direction order, sweeps commute with a permutation of the axes:
// refining v(q) = u(Pq), (Pq)[d] = q[perm[d]], gives the refinement of u at Pf
template<std::size_t dim>
void allOrdersCommuteWithAxisPermutations()
{
    auto const coarse = coarseData<dim>(53 + dim);
    auto const scale  = maxAbs(coarse);
    auto const region = interiorFineRegion<dim>();

    std::array<std::size_t, dim> perm{};
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    auto const permuted = [&](Point_t<dim> const& q) {
        Point_t<dim> p;
        for (std::size_t d = 0; d < dim; ++d)
            p[d] = q[perm[d]];
        return p;
    };

    forEachRule([&]<typename Rule>(Rule) {
        using Fill        = Fill_t<dim, Rule>;
        auto const sweeps = typename Fill::Options{.tensor = RefinementTensorMode::Sweeps};
        auto const ref
            = Fill::refineRegion(reader(coarse), region, allDual<dim>(), sweeps, nullptr);

        std::sort(perm.begin(), perm.end());
        while (std::next_permutation(perm.begin(), perm.end()))
        {
            auto const v     = [&](Point_t<dim> const& q) { return coarse(permuted(q)); };
            auto const block = Fill::refineRegion(v, region, allDual<dim>(), sweeps, nullptr);

            double worst = 0.;
            for_each_point(region, [&](auto const& f) {
                worst = std::max(worst, std::abs(block(f) - ref(permuted(Point_t<dim>{f}))));
            });
            EXPECT_LE(worst, 1e-14 * scale) << "dim " << dim << " width " << Rule::width;
        }
    });
}

TEST(limitedFillSweeps, allOrdersCommuteWithTranspose2D)
{
    allOrdersCommuteWithAxisPermutations<2>();
}
TEST(limitedFillSweeps, allOrdersCommuteWithAxisPermutations3D)
{
    allOrdersCommuteWithAxisPermutations<3>();
}



// (i) a step corner: 1 on one quadrant, 0.1 elsewhere. At the anchor diagonal to the corner both
// lines are flat (c = s = 0, phi = 1), so the product keeps master's cross term and undershoots;
// the van Leer sweeps see flat or extremal lines only and keep every child within the coarse
// 3x3 neighbourhood of its anchor
TEST(limitedFillSweeps, vanLeerSweepsStayBoundedAtAStepCornerWhereProductDoesNot)
{
    constexpr std::size_t dim = 2;
    constexpr int n           = nCoarse<dim>;
    using Fill                = Fill_t<dim, refinement::VanLeerRule>;

    Grid_t<dim> coarse{"coarse", HybridQuantity::Scalar::rho, cubeShape<dim>(n)};
    for_each_point(cube<dim>(0, n - 1), [&](auto const& p) {
        coarse(p) = (p[0] >= n / 2 and p[1] >= n / 2) ? 1. : 0.1;
    });
    auto const region = interiorFineRegion<dim>();
    IBox<dim> const anchors{toCoarse<dim>(region.lower), toCoarse<dim>(region.upper)};

    auto const product = Fill::refineRegion(reader(coarse), region, allDual<dim>(), {}, nullptr);

    for (auto const orders : allOrderSets)
    {
        auto const sweeps = Fill::refineRegion(
            reader(coarse), region, allDual<dim>(),
            {.tensor = RefinementTensorMode::Sweeps, .sweepOrders = orders}, nullptr);

        std::size_t sweepsOut = 0, productOut = 0;
        double productMin = 1., productMax = 0.;
        for_each_point(anchors, [&](auto const& a) {
            Point_t<dim> const I{a};
            double lo = 1., hi = 0.;
            for_each_point(IBox<dim>{I - 1, I + 1}, [&](auto const& J) {
                lo = std::min(lo, coarse(J));
                hi = std::max(hi, coarse(J));
            });
            for_each_point(IBox<dim>{I * 2, I * 2 + 1}, [&](auto const& f) {
                sweepsOut += sweeps(f) < lo - 1e-15 or sweeps(f) > hi + 1e-15;
                productOut += product(f) < lo - 1e-15 or product(f) > hi + 1e-15;
                productMin = std::min(productMin, product(f));
                productMax = std::max(productMax, product(f));
            });
        });

        EXPECT_EQ(sweepsOut, 0u) << "orders " << static_cast<int>(orders);
        EXPECT_GT(productOut, 0u) << "product extremes " << productMin << " " << productMax;
    }
}



TEST(limitedFieldRefiner, factoryMapsEverySettingToItsKernel)
{
    using Layout = GridLayout_t<1>;
    using G      = Grid_t<1>;
    using Master = CompositeFieldRefiner<Layout, G, 2>;

    auto const master = makeRefineKernel<Layout, G>(FieldRefinementOrder::Linear);
    EXPECT_TRUE(isA<Master>(master));

    for (auto const [limiter, width] : std::array<std::pair<FieldRefinementLimiter, int>, 5>{
             {{FieldRefinementLimiter::VanLeer, 1},
              {FieldRefinementLimiter::MC, 1},
              {FieldRefinementLimiter::MCGate, 1},
              {FieldRefinementLimiter::AO5Z, 2},
              {FieldRefinementLimiter::MP, 2}}})
    {
        LimiterSettings settings{};
        settings.limiter = limiter;
        EXPECT_EQ(limiterStencilWidth(settings), width);
        auto const kernel = makeRefineKernel<Layout, G>(FieldRefinementOrder::Linear, settings);
        EXPECT_FALSE(isA<Master>(kernel));
        EXPECT_EQ(kernel->coarseStencilWidth(), width);
    }
    EXPECT_EQ(limiterStencilWidth(LimiterSettings{}), 1);

    // none in sweeps mode is not master's table: the limited kernel
    LimiterSettings sweeps{};
    sweeps.tensor     = RefinementTensorMode::Sweeps;
    auto const kernel = makeRefineKernel<Layout, G>(FieldRefinementOrder::Linear, sweeps);
    EXPECT_FALSE(isA<Master>(kernel));
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
