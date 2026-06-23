
#include "core/def/phare_mpi.hpp"

#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee.hpp"

#include "amr/data/field/refine/field_linear_refine.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/field_refiner.hpp"
#include "amr/data/field/refine/composite_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_composite_refiner.hpp"

#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/SAMRAIManager.h>

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <limits>



using namespace PHARE::core;
using namespace PHARE::amr;


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------


TEST(UniformIntervalPartition, givesCorrectPartitionsForPrimal)
{
    LinearWeighter linearWeighter{QtyCentering::primal, 2};
    std::array<double, 2> expectedDistances{0, 0.5};

    auto const& actualDistances = linearWeighter.getUniformDistances();

    for (auto i = 0u; i < 2; ++i)
    {
        EXPECT_DOUBLE_EQ(expectedDistances[i], actualDistances[i]);
    }
}


TEST(UniformIntervalPartition, givesCorrectPartitionsForDual)
{
    LinearWeighter linearWeighter{QtyCentering::dual, 2};
    std::array<double, 2> expectedDistances{0.75, 0.25};

    auto const& actualDistances = linearWeighter.getUniformDistances();

    for (auto i = 0u; i < 2; ++i)
    {
        EXPECT_DOUBLE_EQ(expectedDistances[i], actualDistances[i]);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------


template<typename TypeInfo /*= std::pair<DimConst<1>, InterpConst<1>>*/>
struct aFieldRefineOperator : public ::testing::Test
{
};

using aFieldRefineOperatorInfos
    = testing::Types<std::pair<DimConst<1>, InterpConst<1>>, std::pair<DimConst<1>, InterpConst<2>>,
                     std::pair<DimConst<1>, InterpConst<3>>, std::pair<DimConst<2>, InterpConst<1>>,
                     std::pair<DimConst<2>, InterpConst<2>>, std::pair<DimConst<2>, InterpConst<3>>,
                     std::pair<DimConst<3>, InterpConst<1>>, std::pair<DimConst<3>, InterpConst<2>>,
                     std::pair<DimConst<3>, InterpConst<3>>>;

TYPED_TEST_SUITE(aFieldRefineOperator, aFieldRefineOperatorInfos);


TYPED_TEST(aFieldRefineOperator, canBeCreated)
{
    static constexpr auto dim    = typename TypeParam::first_type{}();
    static constexpr auto interp = typename TypeParam::second_type{}();

    using GridYee = GridLayout<GridLayoutImplYee<dim, interp>>;
    using GridT   = Grid<NdArrayVector<dim>, HybridQuantity::Scalar>;

    FieldRefineOperator<GridYee, GridT, DefaultFieldRefiner<dim>> linearRefine{};
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// instantiation gate: forces full compilation of CompositeFieldRefiner<...,order> (vtable →
// refineBox) and the additive KernelFieldRefineOperator across all dim/interp, orders 2 and 4.
TYPED_TEST(aFieldRefineOperator, kernelRefineOperatorCanBeCreated)
{
    static constexpr auto dim    = typename TypeParam::first_type{}();
    static constexpr auto interp = typename TypeParam::second_type{}();

    using GridYee = GridLayout<GridLayoutImplYee<dim, interp>>;
    using GridT   = Grid<NdArrayVector<dim>, HybridQuantity::Scalar>;

    auto linearKernel = makeRefineKernel<GridYee, GridT>(2, "none");
    auto cubicKernel  = makeRefineKernel<GridYee, GridT>(4, "none");
    EXPECT_NE(linearKernel, nullptr);
    EXPECT_NE(cubicKernel, nullptr);

    KernelFieldRefineOperator<GridYee, GridT> kernelRefine{linearKernel};

    auto magLinearKernel = makeMagneticRefineKernel<GridYee, GridT>(2, "none");
    auto magCubicKernel  = makeMagneticRefineKernel<GridYee, GridT>(4, "none");
    EXPECT_NE(magLinearKernel, nullptr);
    EXPECT_NE(magCubicKernel, nullptr);

    // limited kernels (minmod / van Leer) now build at both orders 2 and 4
    EXPECT_NE((makeRefineKernel<GridYee, GridT>(2, "minmod")), nullptr);
    EXPECT_NE((makeRefineKernel<GridYee, GridT>(2, "vanleer")), nullptr);
    EXPECT_NE((makeRefineKernel<GridYee, GridT>(4, "minmod")), nullptr);
    EXPECT_NE((makeRefineKernel<GridYee, GridT>(4, "vanleer")), nullptr);
    EXPECT_NE((makeMagneticRefineKernel<GridYee, GridT>(4, "minmod")), nullptr);
    EXPECT_NE((makeMagneticRefineKernel<GridYee, GridT>(4, "vanleer")), nullptr);

    EXPECT_ANY_THROW((makeRefineKernel<GridYee, GridT>(3, "none")));
    EXPECT_ANY_THROW((makeRefineKernel<GridYee, GridT>(2, "superbee")));
    EXPECT_ANY_THROW((makeMagneticRefineKernel<GridYee, GridT>(0, "none")));
    EXPECT_ANY_THROW((makeMagneticRefineKernel<GridYee, GridT>(4, "superbee")));
}


template<typename dimType>
struct aFieldRefine : public testing::Test
{
};

using WithAllDim = testing::Types<DimConst<1>, DimConst<2>, DimConst<3>>;

TYPED_TEST_SUITE(aFieldRefine, WithAllDim);


TYPED_TEST(aFieldRefine, canBeCreated)
{
    static constexpr auto dim = TypeParam{}();

    SAMRAI::tbox::Dimension dimension{dim};
    std::array<QtyCentering, dim> centering = {{QtyCentering::primal}};
    SAMRAI::hier::Box destinationGhostBox{dimension};
    SAMRAI::hier::Box sourceGhostBox{dimension};
    SAMRAI::hier::IntVector ratio{dimension, 2};

    DefaultFieldRefiner<dim> fieldLinearRefine{centering, destinationGhostBox, sourceGhostBox,
                                               ratio};
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------


template<typename dimType>
struct aFieldLinearRefineIndexesAndWeights : public testing::Test
{
};

using WithAllDim = testing::Types<DimConst<1>, DimConst<2>, DimConst<3>>;

TYPED_TEST_SUITE(aFieldLinearRefineIndexesAndWeights, WithAllDim);



template<int dim, int numOfIndexes>
constexpr std::array<Point<int, dim>, numOfIndexes>
makeArrayOfPoints(std::array<int, numOfIndexes> indexVal)
{
    std::array<Point<int, dim>, numOfIndexes> arrayOfPoints{};

    for (auto i = 0u; i < numOfIndexes; ++i)
    {
        int index = indexVal[i];

        arrayOfPoints[i] = ConstArray<int, dim>(index);
    }
    return arrayOfPoints;
}



TYPED_TEST(aFieldLinearRefineIndexesAndWeights, giveACorrectStartIndexForPrimalQty)
{
    static constexpr auto dim = TypeParam{}();

    auto constexpr centering = ConstArray<QtyCentering, dim>(QtyCentering::primal);
    SAMRAI::hier::IntVector ratio{SAMRAI::tbox::Dimension{dim}, 2};
    FieldRefineIndexesAndWeights<dim> indexesAndWeights{centering, ratio};

    constexpr std::array<Point<int, dim>, 4> fineIndexes = makeArrayOfPoints<dim, 4>({-1, 0, 1, 2});
    constexpr std::array<Point<int, dim>, 4> expectedStartIndexes
        = makeArrayOfPoints<dim, 4>({-1, 0, 0, 1});


    for (auto i = 0u; i < fineIndexes.size(); ++i)
    {
        auto fineIndex          = fineIndexes[i];
        auto expectedStartIndex = expectedStartIndexes[i];

        if constexpr (dim == 1)
        {
            auto startIndex = indexesAndWeights.coarseStartIndex(fineIndex);

            EXPECT_EQ(expectedStartIndex[dirX], startIndex[dirX]);
        }
        if constexpr (dim == 2)
        {
            auto startIndex = indexesAndWeights.coarseStartIndex(fineIndex);

            EXPECT_EQ(expectedStartIndex[dirX], startIndex[dirX]);
            EXPECT_EQ(expectedStartIndex[dirY], startIndex[dirY]);
        }
        if constexpr (dim == 3)
        {
            auto startIndex = indexesAndWeights.coarseStartIndex(fineIndex);

            EXPECT_EQ(expectedStartIndex[dirX], startIndex[dirX]);
            EXPECT_EQ(expectedStartIndex[dirY], startIndex[dirY]);
            EXPECT_EQ(expectedStartIndex[dirZ], startIndex[dirZ]);
        }
    }
}


TYPED_TEST(aFieldLinearRefineIndexesAndWeights, giveACorrectStartIndexForDualQty)
{
    static constexpr auto dim = TypeParam{}();

    auto constexpr centering = ConstArray<QtyCentering, dim>(QtyCentering::dual);
    SAMRAI::hier::IntVector ratio{SAMRAI::tbox::Dimension{dim}, 2};
    FieldRefineIndexesAndWeights<dim> indexesAndWeights{centering, ratio};

    constexpr std::array<Point<int, dim>, 4> fineIndexes = makeArrayOfPoints<dim, 4>({-1, 0, 1, 2});
    constexpr std::array<Point<int, dim>, 4> expectedStartIndexes
        = makeArrayOfPoints<dim, 4>({-1, -1, 0, 0});


    for (auto i = 0u; i < fineIndexes.size(); ++i)
    {
        auto fineIndex          = fineIndexes[i];
        auto expectedStartIndex = expectedStartIndexes[i];

        auto startIndex = indexesAndWeights.coarseStartIndex(fineIndex);

        EXPECT_EQ(expectedStartIndex[dirX], startIndex[dirX]);

        if constexpr (dim > 1)
        {
            EXPECT_EQ(expectedStartIndex[dirY], startIndex[dirY]);
        }

        if constexpr (dim > 2)
        {
            EXPECT_EQ(expectedStartIndex[dirZ], startIndex[dirZ]);
        }
    }
}


TYPED_TEST(aFieldLinearRefineIndexesAndWeights, giveACorrectWeightsForPrimalQty)
{
    static constexpr auto dim = TypeParam{}();

    auto constexpr centering = ConstArray<QtyCentering, dim>(QtyCentering::primal);
    SAMRAI::hier::IntVector ratio{SAMRAI::tbox::Dimension{dim}, 2};
    FieldRefineIndexesAndWeights<dim> indexesAndWeights{centering, ratio};

    std::size_t constexpr primal = 0;
    std::size_t constexpr dual   = 1;


    auto xWeights = indexesAndWeights.weights(Direction::X);

    EXPECT_DOUBLE_EQ(xWeights[primal][1], 0.);
    EXPECT_DOUBLE_EQ(xWeights[primal][0], 1.);

    EXPECT_DOUBLE_EQ(xWeights[dual][1], 0.5);
    EXPECT_DOUBLE_EQ(xWeights[dual][0], 0.5);

    if constexpr (dim > 1)
    {
        auto yWeights = indexesAndWeights.weights(Direction::Y);

        EXPECT_DOUBLE_EQ(yWeights[primal][1], 0.);
        EXPECT_DOUBLE_EQ(yWeights[primal][0], 1.);

        EXPECT_DOUBLE_EQ(yWeights[dual][1], 0.5);
        EXPECT_DOUBLE_EQ(yWeights[dual][0], 0.5);
    }
    if constexpr (dim > 2)
    {
        auto zWeights = indexesAndWeights.weights(Direction::Z);

        EXPECT_DOUBLE_EQ(zWeights[primal][1], 0.);
        EXPECT_DOUBLE_EQ(zWeights[primal][0], 1.);

        EXPECT_DOUBLE_EQ(zWeights[dual][1], 0.5);
        EXPECT_DOUBLE_EQ(zWeights[dual][0], 0.5);
    }
}


TYPED_TEST(aFieldLinearRefineIndexesAndWeights, giveACorrectWeightsForDualQty)
{
    static constexpr auto dim = TypeParam{}();

    auto constexpr centering = ConstArray<QtyCentering, dim>(QtyCentering::dual);
    SAMRAI::hier::IntVector ratio{SAMRAI::tbox::Dimension{dim}, 2};
    FieldRefineIndexesAndWeights<dim> indexesAndWeights{centering, ratio};

    std::size_t constexpr primal = 0;
    std::size_t constexpr dual   = 1;


    auto xWeights = indexesAndWeights.weights(Direction::X);

    EXPECT_DOUBLE_EQ(xWeights[primal][1], 0.75);
    EXPECT_DOUBLE_EQ(xWeights[primal][0], 0.25);

    EXPECT_DOUBLE_EQ(xWeights[dual][1], 0.25);
    EXPECT_DOUBLE_EQ(xWeights[dual][0], 0.75);

    if constexpr (dim > 1)
    {
        auto yWeights = indexesAndWeights.weights(Direction::Y);

        EXPECT_DOUBLE_EQ(yWeights[primal][1], 0.75);
        EXPECT_DOUBLE_EQ(yWeights[primal][0], 0.25);

        EXPECT_DOUBLE_EQ(yWeights[dual][1], 0.25);
        EXPECT_DOUBLE_EQ(yWeights[dual][0], 0.75);
    }
    if constexpr (dim > 2)
    {
        auto zWeights = indexesAndWeights.weights(Direction::Z);

        EXPECT_DOUBLE_EQ(zWeights[primal][1], 0.75);
        EXPECT_DOUBLE_EQ(zWeights[primal][0], 0.25);

        EXPECT_DOUBLE_EQ(zWeights[dual][1], 0.25);
        EXPECT_DOUBLE_EQ(zWeights[dual][0], 0.75);
    }
}


TYPED_TEST(aFieldLinearRefineIndexesAndWeights, giveACorrectWeightIndexesForPrimalQty)
{
    static constexpr auto dim = TypeParam{}();

    auto constexpr centering = ConstArray<QtyCentering, dim>(QtyCentering::primal);
    SAMRAI::hier::IntVector ratio{SAMRAI::tbox::Dimension{dim}, 2};
    FieldRefineIndexesAndWeights<dim> indexesAndWeights{centering, ratio};

    constexpr std::array<Point<int, dim>, 4> fineIndexes = makeArrayOfPoints<dim, 4>({-1, 0, 1, 2});
    constexpr std::array<int, 4> expectedWeightIndexes{1, 0, 1, 0};


    for (auto i = 0u; i < fineIndexes.size(); ++i)
    {
        auto fineIndex           = fineIndexes[i];
        auto expectedWeightIndex = expectedWeightIndexes[i];

        auto xWeight = indexesAndWeights.computeWeightIndex(fineIndex)[dirX];

        EXPECT_EQ(expectedWeightIndex, xWeight);

        if constexpr (dim > 1)
        {
            auto yWeight = indexesAndWeights.computeWeightIndex(fineIndex)[dirY];

            EXPECT_EQ(expectedWeightIndex, yWeight);
        }

        if constexpr (dim > 2)
        {
            auto zWeight = indexesAndWeights.computeWeightIndex(fineIndex)[dirZ];

            EXPECT_EQ(expectedWeightIndex, zWeight);
        }
    }
}


TYPED_TEST(aFieldLinearRefineIndexesAndWeights, giveACorrectWeightIndexesForDualQty)
{
    static constexpr auto dim = TypeParam{}();

    auto constexpr centering = ConstArray<QtyCentering, dim>(QtyCentering::dual);
    SAMRAI::hier::IntVector ratio{SAMRAI::tbox::Dimension{dim}, 2};
    FieldRefineIndexesAndWeights<dim> indexesAndWeights{centering, ratio};

    constexpr std::array<Point<int, dim>, 4> fineIndexes = makeArrayOfPoints<dim, 4>({-1, 0, 1, 2});
    constexpr std::array<int, 4> expectedWeightIndexes{1, 0, 1, 0};


    for (auto i = 0u; i < fineIndexes.size(); ++i)
    {
        auto fineIndex           = fineIndexes[i];
        auto expectedWeightIndex = expectedWeightIndexes[i];

        auto xWeight = indexesAndWeights.computeWeightIndex(fineIndex)[dirX];

        EXPECT_EQ(expectedWeightIndex, xWeight);

        if constexpr (dim > 1)
        {
            auto yWeight = indexesAndWeights.computeWeightIndex(fineIndex)[dirY];

            EXPECT_EQ(expectedWeightIndex, yWeight);
        }

        if constexpr (dim > 2)
        {
            auto zWeight = indexesAndWeights.computeWeightIndex(fineIndex)[dirZ];

            EXPECT_EQ(expectedWeightIndex, zWeight);
        }
    }
}




// ----- value-level refineBox tests for the composite (limited) kernel ----------------------------
//
// Boxes are placed with lower=0 so AMR == local indexing; ratio 2. The fine destination is filled
// with NaN so the kernel's NaN-guard writes every targeted index. The numeric core is separable, so
// 1-D exercises both primitives (dual ±¼ ladder, primal half-point) at both orders; one 2-D magnetic
// case covers the sharedFacesOnly skip + the runtime tensor product of limited rows.

namespace
{
    using GridYee1D = GridLayout<GridLayoutImplYee<1, 1>>;
    using Grid1D    = Grid<NdArrayVector<1>, HybridQuantity::Scalar>;
    using GridYee2D = GridLayout<GridLayoutImplYee<2, 1>>;
    using Grid2D    = Grid<NdArrayVector<2>, HybridQuantity::Scalar>;

    template<std::size_t dim>
    SAMRAI::hier::Box boxOf(std::array<int, dim> lo, std::array<int, dim> up)
    {
        SAMRAI::tbox::Dimension d{dim};
        SAMRAI::hier::Index loi{d, 0}, upi{d, 0};
        for (std::size_t k = 0; k < dim; ++k)
        {
            loi(k) = lo[k];
            upi(k) = up[k];
        }
        return SAMRAI::hier::Box{loi, upi, SAMRAI::hier::BlockId{0}};
    }

    SAMRAI::hier::IntVector ratio2(std::size_t dim)
    {
        return SAMRAI::hier::IntVector{SAMRAI::tbox::Dimension{static_cast<unsigned short>(dim)}, 2};
    }

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
} // namespace


// the two fine children of a coarse dual cell always mean back to its average (conservation),
// limited or not, at every order.
TEST(compositeRefiner1D, dualChildrenConserveCoarseAverage)
{
    std::array<double, 8> coarse = {1.0, 1.3, 0.4, 2.0, -0.5, 0.7, 0.9, 0.2};
    std::array<QtyCentering, 1> centering{QtyCentering::dual};

    auto run = [&](auto refiner) {
        Grid1D src{"c", HybridQuantity::Scalar::rho, 8u};
        Grid1D dst{"f", HybridQuantity::Scalar::rho, 16u};
        for (std::size_t i = 0; i < 8; ++i)
            src(i) = coarse[i];
        for (std::size_t i = 0; i < 16; ++i)
            dst(i) = NaN;

        refiner.refineBox(src, dst, boxOf<1>({4}, {11}), centering, boxOf<1>({0}, {15}),
                          boxOf<1>({0}, {7}), ratio2(1));

        for (int I = 2; I <= 5; ++I)
            EXPECT_NEAR(0.5 * (dst(2 * I) + dst(2 * I + 1)), coarse[I], 1e-12);
    };

    run(CompositeFieldRefiner<GridYee1D, Grid1D, 2>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 4>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 2, MinModLimiter>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 4, MinModLimiter>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 2, VanLeerLimiter>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 4, VanLeerLimiter>{});
}


// on a sharp step the unlimited cubic ladder overshoots the coarse bracket; the limited row must
// not create a new extremum.
TEST(compositeRefiner1D, unlimitedDualOvershootsWhereLimitedDoesNot)
{
    std::array<double, 8> coarse = {0, 0, 0, 1, 1, 1, 1, 1};
    std::array<QtyCentering, 1> centering{QtyCentering::dual};

    auto fill7 = [&](auto refiner) {
        Grid1D src{"c", HybridQuantity::Scalar::rho, 8u};
        Grid1D dst{"f", HybridQuantity::Scalar::rho, 16u};
        for (std::size_t i = 0; i < 8; ++i)
            src(i) = coarse[i];
        for (std::size_t i = 0; i < 16; ++i)
            dst(i) = NaN;
        refiner.refineBox(src, dst, boxOf<1>({4}, {11}), centering, boxOf<1>({0}, {15}),
                          boxOf<1>({0}, {7}), ratio2(1));
        return dst(7); // I=3 (C=1), σ=+1 child
    };

    EXPECT_GT(fill7(CompositeFieldRefiner<GridYee1D, Grid1D, 4>{}), 1.0 + 1e-6);
    EXPECT_LE(fill7(CompositeFieldRefiner<GridYee1D, Grid1D, 4, MinModLimiter>{}), 1.0 + 1e-12);
    EXPECT_LE(fill7(CompositeFieldRefiner<GridYee1D, Grid1D, 4, VanLeerLimiter>{}), 1.0 + 1e-12);
}


TEST(compositeRefiner1D, dualLimitedChildrenStayInCoarseBracket)
{
    std::array<double, 8> coarse = {0, 0, 0, 1, 1, 1, 1, 1};
    std::array<QtyCentering, 1> centering{QtyCentering::dual};

    auto run = [&](auto refiner) {
        Grid1D src{"c", HybridQuantity::Scalar::rho, 8u};
        Grid1D dst{"f", HybridQuantity::Scalar::rho, 16u};
        for (std::size_t i = 0; i < 8; ++i)
            src(i) = coarse[i];
        for (std::size_t i = 0; i < 16; ++i)
            dst(i) = NaN;
        refiner.refineBox(src, dst, boxOf<1>({4}, {11}), centering, boxOf<1>({0}, {15}),
                          boxOf<1>({0}, {7}), ratio2(1));
        for (int I = 2; I <= 5; ++I)
        {
            double const lo = std::min({coarse[I - 1], coarse[I], coarse[I + 1]});
            double const hi = std::max({coarse[I - 1], coarse[I], coarse[I + 1]});
            for (int p = 0; p < 2; ++p)
            {
                EXPECT_GE(dst(2 * I + p), lo - 1e-12);
                EXPECT_LE(dst(2 * I + p), hi + 1e-12);
            }
        }
    };

    run(CompositeFieldRefiner<GridYee1D, Grid1D, 2, MinModLimiter>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 4, MinModLimiter>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 2, VanLeerLimiter>{});
    run(CompositeFieldRefiner<GridYee1D, Grid1D, 4, VanLeerLimiter>{});
}


// on a smooth (linear) profile no clipping happens: the limited row recovers the unlimited row
// exactly, preserving the formal order.
TEST(compositeRefiner1D, limitedMatchesUnlimitedOnLinearProfile)
{
    std::array<double, 8> coarse;
    for (int i = 0; i < 8; ++i)
        coarse[i] = 0.5 * i + 0.3;
    std::array<QtyCentering, 1> centering{QtyCentering::dual};

    auto fill = [&](auto refiner) {
        Grid1D src{"c", HybridQuantity::Scalar::rho, 8u};
        Grid1D dst{"f", HybridQuantity::Scalar::rho, 16u};
        for (std::size_t i = 0; i < 8; ++i)
            src(i) = coarse[i];
        for (std::size_t i = 0; i < 16; ++i)
            dst(i) = NaN;
        refiner.refineBox(src, dst, boxOf<1>({4}, {11}), centering, boxOf<1>({0}, {15}),
                          boxOf<1>({0}, {7}), ratio2(1));
        std::array<double, 8> out;
        for (int f = 4; f <= 11; ++f)
            out[f - 4] = dst(f);
        return out;
    };

    auto u2 = fill(CompositeFieldRefiner<GridYee1D, Grid1D, 2>{});
    auto l2 = fill(CompositeFieldRefiner<GridYee1D, Grid1D, 2, MinModLimiter>{});
    auto u4 = fill(CompositeFieldRefiner<GridYee1D, Grid1D, 4>{});
    auto l4 = fill(CompositeFieldRefiner<GridYee1D, Grid1D, 4, VanLeerLimiter>{});
    for (int k = 0; k < 8; ++k)
    {
        EXPECT_NEAR(u2[k], l2[k], 1e-12);
        EXPECT_NEAR(u4[k], l4[k], 1e-12);
    }
}


// order-4 primal half-point has negative outer weights and can overshoot the node bracket; the
// limited row median-clamps it. order-2 (convex 2-pt mean) limiting is a no-op.
TEST(compositeRefiner1D, primalHalfPointLimitedStaysInNodeBracket)
{
    std::array<double, 6> coarse = {0, 0, 0, 1, 1, 1};
    std::array<QtyCentering, 1> centering{QtyCentering::primal};

    auto half = [&](auto refiner) {
        Grid1D src{"c", HybridQuantity::Scalar::rho, 6u};
        Grid1D dst{"f", HybridQuantity::Scalar::rho, 12u};
        for (std::size_t i = 0; i < 6; ++i)
            src(i) = coarse[i];
        for (std::size_t i = 0; i < 12; ++i)
            dst(i) = NaN;
        refiner.refineBox(src, dst, boxOf<1>({6}, {7}), centering, boxOf<1>({0}, {11}),
                          boxOf<1>({0}, {5}), ratio2(1));
        return dst(7); // half-point between nodes 3 and 4 (both = 1)
    };

    EXPECT_GT(half(CompositeFieldRefiner<GridYee1D, Grid1D, 4>{}), 1.0 + 1e-6); // 17/16
    EXPECT_NEAR(half(CompositeFieldRefiner<GridYee1D, Grid1D, 4, MinModLimiter>{}), 1.0, 1e-12);
    EXPECT_NEAR(half(CompositeFieldRefiner<GridYee1D, Grid1D, 4, VanLeerLimiter>{}), 1.0, 1e-12);

    // order 2: half-point limiting self-disables
    EXPECT_NEAR(half(CompositeFieldRefiner<GridYee1D, Grid1D, 2, MinModLimiter>{}),
                half(CompositeFieldRefiner<GridYee1D, Grid1D, 2>{}), 1e-12);
}


// B (Bx: primal-x normal, dual-y tangential), sharedFacesOnly: the two tangential (y) children of a
// shared (even-x) face are antisymmetric about the coarse value ⇒ sum = 2·C ⇒ ∇·B-neutral, even
// limited. Interior (odd-x) faces are left untouched (NaN) by the Tóth-Roe owner.
TEST(magneticCompositeRefiner2D, sharedFaceTangentialChildrenAreDivBNeutral)
{
    auto src_at = [](int ix, int iy) { return 1.0 * ix + 0.5 * iy + 0.1 * ix * iy; };
    std::array<QtyCentering, 2> centering{QtyCentering::primal, QtyCentering::dual}; // Bx

    auto run = [&](auto refiner) {
        Grid2D src{"c", HybridQuantity::Scalar::Bx, 6u, 6u};
        Grid2D dst{"f", HybridQuantity::Scalar::Bx, 12u, 12u};
        for (std::size_t ix = 0; ix < 6; ++ix)
            for (std::size_t iy = 0; iy < 6; ++iy)
                src(ix, iy) = src_at(ix, iy);
        for (std::size_t ix = 0; ix < 12; ++ix)
            for (std::size_t iy = 0; iy < 12; ++iy)
                dst(ix, iy) = NaN;

        refiner.refineBox(src, dst, boxOf<2>({4, 4}, {5, 7}), centering, boxOf<2>({0, 0}, {11, 11}),
                          boxOf<2>({0, 0}, {5, 5}), ratio2(2));

        // fine x=4 is a shared (even) face, anchor Ix=2; x=5 is interior (odd) → skipped (NaN)
        for (int Iy = 2; Iy <= 3; ++Iy)
        {
            double const c0 = dst(4, 2 * Iy);
            double const c1 = dst(4, 2 * Iy + 1);
            EXPECT_NEAR(c0 + c1, 2.0 * src_at(2, Iy), 1e-12);
        }
        EXPECT_TRUE(std::isnan(dst(5, 4))); // interior-normal face untouched
    };

    run(MagneticCompositeRefiner<GridYee2D, Grid2D, 4>{});
    run(MagneticCompositeRefiner<GridYee2D, Grid2D, 4, MinModLimiter>{});
    run(MagneticCompositeRefiner<GridYee2D, Grid2D, 4, VanLeerLimiter>{});
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();


    int testResult = RUN_ALL_TESTS();

    // Finalize
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();

    return testResult;
}
