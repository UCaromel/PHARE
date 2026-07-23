
#include "core/def/phare_mpi.hpp"

#include "core/data/grid/grid.hpp"
#include "phare_core.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee.hpp"

#include "amr/data/field/refine/field_linear_refine.hpp"
#include "amr/data/field/refine/field_refine_operator.hpp"
#include "amr/data/field/refine/field_refiner.hpp"
#include "amr/data/field/refine/composite_field_refiner.hpp"
#include "amr/data/field/refine/magnetic_composite_refiner.hpp"
#include "amr/data/field/refine/adpt_magnetic_refine_patch_strategy.hpp"

#include "core/utilities/box/box.hpp"
#include "core/utilities/index/index.hpp"

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

    using GridYee = typename PHARE::core::PHARE_Types<PHARE::SimOpts{dim, interp}>::Hybrid::GridLayout_t;
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

    using GridYee = typename PHARE::core::PHARE_Types<PHARE::SimOpts{dim, interp}>::Hybrid::GridLayout_t;
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
// case covers the runtime tensor product of limited rows on an ungated (fill-all) B component.

namespace
{
    using GridYee1D = typename PHARE::core::PHARE_Types<PHARE::SimOpts{1, 1}>::Hybrid::GridLayout_t;
    using Grid1D    = Grid<NdArrayVector<1>, HybridQuantity::Scalar>;
    using GridYee2D = typename PHARE::core::PHARE_Types<PHARE::SimOpts{2, 1}>::Hybrid::GridLayout_t;
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


// B (Bx: primal-x normal, dual-y tangential): the two tangential (y) children of a shared
// (even-x) face are antisymmetric about the coarse value ⇒ sum = 2·C ⇒ ∇·B-neutral, even limited
// (stage-1 property, unaffected by de-gating). Interior (odd-x, normal-direction) faces are now
// filled too — stage 1 of the ADPT prolongation fills ALL fine faces; ownership of divB-safety
// there moves to the stage-2 touch-up (not exercised by this kernel-only test). For the bilinear
// test data below, the primal-odd (directionalInterp) and dual (directionalProlongation) rows
// reduce exactly to their unlimited form even under limiting: the local slope is the same on both
// sides (affine data) so the limiter clip is a no-op and phi/psi collapse to a self-division ⇒
// exactly 1. So all three refiner variants (none/minmod/vanleer) must match the same
// tensor-product value, computed here from the same primitive weight tables the kernel uses
// (directionalInterp / directionalProlongation), independently of CompositeFieldRefiner's own
// orchestration.
TEST(magneticCompositeRefiner2D, sharedFaceTangentialChildrenAreDivBNeutral)
{
    using ImplYee2D = GridYee2D::implT;

    auto src_at = [](int ix, int iy) { return 1.0 * ix + 0.5 * iy + 0.1 * ix * iy; };
    std::array<QtyCentering, 2> centering{QtyCentering::primal, QtyCentering::dual}; // Bx

    auto rowX = [](int parity) {
        std::vector<WeightPoint<2>> row;
        if (parity == 0)
            row.push_back({Point<int, 2>{}, 1.0});
        else
            for (auto const& w :
                 ImplYee2D::directionalInterp<0, ImplYee2D::InterpDir::PrimalToDual, 4>())
                row.push_back(w);
        return row;
    };
    auto rowY = [](int parity) {
        std::vector<WeightPoint<2>> row;
        if (parity == 0)
            for (auto const& w : ImplYee2D::directionalProlongation<1, -1, 4>())
                row.push_back(w);
        else
            for (auto const& w : ImplYee2D::directionalProlongation<1, +1, 4>())
                row.push_back(w);
        return row;
    };
    auto expectedAt = [&](int Ix, int Iy, int px, int py) {
        double v = 0.0;
        for (auto const& wx : rowX(px))
            for (auto const& wy : rowY(py))
                v += wx.coef * wy.coef * src_at(Ix + wx.indexes[0], Iy + wy.indexes[1]);
        return v;
    };

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

        // fine x=4 is a shared (even) face, anchor Ix=2
        for (int Iy = 2; Iy <= 3; ++Iy)
        {
            double const c0 = dst(4, 2 * Iy);
            double const c1 = dst(4, 2 * Iy + 1);
            EXPECT_NEAR(c0 + c1, 2.0 * src_at(2, Iy), 1e-12);
        }

        // fine x=5 is interior (odd-x, normal-direction), anchor Ix=2: stage 1 now fills it
        // (no more Tóth-Roe-only ownership). Assert filled (non-NaN) and matching the
        // tensor-product stencil value built from the primitive weight tables above.
        for (int iy = 4; iy <= 7; ++iy)
        {
            int const Iy = iy / 2;
            int const py = iy - 2 * Iy;
            EXPECT_FALSE(std::isnan(dst(5, iy)));
            EXPECT_NEAR(dst(5, iy), expectedAt(2, Iy, 1, py), 1e-12);
        }
    };

    run(CompositeFieldRefiner<GridYee2D, Grid2D, 4>{});
    run(CompositeFieldRefiner<GridYee2D, Grid2D, 4, MinModLimiter>{});
    run(CompositeFieldRefiner<GridYee2D, Grid2D, 4, VanLeerLimiter>{});
}


// ----- point-value (PointValue Representation) dual refinement -----------------------------------
//
// The PV dual is a plain point Lagrange at the ±¼ child positions (NON-conservative), as opposed to
// the average world's mean-back ladder. In coarse-index space the σ=+1 child sits at anchor+¼, σ=−1
// at anchor−¼; for ratio 2 fine 2I+1 is σ=+1 and fine 2I is σ=−1 of coarse anchor I. These three
// tests pin the PV-specific behaviour: 4th-order point exactness, ψ-clamp bracketing, and the
// non-mean-back distinguisher against the average world (the key correctness discriminator).

// PV is an MHD-only prolongation world: directionalProlongationPointValue lives only in the MHD Yee
// layout (the hybrid layout stays order-2 average and has no PV seam). So PV refiners instantiate
// against GridLayoutImplYeeMHD, not the hybrid GridLayoutImplYee used by the average tests above.
using GridYeeMHD1D = typename PHARE::core::PHARE_Types<PHARE::SimOpts{1, 2}>::MHD::GridLayout_t;

template<std::size_t order, typename Limiter = NoLimiter>
using PVRefiner1D
    = CompositeFieldRefiner<GridYeeMHD1D, Grid1D, order, Limiter, Representation::PointValue>;


// a degree-4 (5-point) Lagrange stencil reproduces any cubic exactly: the PV order-4 dual children
// equal the underlying cubic sampled at the ±¼ child coordinates (machine zero).
TEST(compositeRefinerPV1D, cubicReconstructedExactlyAtQuarterPoints)
{
    auto g = [](double x) { return 1.0 + 0.7 * x - 0.3 * x * x + 0.2 * x * x * x; };
    std::array<QtyCentering, 1> centering{QtyCentering::dual};

    Grid1D src{"c", HybridQuantity::Scalar::rho, 8u};
    Grid1D dst{"f", HybridQuantity::Scalar::rho, 16u};
    for (std::size_t i = 0; i < 8; ++i)
        src(i) = g(static_cast<double>(i)); // point value at coarse-index coordinate i
    for (std::size_t i = 0; i < 16; ++i)
        dst(i) = NaN;

    PVRefiner1D<4>{}.refineBox(src, dst, boxOf<1>({4}, {11}), centering, boxOf<1>({0}, {15}),
                               boxOf<1>({0}, {7}), ratio2(1));

    for (int I = 2; I <= 5; ++I)
    {
        EXPECT_NEAR(dst(2 * I), g(I - 0.25), 1e-12);     // σ = −1 child at anchor−¼
        EXPECT_NEAR(dst(2 * I + 1), g(I + 0.25), 1e-12); // σ = +1 child at anchor+¼
    }
}


// on a sharp step the unlimited PV ladder overshoots the bracketing nodes; the ψ-clamped row keeps
// every child inside the local coarse bracket (no new extremum).
TEST(compositeRefinerPV1D, dualLimitedChildrenStayInCoarseBracket)
{
    std::array<double, 8> coarse = {0, 0, 0, 1, 1, 1, 1, 1};
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
        return dst;
    };

    // unlimited PV overshoots the [0,1] bracket on the step (fine 7 = I=3, σ=+1 at 3.25)
    EXPECT_GT(fill(PVRefiner1D<4>{})(7), 1.0 + 1e-6);

    auto inBracket = [&](auto dst) {
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

    inBracket(fill(PVRefiner1D<2, MinModLimiter>{}));
    inBracket(fill(PVRefiner1D<4, MinModLimiter>{}));
    inBracket(fill(PVRefiner1D<2, VanLeerLimiter>{}));
    inBracket(fill(PVRefiner1D<4, VanLeerLimiter>{}));
}


// the discriminator vs the average world: on a curved (quadratic) profile the two PV children do
// NOT mean back to the coarse value — they carry the O(H²·u″) curvature the average ladder cancels.
// coarse(i) = i² (u″ = 2): average means back exactly; PV mean = I² + ½·(¼)²·2 = I² + 1/16.
TEST(compositeRefinerPV1D, dualDoesNotMeanBackUnlikeAverage)
{
    std::array<double, 8> coarse;
    for (int i = 0; i < 8; ++i)
        coarse[i] = static_cast<double>(i) * static_cast<double>(i);
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
        return dst;
    };

    auto avg = fill(CompositeFieldRefiner<GridYeeMHD1D, Grid1D, 4>{});
    auto pv  = fill(PVRefiner1D<4>{});

    for (int I = 2; I <= 5; ++I)
    {
        EXPECT_NEAR(0.5 * (avg(2 * I) + avg(2 * I + 1)), coarse[I], 1e-12);
        EXPECT_NEAR(0.5 * (pv(2 * I) + pv(2 * I + 1)), coarse[I] + 1.0 / 16.0, 1e-12);
    }
}



// ==================================================================================================
// ADPTMagneticRefinePatchStrategy stage-2 touch-up tests (Balsara divB-free prolongation)
// ==================================================================================================
//
// Exercises the public static correctBx2d/correctBy2d directly: they are plain static functions,
// so no SAMRAI Patch/ResourcesManager machinery is needed. Stage 1 (CompositeFieldRefiner, reused
// from the value-level tests above) fills every fine face of Bx/By from coarse data; these statics
// then apply the stage-2 divergence-equalizing correction, sharing one DivCache per postprocess
// pass -- the same contract as ADPTMagneticRefinePatchStrategy::postprocessRefine.
//
// The strategy class only reads, from its TensorFieldDataT template parameter, the compile-time
// typedefs used at class scope (Geometry/gridlayout_type/N/dimension); Geometry and ResMan are
// never touched by the statics under test, so both are empty stand-ins.
namespace
{
    struct DummyGeometry2D
    {
    };

    struct DummyTensorFieldData2D
    {
        using Geometry        = DummyGeometry2D;
        using gridlayout_type = GridYee2D;
        static constexpr std::size_t N         = 3;
        static constexpr std::size_t dimension = 2;
    };

    struct DummyResMan2D
    {
    };

    using ADPT2D = ADPTMagneticRefinePatchStrategy<DummyResMan2D, DummyTensorFieldData2D>;

    // A GridLayout whose AMRToLocal is the identity (AMR index == array-local index): avoids
    // depending on GridLayoutImplYee's internal ghost-width value, matching the "lower=0 =>
    // AMR==local" convention the CompositeFieldRefiner value tests above already rely on.
    GridYee2D identityLayout2D(std::array<double, 2> const meshSize = {1., 1.})
    {
        using CoreBox = PHARE::core::Box<int, 2>;
        GridYee2D probe{meshSize, {40u, 40u}, {{0., 0.}}, CoreBox{Point{0, 0}, Point{39, 39}}};
        auto const g0 = probe.AMRToLocal(Point{0, 0});
        int const G   = static_cast<int>(g0[0]);
        return GridYee2D{
            meshSize, {40u, 40u}, {{0., 0.}}, CoreBox{Point{G, G}, Point{G + 39, G + 39}}};
    }

    // raw ("flux") divergence of fine cell (cx,cy): the same quantity the strategy's
    // subzoneDiv2d_ computes internally.
    double rawDiv2d(Grid2D& bx, Grid2D& by, int cx, int cy)
    {
        return (bx(cx + 1, cy) - bx(cx, cy)) + (by(cx, cy + 1) - by(cx, cy));
    }

    Grid2D makeGrid2D(HybridQuantity::Scalar qty, int n)
    {
        return Grid2D{"g", qty, static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n)};
    }

    constexpr int coarseN_ = 12;
    constexpr int fineN_   = 2 * coarseN_;

    // fills every fine face of bx (primal-x/dual-y) and by (dual-x/primal-y) of the fine CELL box
    // [6,17]^2 (coarse cells [3,8]) from full [0,coarseN_)^2 coarse arrays, via
    // CompositeFieldRefiner. Per-component face boxes mirror FieldGeometry::toFieldBox: the primal
    // (face-normal) direction gains the box's high shared face (+1), so every face a subsequent
    // touch-up reads -- including the high subzones' high faces -- is filled, as in production.
    template<int order>
    void fillFaces2D(Grid2D& bxCoarse, Grid2D& byCoarse, Grid2D& bxFine, Grid2D& byFine)
    {
        std::array<QtyCentering, 2> const bxCentering{QtyCentering::primal, QtyCentering::dual};
        std::array<QtyCentering, 2> const byCentering{QtyCentering::dual, QtyCentering::primal};

        auto const bxBox     = boxOf<2>({6, 6}, {18, 17});
        auto const byBox     = boxOf<2>({6, 6}, {17, 18});
        auto const destGhost = boxOf<2>({0, 0}, {fineN_ - 1, fineN_ - 1});
        auto const srcGhost  = boxOf<2>({0, 0}, {coarseN_ - 1, coarseN_ - 1});

        CompositeFieldRefiner<GridYee2D, Grid2D, order> refiner{};
        refiner.refineBox(bxCoarse, bxFine, bxBox, bxCentering, destGhost, srcGhost, ratio2(2));
        refiner.refineBox(byCoarse, byFine, byBox, byCentering, destGhost, srcGhost, ratio2(2));
    }

    // applies the stage-2 touch-up over the same fine box used to fill the faces (matches
    // ADPTMagneticRefinePatchStrategy::postprocessRefine's per-component loop shape: the parity
    // gate inside correctBx2d/correctBy2d selects only the interior/odd faces).
    void touchUp2D(Grid2D& bxFine, Grid2D& byFine, std::array<double, 2> const meshSize = {1., 1.})
    {
        auto const layout  = identityLayout2D(meshSize);
        auto const destBox = boxOf<2>({6, 6}, {17, 17});

        ADPT2D::DivCache cache;
        for (auto const& i : phare_box_from<2>(destBox))
            ADPT2D::correctBx2d(cache, bxFine, byFine, layout, i);
        for (auto const& i : phare_box_from<2>(destBox))
            ADPT2D::correctBy2d(cache, bxFine, byFine, layout, i);
    }

    void fillNaN2D(Grid2D& bxFine, Grid2D& byFine)
    {
        for (int i = 0; i < fineN_; ++i)
            for (int j = 0; j < fineN_; ++j)
            {
                bxFine(i, j) = NaN;
                byFine(i, j) = NaN;
            }
    }
} // namespace


// Coarse B built from a stream function psi (bx = y-difference, by = -x-difference of psi at the
// same 4 corner nodes bounding the cell) is discretely divergence-free by construction, at ANY
// grid resolution, independently of psi's shape -- the corner-algebra telescopes to zero
// regardless of mesh spacing. psi here is a generic (non-affine) cubic, so the order-2 stage-1
// fill is not already exact: the stage-2 touch-up is what must zero the fine divergence.
template<int order>
static void runDivFreeCase()
{
    auto psi = [](int x, int y) {
        return 0.7 * x * x * y - 0.3 * x * y * y + 1.3 * x - 0.6 * y;
    };

    Grid2D bxCoarse = makeGrid2D(HybridQuantity::Scalar::Bx, coarseN_);
    Grid2D byCoarse = makeGrid2D(HybridQuantity::Scalar::By, coarseN_);
    Grid2D bxFine   = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byFine   = makeGrid2D(HybridQuantity::Scalar::By, fineN_);

    for (int I = 0; I < coarseN_; ++I)
        for (int J = 0; J < coarseN_; ++J)
        {
            bxCoarse(I, J) = psi(I, J + 1) - psi(I, J);
            byCoarse(I, J) = -(psi(I + 1, J) - psi(I, J));
        }
    fillNaN2D(bxFine, byFine);

    fillFaces2D<order>(bxCoarse, byCoarse, bxFine, byFine);
    touchUp2D(bxFine, byFine);

    for (int cx = 8; cx <= 15; ++cx)
        for (int cy = 8; cy <= 15; ++cy)
            EXPECT_NEAR(rawDiv2d(bxFine, byFine, cx, cy), 0.0, 1e-12)
                << "order=" << order << " cx=" << cx << " cy=" << cy;
}

TEST(ADPTMagneticTouchUp2D, correctsToExactDivBFreeOnGenericDivFreeCoarseData)
{
    runDivFreeCase<2>();
    runDivFreeCase<4>();
}


// Same flow but the coarse field is NOT divergence-free: the touch-up doesn't (and can't) zero
// out the divergence -- per the class docstring it *equalizes* the 4 fine-subzone divergences of
// the coarse cell they split (they all become the transported zone divergence q0 != 0). Assert
// equality within each complete coarse zone, not zero.
template<int order>
static void runEqualizeCase()
{
    auto bxOf = [](int I, int J) { return I * I - 0.3 * J + 0.2 * I * J; };
    auto byOf = [](int I, int J) { return 0.5 * J * J + 0.1 * I - I * J; };

    Grid2D bxCoarse = makeGrid2D(HybridQuantity::Scalar::Bx, coarseN_);
    Grid2D byCoarse = makeGrid2D(HybridQuantity::Scalar::By, coarseN_);
    Grid2D bxFine   = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byFine   = makeGrid2D(HybridQuantity::Scalar::By, fineN_);

    for (int I = 0; I < coarseN_; ++I)
        for (int J = 0; J < coarseN_; ++J)
        {
            bxCoarse(I, J) = bxOf(I, J);
            byCoarse(I, J) = byOf(I, J);
        }
    fillNaN2D(bxFine, byFine);

    fillFaces2D<order>(bxCoarse, byCoarse, bxFine, byFine);
    touchUp2D(bxFine, byFine);

    for (int I = 4; I <= 7; ++I)
        for (int J = 4; J <= 7; ++J)
        {
            double const d00 = rawDiv2d(bxFine, byFine, 2 * I, 2 * J);
            double const d10 = rawDiv2d(bxFine, byFine, 2 * I + 1, 2 * J);
            double const d01 = rawDiv2d(bxFine, byFine, 2 * I, 2 * J + 1);
            double const d11 = rawDiv2d(bxFine, byFine, 2 * I + 1, 2 * J + 1);

            EXPECT_NEAR(d10, d00, 1e-12) << "order=" << order << " I=" << I << " J=" << J;
            EXPECT_NEAR(d01, d00, 1e-12) << "order=" << order << " I=" << I << " J=" << J;
            EXPECT_NEAR(d11, d00, 1e-12) << "order=" << order << " I=" << I << " J=" << J;
        }
}

TEST(ADPTMagneticTouchUp2D, equalizesSubzoneDivergenceOnGenericNonDivFreeCoarseData)
{
    runEqualizeCase<2>();
    runEqualizeCase<4>();
}


// Order-4 exactness, decomposed to sidestep any cross-level differencing subtlety: bx depends
// only on its own prolongation direction (y) and by only on its own prolongation direction (x),
// each a cubic polynomial of the raw index -- exactly the space a cubic (order-4) prolongation
// stencil is meant to reproduce at any continuous position, including the half-integer fine
// positions. Both components are trivially divergence-free (each raw difference cancels
// identically, being independent of the differencing direction), so stage 1 should already be
// exact and the stage-2 touch-up should be a true no-op.
TEST(ADPTMagneticTouchUp2D, order4TouchUpIsNoOpOnSingleDirectionCubicData)
{
    auto h = [](int J) { return J * J * J - 2.0 * J * J + J + 3.0; }; // bx(I,J) = h(J), indep of I
    auto g = [](int I) { return I * I * I + 2.0 * I * I - I + 1.0; }; // by(I,J) = g(I), indep of J

    Grid2D bxCoarse = makeGrid2D(HybridQuantity::Scalar::Bx, coarseN_);
    Grid2D byCoarse = makeGrid2D(HybridQuantity::Scalar::By, coarseN_);
    Grid2D bxFine   = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byFine   = makeGrid2D(HybridQuantity::Scalar::By, fineN_);

    for (int I = 0; I < coarseN_; ++I)
        for (int J = 0; J < coarseN_; ++J)
        {
            bxCoarse(I, J) = h(J);
            byCoarse(I, J) = g(I);
        }
    fillNaN2D(bxFine, byFine);

    fillFaces2D<4>(bxCoarse, byCoarse, bxFine, byFine);

    constexpr int lo = 8, hi = 15, span = hi - lo + 1;
    std::array<std::array<double, span>, span> bxBefore{}, byBefore{};
    for (int i = lo; i <= hi; ++i)
        for (int j = lo; j <= hi; ++j)
        {
            bxBefore[i - lo][j - lo] = bxFine(i, j);
            byBefore[i - lo][j - lo] = byFine(i, j);
        }

    touchUp2D(bxFine, byFine);

    double maxCorrection = 0.0;
    for (int i = lo; i <= hi; ++i)
        for (int j = lo; j <= hi; ++j)
        {
            maxCorrection = std::max(maxCorrection, std::abs(bxFine(i, j) - bxBefore[i - lo][j - lo]));
            maxCorrection = std::max(maxCorrection, std::abs(byFine(i, j) - byBefore[i - lo][j - lo]));
        }

    for (int cx = lo; cx <= hi - 1; ++cx) // divergence reads cx+1, cy+1: stay one cell further in
        for (int cy = lo; cy <= hi - 1; ++cy)
            EXPECT_NEAR(rawDiv2d(bxFine, byFine, cx, cy), 0.0, 1e-12) << "cx=" << cx << " cy=" << cy;

    EXPECT_NEAR(maxCorrection, 0.0, 1e-13) << "max stage-2 correction magnitude on data that "
                                               "should already lie in the order-4 reproducing "
                                               "space";
}


// Full order-4 cubic exactness in the face-average (FV) convention -- the strongest form of the
// S8 claim: coarse faces carry the EXACT face-averages of a generic divergence-free cubic B
// (all cross terms live, built as curl of a generic quartic stream function), and after the
// Cubic4 stage-1 fill + stage-2 touch-up every fine face must carry the exact fine face-average
// of the same field. Even fine indices pin the kernel's shared-face convention (a point-value
// kernel deviates at O(dx^2) here); odd indices pin the interior reconstruction + zero-correction
// touch-up. Coordinates are centered/scaled so B stays O(1) and 1e-12 is a relative-tight bound.
TEST(ADPTMagneticTouchUp2D, order4ReproducesFaceAveragesOfGenericDivFreeCubic)
{
    auto psi = [](double x, double y) {
        double const u = (x - 6.0) / 4.0;
        double const v = (y - 6.0) / 4.0;
        return 0.31 * u * u * u * u - 0.47 * u * u * u * v + 0.23 * u * u * v * v
               + 0.59 * u * v * v * v - 0.37 * v * v * v * v + 0.83 * u * u * v - 0.29 * u * v * v
               + 0.41 * u * u - 0.53 * u * v + 0.67 * v * v + 1.3 * u - 0.6 * v;
    };
    // exact tangential face-average of (bx,by) = (dpsi/dy, -dpsi/dx) over a face of width h
    auto bxAvg = [&](double x, double y, double h) { return (psi(x, y + h) - psi(x, y)) / h; };
    auto byAvg = [&](double x, double y, double h) { return -(psi(x + h, y) - psi(x, y)) / h; };

    Grid2D bxCoarse = makeGrid2D(HybridQuantity::Scalar::Bx, coarseN_);
    Grid2D byCoarse = makeGrid2D(HybridQuantity::Scalar::By, coarseN_);
    Grid2D bxFine   = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byFine   = makeGrid2D(HybridQuantity::Scalar::By, fineN_);

    for (int I = 0; I < coarseN_; ++I)
        for (int J = 0; J < coarseN_; ++J)
        {
            bxCoarse(I, J) = bxAvg(I, J, 1.0);
            byCoarse(I, J) = byAvg(I, J, 1.0);
        }
    fillNaN2D(bxFine, byFine);

    fillFaces2D<4>(bxCoarse, byCoarse, bxFine, byFine);
    touchUp2D(bxFine, byFine);

    for (int i = 8; i <= 15; ++i)
        for (int j = 8; j <= 15; ++j)
        {
            EXPECT_NEAR(bxFine(i, j), bxAvg(i / 2.0, j / 2.0, 0.5), 1e-12)
                << "bx i=" << i << " j=" << j << (i % 2 ? " (interior)" : " (shared)");
            EXPECT_NEAR(byFine(i, j), byAvg(i / 2.0, j / 2.0, 0.5), 1e-12)
                << "by i=" << i << " j=" << j << (j % 2 ? " (interior)" : " (shared)");
        }
}


// Equal-mesh reduction identity (aniso derivation §S15, C++ gate 1): on dx = dy = h the weighted
// operator is algebraically the historical unweighted one — the uniform scale cancels between the
// 1/h-weighted subzone divergence and the h prefactor. The h = 1 run IS the historical operator
// (weights and prefactor literally 1), so agreement with a generic h to roundoff proves the
// reduction. Generic non-div-free coarse data so every correction is non-trivial.
TEST(ADPTMagneticTouchUp2D, equalMeshTouchUpIsIndependentOfUniformMeshScale)
{
    auto bxOf = [](int I, int J) { return I * I - 0.3 * J + 0.2 * I * J; };
    auto byOf = [](int I, int J) { return 0.5 * J * J + 0.1 * I - I * J; };

    Grid2D bxCoarse = makeGrid2D(HybridQuantity::Scalar::Bx, coarseN_);
    Grid2D byCoarse = makeGrid2D(HybridQuantity::Scalar::By, coarseN_);
    Grid2D bxA      = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byA      = makeGrid2D(HybridQuantity::Scalar::By, fineN_);
    Grid2D bxB      = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byB      = makeGrid2D(HybridQuantity::Scalar::By, fineN_);

    for (int I = 0; I < coarseN_; ++I)
        for (int J = 0; J < coarseN_; ++J)
        {
            bxCoarse(I, J) = bxOf(I, J);
            byCoarse(I, J) = byOf(I, J);
        }
    fillNaN2D(bxA, byA);
    fillNaN2D(bxB, byB);

    fillFaces2D<2>(bxCoarse, byCoarse, bxA, byA);
    fillFaces2D<2>(bxCoarse, byCoarse, bxB, byB);

    touchUp2D(bxA, byA);                // h = 1: the historical unweighted operator
    touchUp2D(bxB, byB, {0.37, 0.37});  // generic equal mesh

    // fillFaces2D fills full per-component field boxes, so every face the touch-up reads is
    // filled and all corrected faces (odd normal index in [7,17]) compare finitely.
    for (int i = 6; i <= 17; ++i)
        for (int j = 6; j <= 17; ++j)
        {
            EXPECT_NEAR(bxB(i, j), bxA(i, j), 1e-12) << "bx i=" << i << " j=" << j;
            EXPECT_NEAR(byB(i, j), byA(i, j), 1e-12) << "by i=" << i << " j=" << j;
        }
}


// Anisotropic divB exactness (aniso derivation §S15, C++ gate 2, unit form): coarse B is
// PHYSICALLY divergence-free on an anisotropic mesh (stream-function construction carries the
// 1/dy, 1/dx factors), and after stage-1 fill + weighted touch-up every fine PHYSICAL subzone
// divergence must vanish to roundoff (s11b check 4 / s12 e2e). The historical unweighted operator
// fails this — asserted below to keep the test discriminating.
template<int order>
static void runAnisoDivFreeCase()
{
    double const dx = 1.0;
    double const dy = 0.55;
    auto psi        = [](int x, int y) {
        return 0.7 * x * x * y - 0.3 * x * y * y + 1.3 * x - 0.6 * y;
    };

    auto physDiv = [&](Grid2D& bx, Grid2D& by, int cx, int cy) {
        return (bx(cx + 1, cy) - bx(cx, cy)) / dx + (by(cx, cy + 1) - by(cx, cy)) / dy;
    };

    Grid2D bxCoarse = makeGrid2D(HybridQuantity::Scalar::Bx, coarseN_);
    Grid2D byCoarse = makeGrid2D(HybridQuantity::Scalar::By, coarseN_);

    for (int I = 0; I < coarseN_; ++I)
        for (int J = 0; J < coarseN_; ++J)
        {
            bxCoarse(I, J) = (psi(I, J + 1) - psi(I, J)) / dy;
            byCoarse(I, J) = -(psi(I + 1, J) - psi(I, J)) / dx;
        }

    // weighted touch-up: physical divergence zero to roundoff
    Grid2D bxFine = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byFine = makeGrid2D(HybridQuantity::Scalar::By, fineN_);
    fillNaN2D(bxFine, byFine);
    fillFaces2D<order>(bxCoarse, byCoarse, bxFine, byFine);
    touchUp2D(bxFine, byFine, {dx, dy});

    for (int cx = 6; cx <= 17; ++cx)
        for (int cy = 6; cy <= 17; ++cy)
            EXPECT_NEAR(physDiv(bxFine, byFine, cx, cy), 0.0, 1e-11)
                << "order=" << order << " cx=" << cx << " cy=" << cy;

    // historical (equal-mesh) touch-up on the same stage-1 data: physical divergence does NOT
    // vanish — the defect this generalisation fixes
    Grid2D bxOld = makeGrid2D(HybridQuantity::Scalar::Bx, fineN_);
    Grid2D byOld = makeGrid2D(HybridQuantity::Scalar::By, fineN_);
    fillNaN2D(bxOld, byOld);
    fillFaces2D<order>(bxCoarse, byCoarse, bxOld, byOld);
    touchUp2D(bxOld, byOld);

    double maxOldDiv = 0.0;
    for (int cx = 6; cx <= 17; ++cx)
        for (int cy = 6; cy <= 17; ++cy)
            maxOldDiv = std::max(maxOldDiv, std::abs(physDiv(bxOld, byOld, cx, cy)));
    EXPECT_GT(maxOldDiv, 1e-3) << "order=" << order;
}

TEST(ADPTMagneticTouchUp2D, correctsToExactPhysicalDivBFreeOnAnisotropicMesh)
{
    runAnisoDivFreeCase<2>();
    runAnisoDivFreeCase<4>();
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
