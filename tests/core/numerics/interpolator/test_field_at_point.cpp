#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>

#include "core/data/field/field.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/numerics/interpolator/field_at_point.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/point/point.hpp"

using namespace PHARE::core;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Fill every node of @p field (including ghost cells) with the value returned
 * by @p func evaluated at the node's physical coordinates.  Uses
 * @p layout.fieldNodeCoordinates to obtain those coordinates.
 */
template<typename GridLayout, typename Field, typename Func>
void fillWithAnalytic(GridLayout const& layout, Field& field, Func func)
{
    constexpr auto dim = GridLayout::dimension;
    if constexpr (dim == 1)
    {
        for (auto i = 0u; i < field.size(); ++i)
        {
            auto const amr_u = layout.localToAMR(Point<std::uint32_t, 1>{i});
            auto const amr_i = Point<int, 1>{static_cast<int>(amr_u[0])};
            auto const pos   = layout.fieldNodeCoordinates(field, amr_i);
            field(i)         = func(pos);
        }
    }
    else if constexpr (dim == 2)
    {
        for (auto i = 0u; i < field.shape()[0]; ++i)
            for (auto j = 0u; j < field.shape()[1]; ++j)
            {
                auto const amr_u  = layout.localToAMR(Point<std::uint32_t, 2>{i, j});
                auto const amr_ij = Point<int, 2>{static_cast<int>(amr_u[0]),
                                                  static_cast<int>(amr_u[1])};
                auto const pos    = layout.fieldNodeCoordinates(field, amr_ij);
                field(i, j)       = func(pos);
            }
    }
}

// ---------------------------------------------------------------------------
// 1D tests
// ---------------------------------------------------------------------------

template<std::size_t interpOrder>
struct FieldAtPoint1D
{
    static constexpr std::size_t dim  = 1;
    static constexpr double dx        = 0.5;
    static constexpr std::uint32_t nx = 20;

    using GLImpl  = GridLayoutImplYeeMHD<dim, interpOrder, 0>;
    using GL      = GridLayout<GLImpl>;
    using Scalar  = MHDQuantity::Scalar;
    using FieldT  = Field<dim, Scalar, double>;
    using Storage = NdArrayVector<dim, double>;
    using FAP     = FieldAtPoint<dim, interpOrder>;

    FieldAtPoint1D()
        : layout{{dx}, {nx}, {0.}, Box<int, dim>{Point{0}, Point{(int)nx - 1}}}
        , node_storage{layout.allocSize(Scalar::NodeCentered)}
        , cell_storage{layout.allocSize(Scalar::CellCentered)}
        , node_field{"phi", Scalar::NodeCentered, node_storage.data(), node_storage.shape()}
        , cell_field{"rho", Scalar::CellCentered, cell_storage.data(), cell_storage.shape()}
    {
        // f(x) = 2*x + 1
        auto linear = [](auto const& p) { return 2. * p[0] + 1.; };
        fillWithAnalytic(layout, node_field, linear);
        fillWithAnalytic(layout, cell_field, linear);
    }

    GL      layout;
    Storage node_storage;
    Storage cell_storage;
    FieldT  node_field;
    FieldT  cell_field;
    FAP     fap;
};

using FieldAtPoint1DOrder1 = FieldAtPoint1D<1>;
using FieldAtPoint1DOrder2 = FieldAtPoint1D<2>;
using FieldAtPoint1DOrder3 = FieldAtPoint1D<3>;

/**
 * For any interpolation order, a linear function must be reproduced exactly.
 * Test node-centred (all-primal) fields.
 */
TEST(FieldAtPoint1DOrder1, NodeCentredLinearExact)
{
    FieldAtPoint1D<1> fix;
    constexpr double tol = 1e-12;
    for (double x : {1.0, 2.3, 4.75, 7.0, 9.5})
    {
        double expected = 2. * x + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::NodeCentered>(
            fix.layout, fix.node_field, Point<double, 1>{x});
        EXPECT_NEAR(got, expected, tol) << "x = " << x;
    }
}

TEST(FieldAtPoint1DOrder2, NodeCentredLinearExact)
{
    FieldAtPoint1D<2> fix;
    constexpr double tol = 1e-10;
    for (double x : {1.0, 2.3, 4.75, 7.0, 9.5})
    {
        double expected = 2. * x + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::NodeCentered>(
            fix.layout, fix.node_field, Point<double, 1>{x});
        EXPECT_NEAR(got, expected, tol) << "x = " << x;
    }
}

TEST(FieldAtPoint1DOrder3, NodeCentredLinearExact)
{
    FieldAtPoint1D<3> fix;
    constexpr double tol = 1e-9;
    for (double x : {1.0, 2.3, 4.75, 7.0, 9.5})
    {
        double expected = 2. * x + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::NodeCentered>(
            fix.layout, fix.node_field, Point<double, 1>{x});
        EXPECT_NEAR(got, expected, tol) << "x = " << x;
    }
}

/**
 * Same for cell-centred (all-dual) fields.
 */
TEST(FieldAtPoint1DOrder1, CellCentredLinearExact)
{
    FieldAtPoint1D<1> fix;
    constexpr double tol = 1e-12;
    for (double x : {1.0, 2.3, 4.75, 7.0, 9.5})
    {
        double expected = 2. * x + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::CellCentered>(
            fix.layout, fix.cell_field, Point<double, 1>{x});
        EXPECT_NEAR(got, expected, tol) << "x = " << x;
    }
}

TEST(FieldAtPoint1DOrder2, CellCentredLinearExact)
{
    FieldAtPoint1D<2> fix;
    constexpr double tol = 1e-10;
    for (double x : {1.0, 2.3, 4.75, 7.0, 9.5})
    {
        double expected = 2. * x + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::CellCentered>(
            fix.layout, fix.cell_field, Point<double, 1>{x});
        EXPECT_NEAR(got, expected, tol) << "x = " << x;
    }
}

// ---------------------------------------------------------------------------
// 2D tests
// ---------------------------------------------------------------------------

template<std::size_t interpOrder>
struct FieldAtPoint2D
{
    static constexpr std::size_t dim  = 2;
    static constexpr double dx        = 0.5;
    static constexpr double dy        = 0.5;
    static constexpr std::uint32_t nx = 20;
    static constexpr std::uint32_t ny = 20;

    using GLImpl  = GridLayoutImplYeeMHD<dim, interpOrder, 0>;
    using GL      = GridLayout<GLImpl>;
    using Scalar  = MHDQuantity::Scalar;
    using FieldT  = Field<dim, Scalar, double>;
    using Storage = NdArrayVector<dim, double>;
    using FAP     = FieldAtPoint<dim, interpOrder>;

    FieldAtPoint2D()
        : layout{{dx, dy}, {nx, ny}, {0., 0.},
                 Box<int, dim>{Point{0, 0}, Point{(int)nx - 1, (int)ny - 1}}}
        , node_storage{layout.allocSize(Scalar::NodeCentered)}
        , cell_storage{layout.allocSize(Scalar::CellCentered)}
        , face_x_storage{layout.allocSize(Scalar::FaceCenteredX)}
        , node_field{"phi", Scalar::NodeCentered, node_storage.data(), node_storage.shape()}
        , cell_field{"rho", Scalar::CellCentered, cell_storage.data(), cell_storage.shape()}
        , face_x_field{"Bx", Scalar::FaceCenteredX, face_x_storage.data(),
                       face_x_storage.shape()}
    {
        // f(x, y) = 3*x + 2*y + 1  (bilinear — reproduced exactly by order-1 interpolation)
        auto bilinear = [](auto const& p) { return 3. * p[0] + 2. * p[1] + 1.; };
        fillWithAnalytic(layout, node_field, bilinear);
        fillWithAnalytic(layout, cell_field, bilinear);
        fillWithAnalytic(layout, face_x_field, bilinear);
    }

    GL      layout;
    Storage node_storage;
    Storage cell_storage;
    Storage face_x_storage;
    FieldT  node_field;
    FieldT  cell_field;
    FieldT  face_x_field;
    FAP     fap;
};

/**
 * Bilinear function reproduced exactly by order-1 interpolation for all centerings.
 */
TEST(FieldAtPoint2DOrder1, NodeCentredBilinearExact)
{
    FieldAtPoint2D<1> fix;
    constexpr double tol = 1e-11;
    for (auto [x, y] : std::array<std::pair<double, double>, 4>{
             {{1.0, 1.0}, {2.3, 3.7}, {5.1, 7.4}, {9.0, 9.0}}})
    {
        double expected = 3. * x + 2. * y + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::NodeCentered>(
            fix.layout, fix.node_field, Point<double, 2>{x, y});
        EXPECT_NEAR(got, expected, tol) << "x=" << x << " y=" << y;
    }
}

TEST(FieldAtPoint2DOrder1, CellCentredBilinearExact)
{
    FieldAtPoint2D<1> fix;
    constexpr double tol = 1e-11;
    for (auto [x, y] : std::array<std::pair<double, double>, 4>{
             {{1.0, 1.0}, {2.3, 3.7}, {5.1, 7.4}, {9.0, 9.0}}})
    {
        double expected = 3. * x + 2. * y + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::CellCentered>(
            fix.layout, fix.cell_field, Point<double, 2>{x, y});
        EXPECT_NEAR(got, expected, tol) << "x=" << x << " y=" << y;
    }
}

TEST(FieldAtPoint2DOrder1, FaceCentredXBilinearExact)
{
    FieldAtPoint2D<1> fix;
    constexpr double tol = 1e-11;
    // FaceCenteredX is dual-X, primal-Y: avoid ghost cell boundaries
    for (auto [x, y] : std::array<std::pair<double, double>, 4>{
             {{1.0, 1.0}, {2.3, 3.7}, {5.1, 7.4}, {8.0, 8.0}}})
    {
        double expected = 3. * x + 2. * y + 1.;
        double got = fix.fap.template operator()<MHDQuantity::Scalar::FaceCenteredX>(
            fix.layout, fix.face_x_field, Point<double, 2>{x, y});
        EXPECT_NEAR(got, expected, tol) << "x=" << x << " y=" << y;
    }
}

// ---------------------------------------------------------------------------
// pointIsInterpolable: primal vs dual margin
// ---------------------------------------------------------------------------

/**
 * @brief The stencil reserves 1 cell of slack at each end of the ghost box for
 * dual-centred directions but 0 for primal-centred ones. So a point one cell
 * outside the ghost box upper bound is interpolable for primal but not for
 * dual along that direction. The sibling-component check in
 * Field{,Anti}SymmetricInnerBoundaryCondition relies on this asymmetry.
 */
TEST(FieldAtPointInterpolable, primalAndDualDifferAtSlackMargin)
{
    using GridLayoutImpl = GridLayoutImplYeeMHD<2, 2, 1>;
    using GridLayoutT    = GridLayout<GridLayoutImpl>;

    Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayoutT layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    auto const nG = static_cast<int>(layout.nbrGhosts());
    ASSERT_GE(nG, 1);

    // Interior point: inside both primal and dual ghost boxes.
    Point<double, 2> interior{0.5, 0.5};
    EXPECT_TRUE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, interior, {QtyCentering::primal, QtyCentering::primal})));
    EXPECT_TRUE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, interior, {QtyCentering::dual, QtyCentering::dual})));

    // Point in the dual-margin slack along x: floor(x/dx) == amr_box.lower[0] - nG.
    // Primal margin 0 -> still interpolable; dual margin 1 -> rejected.
    double const x_slack = (amr_box.lower[0] - nG) + 0.5; // dx = 1
    Point<double, 2> slack_x{x_slack, 0.5};
    EXPECT_TRUE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, slack_x, {QtyCentering::primal, QtyCentering::primal})));
    EXPECT_FALSE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, slack_x, {QtyCentering::dual, QtyCentering::primal})));

    // Same idea on the upper-y end: floor(y/dy) == amr_box.upper[1] + nG.
    double const y_slack = (amr_box.upper[1] + nG) + 0.5;
    Point<double, 2> slack_y{0.5, y_slack};
    EXPECT_TRUE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, slack_y, {QtyCentering::primal, QtyCentering::primal})));
    EXPECT_FALSE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, slack_y, {QtyCentering::primal, QtyCentering::dual})));

    // Mixed centering: primal in slack direction, dual orthogonal -> accepted.
    EXPECT_TRUE((FieldAtPoint<2, 1>::pointIsInterpolable(
        layout, slack_x, {QtyCentering::primal, QtyCentering::dual})));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
