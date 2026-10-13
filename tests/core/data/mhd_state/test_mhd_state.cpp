
#include "phare_core.hpp"
#include "core/data/field/field.hpp"
#include "core/data/field/initializers/field_user_initializer.hpp"
#include "core/models/mhd_state.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include "initializer/data_provider.hpp"

#include "tests/core/data/mhd_state/init_functions.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"

#include <cmath>
#include <numbers>

#include "gtest/gtest.h"

using namespace PHARE::core;
using namespace PHARE::initializer;
using namespace PHARE::initializer::test_fn::func_1d;

using Field_t    = Field<1, MHDQuantity::Scalar>;
using VecField1D = VecField<Field_t, MHDQuantity>;


PHAREDict getDict()
{
    using initfunc = InitFunction<1>;
    PHAREDict dict;

    dict["name"] = std::string("state");

    dict["density"]["initializer"] = static_cast<initfunc>(density);

    dict["rhoV"]["initializer"]["x_component"] = static_cast<initfunc>(rhoVx);
    dict["rhoV"]["initializer"]["y_component"] = static_cast<initfunc>(rhoVy);
    dict["rhoV"]["initializer"]["z_component"] = static_cast<initfunc>(rhoVz);

    dict["magnetic"]["initializer"]["x_component"] = static_cast<initfunc>(bx);
    dict["magnetic"]["initializer"]["y_component"] = static_cast<initfunc>(by);
    dict["magnetic"]["initializer"]["z_component"] = static_cast<initfunc>(bz);

    dict["Etot"]["initializer"] = static_cast<initfunc>(etot);

    return dict;
}

struct AnMHDState : public ::testing::Test
{
    MHDState<VecField1D> state{getDict()};
    virtual ~AnMHDState();
};

AnMHDState::~AnMHDState() {}

TEST_F(AnMHDState, noUsableFieldsUponConstruction)
{
    EXPECT_FALSE(state.isUsable());
}

TEST_F(AnMHDState, fieldsAreSettable)
{
    EXPECT_TRUE(state.isSettable());
}

TEST_F(AnMHDState, hasTupleResourceList)
{
    auto resources              = state.getCompileTimeResourcesViewList();
    [[maybe_unused]] auto& rho  = std::get<0>(resources);
    [[maybe_unused]] auto& v    = std::get<1>(resources);
    [[maybe_unused]] auto& b    = std::get<2>(resources);
    [[maybe_unused]] auto& p    = std::get<3>(resources);
    [[maybe_unused]] auto& rhov = std::get<4>(resources);
    [[maybe_unused]] auto& etot = std::get<5>(resources);
    [[maybe_unused]] auto& j    = std::get<6>(resources);
    [[maybe_unused]] auto& e    = std::get<7>(resources);

    EXPECT_FALSE(rho.isUsable());
    EXPECT_FALSE(v.isUsable());
    EXPECT_FALSE(b.isUsable());
    EXPECT_FALSE(p.isUsable());
    EXPECT_FALSE(rhov.isUsable());
    EXPECT_FALSE(etot.isUsable());
    EXPECT_FALSE(j.isUsable());
    EXPECT_FALSE(e.isUsable());
}

// The initializer stores what the field means on the grid: a cell average must match the exact
// average of the user function over the element the field lives on (a cell for rho, a face for
// Bx), to fourth order -- point sampling there is only second order and would cap an O4 run.
namespace
{
constexpr double kx = 2. * std::numbers::pi;
constexpr double ky = 4. * std::numbers::pi;

// f(x, y) = sin(kx x) cos(ky y)
PHARE::initializer::InitFunction<2> const profile = [](auto const& x, auto const& y) {
    std::vector<double> v(x.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = std::sin(kx * x[i]) * std::cos(ky * y[i]);
    return std::make_shared<PHARE::core::VectorSpan<double>>(std::move(v));
};

// exact average of sin(k s) / cos(k s) over [c - h/2, c + h/2]; h == 0 is the point value
double avgSin(double c, double h, double k)
{
    return h == 0. ? std::sin(k * c)
                   : (std::cos(k * (c - h / 2)) - std::cos(k * (c + h / 2))) / (k * h);
}
double avgCos(double c, double h, double k)
{
    return h == 0. ? std::cos(k * c)
                   : (std::sin(k * (c + h / 2)) - std::sin(k * (c - h / 2))) / (k * h);
}

template<InitRepresentation Representation>
double maxInitError(std::uint32_t const n, MHDQuantity::Scalar const qty)
{
    static constexpr PHARE::SimOpts opts{2,
                                         0,
                                         0,
                                         PHARE::MHDOpts::MHDOrder::O4,
                                         PHARE::MHDOpts::ReconstructionType::WENOZ,
                                         PHARE::MHDOpts::SlopeLimiterType::None,
                                         PHARE::MHDOpts::RiemannSolverType::Rusanov};
    using Layout = PHARE_Types<opts>::MHD::GridLayout_t;

    double const dl = 1. / n;
    Layout layout{{dl, dl}, {n, n}, Point{0., 0.}};
    UsableFieldMHD<2> field{"f", layout, qty};
    FieldUserFunctionInitializer::initialize<Representation>(field.super(), layout, profile);

    auto const centering = layout.centering(qty);
    // the reference is the exact average over the element: a dual direction spans a cell width,
    // a primal one is a point
    auto const width = [&](auto dir) { return centering[dir] == QtyCentering::dual ? dl : 0.; };

    double err = 0.;
    for (auto i = layout.physicalStartIndex(centering[0], Direction::X);
         i <= layout.physicalEndIndex(centering[0], Direction::X); ++i)
        for (auto j = layout.physicalStartIndex(centering[1], Direction::Y);
             j <= layout.physicalEndIndex(centering[1], Direction::Y); ++j)
        {
            auto const c
                = layout.fieldNodeCoordinates(field.super(), layout.localToAMR(Point{i, j}));
            auto const reference = avgSin(c[0], width(0), kx) * avgCos(c[1], width(1), ky);
            err                  = std::max(err, std::abs(field.super()(i, j) - reference));
        }
    return err;
}

template<InitRepresentation Representation>
double initOrder(MHDQuantity::Scalar const qty)
{
    return std::log2(maxInitError<Representation>(32, qty) / maxInitError<Representation>(64, qty));
}
} // namespace

TEST(FieldUserFunctionInitializer, CellAverageIsFourthOrderOnCellsAndFaces)
{
    for (auto const qty : {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Bx})
        EXPECT_GT(initOrder<InitRepresentation::CellAverage>(qty), 3.8);
}

// negative control: sampling at the node is only second order against the same reference
TEST(FieldUserFunctionInitializer, PointValueIsSecondOrderAgainstAverages)
{
    for (auto const qty : {MHDQuantity::Scalar::rho, MHDQuantity::Scalar::Bx})
    {
        auto const order = initOrder<InitRepresentation::PointValue>(qty);
        EXPECT_GT(order, 1.8);
        EXPECT_LT(order, 2.2);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
