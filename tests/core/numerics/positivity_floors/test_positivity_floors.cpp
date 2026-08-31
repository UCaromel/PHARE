#include "core/numerics/positivity_floors/positivity_floors.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <limits>
#include <sstream>

using namespace PHARE::core;

namespace
{
// floorScalarInPlace only uses `index` to enrich the debug-throw message (via operator<<); a
// plain int stands in for the MeshIndex used at the real call sites.
constexpr int dummyIndex = 7;

FloorParams enabledParams()
{
    FloorParams p;
    p.enabled        = true;
    p.density_floor  = 1.0e-2;
    p.pressure_floor = 1.0e-2;
    return p;
}

// Every test starts from a clean accumulator: FloorDiagnostics is a per-rank singleton, reset
// once per SolverMHD::advanceLevel in production, but tests share the process.
struct PositivityFloors : public ::testing::Test
{
    void SetUp() override { FloorDiagnostics::instance().reset(); }
};
} // namespace

TEST_F(PositivityFloors, disabledIsInertEvenOnNonFiniteInput)
{
    FloorParams const off{}; // enabled == false
    double x = std::numeric_limits<double>::quiet_NaN();

    EXPECT_FALSE(floorScalarInPlace(x, x, 1.0, FloorSite::ToPrimitive, true, off, dummyIndex));
    EXPECT_TRUE(std::isnan(x)); // untouched

    std::ostringstream oss;
    FloorDiagnostics::instance().report(oss, 0, 0.0);
    EXPECT_EQ(oss.str(), ""); // no accounting at all when disabled
}

TEST_F(PositivityFloors, finiteBelowFloorFiresNormally)
{
    auto const params = enabledParams();
    double x          = 1.0e-4; // below density_floor

    EXPECT_TRUE(floorScalarInPlace(x, x, params.density_floor, FloorSite::ToPrimitive, true, params,
                                   dummyIndex));
    EXPECT_DOUBLE_EQ(x, params.density_floor);

    std::ostringstream oss;
    FloorDiagnostics::instance().report(oss, 3, 12.5);
    auto const report = oss.str();
    EXPECT_NE(report.find("[floor] level=3 t=12.5 site=ToPrimitive qty=rho count=1"),
              std::string::npos);
    EXPECT_EQ(report.find("nan"), std::string::npos);
    EXPECT_EQ(report.find("[floor-nonfinite]"), std::string::npos);
}

TEST_F(PositivityFloors, nanIsRoutedToNonFiniteAccumulatorNotCount)
{
    auto const params = enabledParams();
    double x          = std::numeric_limits<double>::quiet_NaN();

    // Non-debug semantics: floorScalarInPlace itself never throws outside PHARE_DEBUG_DO; here
    // we only assert on the accumulator, the throw path is covered by nonFiniteThrowsUnderDebug
    // below (this build has PHARE_DEBUG_DO active, so we must still catch/allow the throw).
    EXPECT_THROW(
        {
            floorScalarInPlace(x, x, params.density_floor, FloorSite::ToPrimitive, true, params,
                               dummyIndex);
        },
        std::runtime_error);
    EXPECT_TRUE(std::isnan(x)); // never laundered into eps

    std::ostringstream oss;
    FloorDiagnostics::instance().report(oss, 0, 0.0);
    auto const report = oss.str();
    // Recorded on its own line...
    EXPECT_NE(report.find("[floor-nonfinite] level=0 t=0 site=ToPrimitive qty=rho count=1"),
              std::string::npos);
    // ...and NOT in count/sumDelta: no [floor] line was ever emitted for this site/qty, because
    // record() (which increments s.count) was never called.
    EXPECT_EQ(report.find("[floor] "), std::string::npos);
}

TEST_F(PositivityFloors, infinityIsAlsoNonFinite)
{
    auto const params = enabledParams();

    for (double x :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()})
    {
        double v = x;
        EXPECT_THROW(
            {
                floorScalarInPlace(v, v, params.pressure_floor, FloorSite::Reconstruction, false,
                                   params, dummyIndex);
            },
            std::runtime_error);
        EXPECT_TRUE(std::isinf(v));
    }

    std::ostringstream oss;
    FloorDiagnostics::instance().report(oss, 0, 0.0);
    EXPECT_NE(oss.str().find("[floor-nonfinite] level=0 t=0 site=Reconstruction qty=P count=2"),
              std::string::npos);
}

TEST_F(PositivityFloors, nonFiniteEventDoesNotPoisonALaterLegitimateFiringAtTheSameSite)
{
    auto const params = enabledParams();

    double nanX = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        {
            floorScalarInPlace(nanX, nanX, params.density_floor, FloorSite::PostReflux, true,
                               params, dummyIndex);
        },
        std::runtime_error);

    double smallX = 1.0e-5;
    EXPECT_TRUE(floorScalarInPlace(smallX, smallX, params.density_floor, FloorSite::PostReflux,
                                   true, params, dummyIndex));

    std::ostringstream oss;
    FloorDiagnostics::instance().report(oss, 1, 5.0);
    auto const report = oss.str();
    EXPECT_NE(report.find("[floor] level=1 t=5 site=PostReflux qty=rho count=1"),
              std::string::npos);
    EXPECT_EQ(report.find("nan"), std::string::npos); // sumDelta stayed finite
    EXPECT_NE(report.find("[floor-nonfinite] level=1 t=5 site=PostReflux qty=rho count=1"),
              std::string::npos);
}

TEST_F(PositivityFloors, nonFiniteThrowsUnderDebugWithSiteQtyAndIndexInMessage)
{
    auto const params = enabledParams();
    double x          = std::numeric_limits<double>::quiet_NaN();

    try
    {
        floorScalarInPlace(x, x, params.density_floor, FloorSite::HallCT, true, params, dummyIndex);
        FAIL() << "expected floorScalarInPlace to throw on non-finite input";
    }
    catch (std::runtime_error const& ex)
    {
        std::string const what = ex.what();
        EXPECT_NE(what.find("HallCT"), std::string::npos);
        EXPECT_NE(what.find("rho"), std::string::npos);
        EXPECT_NE(what.find(std::to_string(dummyIndex)), std::string::npos);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
