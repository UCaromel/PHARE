#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"
#include "core/numerics/point_values_handler/point_value_converter.hpp"
#include "core/numerics/point_values_handler/troubled_cell_detector.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace PHARE::core
{
namespace
{
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double pi  = std::numbers::pi;

    constexpr std::uint32_t nCells = 32;
    constexpr double meshSize      = 1. / nCells;
    constexpr std::uint32_t jump   = nCells / 2; // first physical cell of the right state, along x

    constexpr SimOpts fourthOrder(std::size_t const dim)
    {
        return SimOpts{dim,
                       0,
                       0,
                       MHDOpts::MHDOrder::O4,
                       MHDOpts::ReconstructionType::WENOZ,
                       MHDOpts::SlopeLimiterType::None,
                       MHDOpts::RiemannSolverType::Rusanov};
    }

    template<std::size_t dim>
    struct Types
    {
        using Model    = PHARE_Types<fourthOrder(dim)>::MHD;
        using Layout   = Model::GridLayout_t;
        using Grid     = Model::Grid_t;
        using Field    = Model::Field_t;
        using VecField = Model::VecField_t;
    };

    template<typename T>
    struct OwnedField
    {
        OwnedField(typename T::Layout const& layout, MHDQuantity::Scalar const qty)
            : grid{"f", layout, qty, NaN}
            , field{"f", qty}
        {
            field.setBuffer(&grid);
        }

        OwnedField(OwnedField const&)            = delete;
        OwnedField& operator=(OwnedField const&) = delete;

        typename T::Grid grid;
        typename T::Field field;
    };

    template<typename T>
    struct OwnedVecField
    {
        OwnedVecField(typename T::Layout const& layout, MHDQuantity::Vector const qty)
            : view{"v", qty}
            , grids{make_(layout, 0), make_(layout, 1), make_(layout, 2)}
        {
            view.setBuffer(&grids);
        }

        OwnedVecField(OwnedVecField const&)            = delete;
        OwnedVecField& operator=(OwnedVecField const&) = delete;

        typename T::VecField view;
        std::array<typename T::Grid, 3> grids;

    private:
        auto make_(typename T::Layout const& layout, std::size_t const i) const
        {
            return typename T::Grid{view[i].name(), layout, view[i].physicalQuantity(), NaN};
        }
    };

    template<typename T>
    auto makeLayout()
    {
        constexpr auto dim = T::Layout::dimension;
        std::array<double, dim> dl;
        std::array<std::uint32_t, dim> cells;
        dl.fill(meshSize);
        cells.fill(nCells);
        return typename T::Layout{dl, cells, Point<double, dim>{}};
    }

    // position of a local index in units of the domain length, 0 on the lower physical boundary
    template<typename Layout, typename Field>
    auto position(Layout const& layout, Field const& field, MeshIndex<Layout::dimension> const& idx)
    {
        auto const centering = layout.centering(field.physicalQuantity());
        std::array<double, Layout::dimension> x;
        for (std::size_t d = 0; d < Layout::dimension; ++d)
        {
            auto const dir   = static_cast<Direction>(d);
            auto const start = static_cast<double>(layout.physicalStartIndex(centering[d], dir));
            auto const shift = centering[d] == QtyCentering::dual ? 0.5 : 0.;
            x[d]             = (static_cast<double>(idx[d]) - start + shift) * meshSize;
        }
        return x;
    }

    // fn(position, first index) on the whole ghost box
    template<typename Layout, typename Field, typename Fn>
    void fill(Layout const& layout, Field& field, Fn&& fn)
    {
        layout.evalOnGhostBox(field, [&](auto&... args) mutable {
            MeshIndex<Layout::dimension> const idx{args...};
            field(idx) = fn(position(layout, field, idx), idx[0]);
        });
    }

    // a wave along the diagonal, with extrema inside the domain
    template<std::size_t dim>
    double wave(std::array<double, dim> const& x)
    {
        double phase = 0.3;
        for (std::size_t d = 0; d < dim; ++d)
            phase += 2. * pi * x[d];
        return std::sin(phase);
    }

    template<typename Layout>
    bool isRightState(Layout const& layout, std::size_t const xIndex)
    {
        return xIndex >= layout.physicalStartIndex(QtyCentering::dual, Direction::X) + jump;
    }

    // the primitives of a state, the detector's scratch and its flags
    template<typename T>
    struct Primitives
    {
        explicit Primitives(typename T::Layout const& layout)
            : rho{layout, MHDQuantity::Scalar::rho}
            , P{layout, MHDQuantity::Scalar::P}
            , V{layout, MHDQuantity::Vector::V}
            , sensed{layout, MHDQuantity::Scalar::rho}
            , troubled{layout, MHDQuantity::Scalar::rho}
        {
        }

        void detect(typename T::Layout const& layout, ConversionLimiterInfo const& info)
        {
            TroubledCellDetector<typename T::Layout>{layout, info}(rho.field, V.view, P.field,
                                                                   sensed.field, troubled.field);
        }

        OwnedField<T> rho, P;
        OwnedVecField<T> V;
        OwnedField<T> sensed, troubled;
    };

    ConversionLimiterInfo limiter(TroubledCellDetectorType const detector)
    {
        return {detector, ConversionLimiterInfo::defaultThreshold(detector), true};
    }

    constexpr std::array detectors{TroubledCellDetectorType::Jameson,
                                   TroubledCellDetectorType::DerivativeRatio};
} // namespace


TEST(TroubledCellSensors, areSilentOnAResolvedWaveExtremaIncluded)
{
    auto const u = [](int const i) { return 1. + 0.5 * std::sin(2. * pi * i / nCells); };

    for (int i = 0; i < static_cast<int>(nCells); ++i)
    {
        EXPECT_LT(jamesonSensor(u(i - 1), u(i), u(i + 1)),
                  ConversionLimiterInfo::defaultThreshold(TroubledCellDetectorType::Jameson));
        EXPECT_LT(
            derivativeRatioSensor({u(i - 2), u(i - 1), u(i), u(i + 1), u(i + 2)}, u(i)),
            ConversionLimiterInfo::defaultThreshold(TroubledCellDetectorType::DerivativeRatio));
    }
}

TEST(TroubledCellSensors, fireOnAJump)
{
    EXPECT_GT(jamesonSensor(1., 1., 10.), 0.05);
    EXPECT_GT(jamesonSensor(1., 10., 10.), 0.05);
    EXPECT_DOUBLE_EQ(jamesonSensor(1., 1., 1.), 0.);

    // every cell whose five-point stencil holds the jump
    EXPECT_GT(derivativeRatioSensor({1., 1., 1., 1., 10.}, 1.), 0.3);
    EXPECT_GT(derivativeRatioSensor({1., 1., 1., 10., 10.}, 1.), 0.3);
    EXPECT_GT(derivativeRatioSensor({1., 1., 10., 10., 10.}, 10.), 0.3);
    EXPECT_GT(derivativeRatioSensor({1., 10., 10., 10., 10.}, 10.), 0.3);
    EXPECT_DOUBLE_EQ(derivativeRatioSensor({1., 1., 1., 1., 1.}, 1.), 0.);
}

TEST(ConversionLimiterInfo, isReadFromTheAlgoDict)
{
    initializer::PHAREDict dict;
    EXPECT_FALSE(ConversionLimiterInfo::FROM(dict).enabled());
    EXPECT_FALSE(ConversionLimiterInfo::FROM(dict).skipsFluxAveraging());

    dict["conversion_limiter"]["detector"] = std::string{"jameson"};
    auto const jameson                     = ConversionLimiterInfo::FROM(dict);
    EXPECT_EQ(jameson.detector, TroubledCellDetectorType::Jameson);
    EXPECT_DOUBLE_EQ(jameson.threshold, 0.05);
    EXPECT_TRUE(jameson.skipsFluxAveraging());

    dict["conversion_limiter"]["detector"]            = std::string{"derivative_ratio"};
    dict["conversion_limiter"]["flux_averaging_skip"] = false;
    auto const ratio                                  = ConversionLimiterInfo::FROM(dict);
    EXPECT_EQ(ratio.detector, TroubledCellDetectorType::DerivativeRatio);
    EXPECT_DOUBLE_EQ(ratio.threshold, 0.3);
    EXPECT_TRUE(ratio.enabled());
    EXPECT_FALSE(ratio.skipsFluxAveraging());

    dict["conversion_limiter"]["threshold"] = 0.2;
    EXPECT_DOUBLE_EQ(ConversionLimiterInfo::FROM(dict).threshold, 0.2);

    EXPECT_FALSE(ConversionLimiterInfo::FROM(dict).limitsHydroOnly());
    dict["conversion_limiter"]["variables"] = std::string{"hydro"};
    EXPECT_TRUE(ConversionLimiterInfo::FROM(dict).limitsHydroOnly());
    dict["conversion_limiter"]["variables"] = std::string{"magnetic"};
    EXPECT_THROW(ConversionLimiterInfo::FROM(dict), std::runtime_error);
    dict["conversion_limiter"]["variables"] = std::string{"all"};

    dict["conversion_limiter"]["detector"] = std::string{"minmod"};
    EXPECT_THROW(ConversionLimiterInfo::FROM(dict), std::runtime_error);
}


template<typename T>
struct TroubledCells : public ::testing::Test
{
    using Layout                    = typename T::Layout;
    static constexpr auto dimension = Layout::dimension;

    Layout layout = makeLayout<T>();
};

using Dimensions = ::testing::Types<Types<1>, Types<2>, Types<3>>;
TYPED_TEST_SUITE(TroubledCells, Dimensions);


TYPED_TEST(TroubledCells, noneOnASmoothStateAndGhostsAreLeftAlone)
{
    constexpr auto dim = TestFixture::dimension;
    auto const& layout = this->layout;

    for (auto const detector : detectors)
    {
        Primitives<TypeParam> s{layout};
        fill(layout, s.rho.field, [](auto const& x, auto) { return 1. + 0.3 * wave<dim>(x); });
        fill(layout, s.P.field, [](auto const& x, auto) { return 1. - 0.3 * wave<dim>(x); });
        for (auto& component : s.V.view)
            fill(layout, component, [](auto const& x, auto) { return 0.5 * wave<dim>(x); });

        s.detect(layout, limiter(detector));

        std::size_t flagged = 0, cells = 0;
        layout.evalOnBox(s.troubled.field, [&](auto&... args) mutable {
            ++cells;
            flagged += s.troubled.field(args...) != 0.;
        });
        EXPECT_EQ(flagged, 0u);

        std::size_t untouched = 0;
        for (auto const v : s.troubled.grid)
            untouched += std::isnan(v);
        EXPECT_EQ(untouched, s.troubled.grid.size() - cells);
    }
}

TYPED_TEST(TroubledCells, areTheCellsThatSenseADensityJumpAndTheirFaceNeighbours)
{
    auto const& layout = this->layout;
    auto const first   = layout.physicalStartIndex(QtyCentering::dual, Direction::X) + jump;

    for (auto const detector : detectors)
    {
        Primitives<TypeParam> s{layout};
        fill(layout, s.rho.field,
             [&](auto const&, auto const i) { return isRightState(layout, i) ? 10. : 1.; });
        fill(layout, s.P.field, [](auto const&, auto) { return 1.; });
        for (auto& component : s.V.view)
            fill(layout, component, [](auto const&, auto) { return 0.; });

        s.detect(layout, limiter(detector));

        // cells sensing the jump on each side of it, plus the face neighbour
        auto const width
            = TroubledCellDetector<typename TypeParam::Layout>::sensorReach(detector) + 1;
        layout.evalOnBox(s.troubled.field, [&](auto&... args) mutable {
            MeshIndex<TestFixture::dimension> const idx{args...};
            bool const expected = idx[0] + width >= first && idx[0] < first + width;
            EXPECT_EQ(s.troubled.field(idx), expected ? 1. : 0.) << "x index " << idx[0];
        });
    }
}

TYPED_TEST(TroubledCells, pressureAndVelocityJumpsAreSensed)
{
    auto const& layout = this->layout;
    auto const first   = layout.physicalStartIndex(QtyCentering::dual, Direction::X) + jump;

    auto const flaggedAtJump = [&](auto const detector, bool const inPressure) {
        // 1 everywhere, or stepping to 5 at the jump
        auto const profile = [&](bool const stepped) {
            return [&layout, stepped](auto const&, auto const i) {
                return stepped && isRightState(layout, i) ? 5. : 1.;
            };
        };

        Primitives<TypeParam> s{layout};
        fill(layout, s.rho.field, profile(false));
        fill(layout, s.P.field, profile(inPressure));
        for (auto& component : s.V.view)
            fill(layout, component, profile(!inPressure));
        s.detect(layout, limiter(detector));

        MeshIndex<TestFixture::dimension> idx;
        for (std::size_t d = 0; d < TestFixture::dimension; ++d)
            idx[d] = layout.physicalStartIndex(QtyCentering::dual, static_cast<Direction>(d)) + 3;
        idx[0] = first;
        return s.troubled.field(idx) == 1.;
    };

    EXPECT_TRUE(flaggedAtJump(TroubledCellDetectorType::Jameson, true));
    EXPECT_TRUE(flaggedAtJump(TroubledCellDetectorType::DerivativeRatio, true));
    // Jameson senses rho and P only, the derivative ratio V too
    EXPECT_FALSE(flaggedAtJump(TroubledCellDetectorType::Jameson, false));
    EXPECT_TRUE(flaggedAtJump(TroubledCellDetectorType::DerivativeRatio, false));
}


// what PointValueConverter::toPointValues reads and writes
template<typename T>
struct ConservedFields
{
    explicit ConservedFields(typename T::Layout const& layout)
        : rhoOwner{layout, MHDQuantity::Scalar::rho}
        , EtotOwner{layout, MHDQuantity::Scalar::Etot}
        , troubledOwner{layout, MHDQuantity::Scalar::rho}
        , magneticEnergyOwner{layout, MHDQuantity::Scalar::rho}
        , hydroEnergyOwner{layout, MHDQuantity::Scalar::rho}
        , rhoVOwner{layout, MHDQuantity::Vector::rhoV}
        , BOwner{layout, MHDQuantity::Vector::B}
        , rho{rhoOwner.field}
        , Etot{EtotOwner.field}
        , troubled{troubledOwner.field}
        , magneticEnergy{magneticEnergyOwner.field}
        , hydroEnergy{hydroEnergyOwner.field}
        , rhoV{rhoVOwner.view}
        , B{BOwner.view}
    {
    }

    OwnedField<T> rhoOwner, EtotOwner, troubledOwner, magneticEnergyOwner, hydroEnergyOwner;
    OwnedVecField<T> rhoVOwner, BOwner;
    typename T::Field &rho, &Etot, &troubled, &magneticEnergy, &hydroEnergy;
    typename T::VecField &rhoV, &B;
};

TYPED_TEST(TroubledCells, keepTheirAverageAsPointValueAndSoDoTheirFaces)
{
    constexpr auto dim = TestFixture::dimension;
    using Layout       = typename TypeParam::Layout;
    auto const& layout = this->layout;

    ConservedFields<TypeParam> state{layout}, pv{layout}, unlimited{layout};
    auto const profile = [](auto const& x, auto) { return 1. + 0.3 * wave<dim>(x); };
    fill(layout, state.rho, profile);
    fill(layout, state.Etot, profile);
    for (auto& component : state.rhoV)
        fill(layout, component, profile);
    for (auto& component : state.B)
        fill(layout, component, profile);

    // one troubled cell, away from the boundaries
    MeshIndex<dim> troubledCell;
    for (std::size_t d = 0; d < dim; ++d)
        troubledCell[d]
            = layout.physicalStartIndex(QtyCentering::dual, static_cast<Direction>(d)) + 7 + d;
    fill(layout, pv.troubled, [](auto const&, auto) { return 0.; });
    pv.troubled(troubledCell) = 1.;

    PointValueConverter<Layout>{layout}.toPointValues(unlimited, state);
    PointValueConverter<Layout>{layout, limiter(TroubledCellDetectorType::Jameson)}.toPointValues(
        pv, state);

    // cells: the average in the troubled cell, the unlimited conversion in every other
    Point<std::uint32_t, dim> oneLayer;
    for (std::size_t d = 0; d < dim; ++d)
        oneLayer[d] = 1;
    std::size_t averaged = 0;
    layout.evalOnShrinkedGhostBox(state.rho, oneLayer, [&](auto&... args) mutable {
        MeshIndex<dim> const idx{args...};
        if (idx == troubledCell)
        {
            ++averaged;
            EXPECT_EQ(pv.rho(idx), state.rho(idx));
            EXPECT_EQ(pv.Etot(idx), state.Etot(idx));
            EXPECT_EQ(pv.rhoV(Component::Y)(idx), state.rhoV(Component::Y)(idx));
            EXPECT_NE(unlimited.rho(idx), state.rho(idx));
        }
        else
        {
            EXPECT_EQ(pv.rho(idx), unlimited.rho(idx));
            EXPECT_EQ(pv.Etot(idx), unlimited.Etot(idx));
        }
    });
    EXPECT_EQ(averaged, 1u);

    // B: the two faces of the troubled cell along a mesh direction, the cell itself along the
    // others, where B sits at the cell centre
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const component = static_cast<Component>(c);
        auto const& avg      = state.B(component);
        auto const& limited  = pv.B(component);
        auto const& free     = unlimited.B(component);

        auto shrink = oneLayer;
        if (c < dim)
            shrink[c] = 0;
        std::size_t kept = 0;
        layout.evalOnShrinkedGhostBox(avg, shrink, [&](auto&... args) mutable {
            MeshIndex<dim> const idx{args...};
            auto cell    = idx;
            bool touches = idx == troubledCell;
            if (c < dim && idx[c] > 0)
            {
                --cell[c];
                touches = touches || cell == troubledCell;
            }
            if (touches)
            {
                ++kept;
                EXPECT_EQ(limited(idx), avg(idx));
            }
            else
                EXPECT_EQ(limited(idx), free(idx));
        });
        EXPECT_EQ(kept, c < dim ? 2u : 1u);
    }
}

// The Hydro form converts Etot through e_h = Etot - magnetic energy in every cell, limits rho, rhoV
// and e_h in the troubled ones, and never B.
TYPED_TEST(TroubledCells, hydroFormLimitsTheHydrodynamicEnergyAndNeverB)
{
    constexpr auto dim = TestFixture::dimension;
    using Layout       = typename TypeParam::Layout;
    auto const& layout = this->layout;

    ConservedFields<TypeParam> state{layout}, pv{layout}, unlimited{layout};
    auto const profile = [](auto const& x, auto) { return 1. + 0.3 * wave<dim>(x); };
    fill(layout, state.rho, profile);
    fill(layout, state.Etot, [](auto const& x, auto) { return 3. + 0.3 * wave<dim>(x); });
    for (auto& component : state.rhoV)
        fill(layout, component, profile);
    for (auto& component : state.B)
        fill(layout, component, profile);

    MeshIndex<dim> troubledCell;
    for (std::size_t d = 0; d < dim; ++d)
        troubledCell[d]
            = layout.physicalStartIndex(QtyCentering::dual, static_cast<Direction>(d)) + 7 + d;
    fill(layout, pv.troubled, [](auto const&, auto) { return 0.; });
    pv.troubled(troubledCell) = 1.;

    auto hydro      = limiter(TroubledCellDetectorType::Jameson);
    hydro.variables = LimitedVariables::Hydro;
    PointValueConverter<Layout>{layout}.toPointValues(unlimited, state);
    PointValueConverter<Layout>{layout, hydro}.toPointValues(pv, state);

    // B is the unlimited conversion on every face
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const component  = static_cast<Component>(c);
        std::size_t differing = 0, converted = 0;
        for (std::size_t i = 0; i < pv.BOwner.grids[c].size(); ++i)
        {
            auto const limited = pv.BOwner.grids[c].data()[i];
            auto const free    = unlimited.BOwner.grids[c].data()[i];
            converted += !std::isnan(free);
            differing += !(std::isnan(limited) && std::isnan(free)) && limited != free;
        }
        EXPECT_GT(converted, 0u) << "B component " << c;
        EXPECT_EQ(differing, 0u) << "B component " << c;
    }

    // Etot_pv = e_h + m in the troubled cell, e_h - lapl(e_h)/24 + m in the others: known three
    // layers inside the ghost box, and fourth-order close to the direct conversion where smooth
    Point<std::uint32_t, dim> threeLayers;
    for (std::size_t d = 0; d < dim; ++d)
        threeLayers[d] = 3;
    double farthest = 0.;
    layout.evalOnShrinkedGhostBox(state.Etot, threeLayers, [&](auto&... args) mutable {
        MeshIndex<dim> const idx{args...};
        auto const m  = pv.magneticEnergy(idx);
        auto const eh = pv.hydroEnergy(idx);
        if (idx == troubledCell)
        {
            EXPECT_EQ(pv.Etot(idx), eh + m);
            EXPECT_EQ(pv.rho(idx), state.rho(idx));
        }
        else
        {
            EXPECT_EQ(pv.Etot(idx), eh - layout.lapl(pv.hydroEnergy, idx) / 24. + m);
            farthest = std::max(farthest, std::abs(pv.Etot(idx) - unlimited.Etot(idx)));
        }
    });
    EXPECT_GT(farthest, 0.);
    EXPECT_LT(farthest, 1.e-3); // amplitude 0.3, 32 cells per wavelength

    // the outer layers of Etot_pv are not written
    std::size_t unwritten = 0;
    for (auto const v : pv.EtotOwner.grid)
        unwritten += std::isnan(v);
    EXPECT_GT(unwritten, 0u);
}

TYPED_TEST(TroubledCells, flagTheirFacesUpToTheEdgeOfTheGhostBox)
{
    constexpr auto dim = TestFixture::dimension;
    auto const& layout = this->layout;

    OwnedField<TypeParam> troubled{layout, MHDQuantity::Scalar::rho};
    fill(layout, troubled.field, [](auto const&, auto) { return 1.; });

    // every cell is troubled: so is every face, the outermost ones with their single cell
    OwnedField<TypeParam> faces{layout, MHDQuantity::Scalar::Bx};
    std::size_t notTroubled = 0;
    layout.evalOnGhostBox(faces.field, [&](auto&... args) mutable {
        notTroubled
            += !faceIsTroubled<Direction::X>(layout, troubled.field, MeshIndex<dim>{args...});
    });
    EXPECT_EQ(notTroubled, 0u);
}
} // namespace PHARE::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
