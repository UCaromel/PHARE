// Gate 4/5 for the S3 floor (Hall CT density denominators, see
// ~/.claude/plans/ucaromel/prod-ho-refinement-ct-vtkhdf/floors-followup-nonfinite-and-s3-prompt.md):
//
// Drives the REAL core::UpwindConstrainedTransport::operator() (2D, Hall on) with a hand-built
// ct_state whose rhot_x/rhot_y cross zero, and checks that Ez never goes non-finite and that
// |Ez| stays within the 1/eps_rho amplification bound documented in
// upwind_constrained_transport.hpp's EzEq_ Hall block. It must call the actual production floor
// calls (not a re-implementation) so that stubbing them to identity makes this test fail — see
// hallCTFloorIsNotVacuous below (gate 5, negative control).
//
// vt (upwind velocity) and dL/dR (CT weighting) are set to zero everywhere so the non-Hall part
// of Ez collapses to exactly zero, isolating the Hall term's own numerator/denominator algebra:
//   Ez = 0.5*(jxW*ByW/rhoW + jxE*ByE/rhoE) - 0.5*(jyS*BxS/rhoS + jyN*BxN/rhoN)
// with B == J == 1 everywhere and aW==aE==aS==aN==0.5, so with a floored rho >= eps_rho each of
// the four 1/rho terms is bounded by 1/eps_rho and |Ez| <= 2/eps_rho.

#include "phare_core.hpp"

#include "core/numerics/constrained_transport/upwind_constrained_transport.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/data/grid/grid.hpp"

#include "initializer/data_provider.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

using namespace PHARE::core;
using PHARE::initializer::PHAREDict;

namespace
{
constexpr std::size_t dim    = 2;
constexpr std::size_t interp = 1;

// reconstruction_type must be WENOZ to match CT_t's explicit WENOZReconstruction below --
// GridLayout_t's ghost width is sized from this SimOpts, and WENOZ needs a wider stencil than
// the Default reconstruction this layout would otherwise be sized for (OOB reads into B
// otherwise).
using Types        = PHARE::core::PHARE_Types<PHARE::SimOpts{.dimension    = dim,
                                                             .interp_order = interp,
                                                             .reconstruction_type
                                                      = PHARE::MHDOpts::ReconstructionType::WENOZ}>;
using GridLayout_t = Types::MHD::GridLayout_t;
using VecField_t   = Types::MHD::VecField_t;
using Array_t      = Types::Array_t;
using Grid_t       = Grid<Array_t, MHDQuantity::Scalar>;

using CT_t      = UpwindConstrainedTransport<GridLayout_t, WENOZReconstruction, /*Hall=*/true,
                                             /*Resistivity=*/false, /*HyperResistivity=*/false>;
using CTState_t = UpwindConstrainedTransportState<VecField_t, /*Hall=*/true,
                                                  /*Resistivity=*/false>;

GridLayout_t makeLayout()
{
    return {{{1.0, 1.0}}, {{16, 16}}, {0., 0.}};
}

// Allocates real storage for one scalar Field member (owns the Grid_t; caller must keep the
// returned pointer alive for as long as `f` is used).
std::unique_ptr<Grid_t> allocateScalar(auto& f, GridLayout_t const& layout)
{
    auto storage = std::make_unique<Grid_t>(f.name(), f.physicalQuantity(),
                                            layout.allocSize(f.physicalQuantity()));
    f.setBuffer(&(*storage));
    return storage;
}

// Same, for one rank-1 TensorField (VecField) member: allocates the 3 components individually.
std::vector<std::unique_ptr<Grid_t>> allocateVector(auto& tf, GridLayout_t const& layout)
{
    std::vector<std::unique_ptr<Grid_t>> storage;
    for (auto const component : {Component::X, Component::Y, Component::Z})
    {
        auto const qty = MHDQuantity::componentsQuantities(
            tf.physicalQuantity())[static_cast<std::size_t>(component)];
        storage.push_back(
            std::make_unique<Grid_t>(tf.getComponentName(component), qty, layout.allocSize(qty)));
        // Not tf.getComponent(component): TensorField::getComponent() gates on isUsable(),
        // which requires every component to already have a buffer -- a chicken-and-egg problem
        // during setup. operator[] bypasses that check.
        tf[static_cast<std::size_t>(component)].setBuffer(&(*storage.back()));
    }
    return storage;
}

void fillScalar(auto& f, double value)
{
    auto const shape = f.shape();
    for (std::size_t i = 0; i < shape[0]; ++i)
        for (std::size_t j = 0; j < shape[1]; ++j)
            f(i, j) = value;
}

void fillVector(auto& tf, double value)
{
    for (auto const component : {Component::X, Component::Y, Component::Z})
        fillScalar(tf.getComponent(component), value);
}

// rhot crossing zero: mostly +1 (well clear of the floor), with a couple of deeply negative
// rows/columns straddling the middle of the domain so the second (raw-scheme) reconstruction in
// EzEq_'s Hall block sees a genuine sign change, not just a small dip.
void fillCrossingZero(auto& f, std::size_t badLo, std::size_t badHi, bool badAlongJ,
                      double badValue)
{
    auto const shape = f.shape();
    for (std::size_t i = 0; i < shape[0]; ++i)
        for (std::size_t j = 0; j < shape[1]; ++j)
        {
            auto const idx = badAlongJ ? j : i;
            f(i, j)        = (idx >= badLo && idx <= badHi) ? badValue : 1.0;
        }
}

FloorParams enabledFloor(double epsRho)
{
    FloorParams p;
    p.enabled        = true;
    p.density_floor  = epsRho;
    p.pressure_floor = 0.0; // unused by CT
    return p;
}

struct FakeMHDState
{
    VecField_t E{"E", MHDQuantity::Vector::E};
    VecField_t B{"B", MHDQuantity::Vector::B};
};

// Builds a fully-allocated ct_state + mhd_state, isolates the Hall term as described in the
// file comment above, and runs the real CT operator. Returns the resulting Ez field alongside
// the layout so the caller can inspect it (all backing storage stays alive via `storage`).
struct Fixture
{
    GridLayout_t layout = makeLayout();
    CTState_t ct_state;
    FakeMHDState mhd_state;

    // owning storage for every buffer allocated above; order doesn't matter, only lifetime.
    std::vector<std::unique_ptr<Grid_t>> scalarStorage;
    std::vector<std::vector<std::unique_ptr<Grid_t>>> vectorStorage;

    explicit Fixture(double epsRho)
    {
        auto keepVec = [&](auto& tf) { vectorStorage.push_back(allocateVector(tf, layout)); };
        auto keepSc  = [&](auto& f) { scalarStorage.push_back(allocateScalar(f, layout)); };

        keepVec(ct_state.vt_x);
        keepVec(ct_state.vt_y);
        keepVec(ct_state.jt_x);
        keepVec(ct_state.jt_y);
        keepSc(ct_state.aL_x);
        keepSc(ct_state.aR_x);
        keepSc(ct_state.dL_x);
        keepSc(ct_state.dR_x);
        keepSc(ct_state.aL_y);
        keepSc(ct_state.aR_y);
        keepSc(ct_state.dL_y);
        keepSc(ct_state.dR_y);
        keepSc(ct_state.rhot_x);
        keepSc(ct_state.rhot_y);
        keepSc(ct_state.Bt_z_at_Ey);
        keepSc(ct_state.Bt_y_at_Ez);
        keepSc(ct_state.Bt_x_at_Ez);
        keepSc(ct_state.Bt_z_at_Ex);
        keepVec(mhd_state.B);
        keepVec(mhd_state.E);

        // Non-Hall part of Ez collapses to zero: vt == 0 kills the upwind flux term, dL/dR == 0
        // kills the CT correction term.
        fillVector(ct_state.vt_x, 0.0);
        fillVector(ct_state.vt_y, 0.0);
        fillScalar(ct_state.dL_x, 0.0);
        fillScalar(ct_state.dR_x, 0.0);
        fillScalar(ct_state.dL_y, 0.0);
        fillScalar(ct_state.dR_y, 0.0);
        fillScalar(ct_state.aL_x, 0.5);
        fillScalar(ct_state.aR_x, 0.5);
        fillScalar(ct_state.aL_y, 0.5);
        fillScalar(ct_state.aR_y, 0.5);

        fillVector(ct_state.jt_x, 1.0);
        fillVector(ct_state.jt_y, 1.0);
        fillVector(mhd_state.B, 1.0);

        // rhot crossing zero: bad rows straddling the middle of the domain (indices 7-8 of 16).
        // Bad value scaled to epsRho (not a flat -1.0): the bound this test checks
        // (2/epsRho) is only discriminating when the un-floored reconstructed density is
        // itself within a small multiple of epsRho -- an O(1) step under/overshoots to an
        // O(1) reconstructed value, whose reciprocal trivially clears a 1/epsRho-scale bound.
        fillCrossingZero(ct_state.rhot_x, 7, 8, /*badAlongJ=*/true, -epsRho);
        fillCrossingZero(ct_state.rhot_y, 7, 8, /*badAlongJ=*/false, -epsRho);
    }
};
} // namespace

TEST(HallCTFloor, keepsEzFiniteAndBoundedWhenRhotCrossesZero)
{
    double const epsRho = 1.0e-2;
    Fixture fx{epsRho};

    auto const info
        = ConstrainedTransportInfo_t{{0.0, 0.0, HyperMode::constant}, enabledFloor(epsRho)};
    CT_t ct{info, fx.layout};

    ct(fx.ct_state, fx.mhd_state);

    auto const& Ez     = fx.mhd_state.E(Component::Z);
    auto const psi_X   = fx.layout.physicalStartIndex(Ez, Direction::X);
    auto const pei_X   = fx.layout.physicalEndIndex(Ez, Direction::X);
    auto const psi_Y   = fx.layout.physicalStartIndex(Ez, Direction::Y);
    auto const pei_Y   = fx.layout.physicalEndIndex(Ez, Direction::Y);
    double const bound = 2.0 / epsRho; // see file header derivation

    std::size_t checked = 0;
    for (auto ix = psi_X; ix <= pei_X; ++ix)
    {
        for (auto iy = psi_Y; iy <= pei_Y; ++iy)
        {
            SCOPED_TRACE(::testing::Message() << "ix=" << ix << " iy=" << iy);
            EXPECT_TRUE(std::isfinite(Ez(ix, iy)));
            EXPECT_LE(std::abs(Ez(ix, iy)), bound);
            ++checked;
        }
    }
    // Sanity: the loop above actually iterated over the domain (a botched layout could silently
    // iterate zero times and let a broken test pass for the wrong reason).
    EXPECT_GT(checked, 0u);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
