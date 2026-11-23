#include "core/numerics/reconstructions/weno3.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <vector>

using namespace PHARE::core;

namespace
{
// 1D layout stub: the reconstructions only need neighbour navigation and projection
struct Layout1D
{
    static constexpr std::size_t dimension = 1;

    template<auto>
    static MeshIndex<1> next(MeshIndex<1> i)
    {
        return MeshIndex<1>{i[0] + 1};
    }

    template<auto>
    static MeshIndex<1> previous(MeshIndex<1> i)
    {
        return MeshIndex<1>{i[0] - 1};
    }

    template<auto, auto>
    static double project(auto const& U, MeshIndex<1> i)
    {
        return U(i);
    }
};

struct Field1D
{
    static constexpr std::size_t dimension = 1;
    std::vector<double> v;
    double operator()(MeshIndex<1> i) const { return v[i[0]]; }
};

// smooth background with a jump, so both smooth and discontinuous stencils are exercised
Field1D make_field()
{
    Field1D f;
    for (std::size_t i = 0; i < 40; ++i)
        f.v.push_back(std::sin(0.3 * i) + (i >= 20 ? 2.0 : 0.0));
    return f;
}

template<typename Rec>
struct SharedBetas : public ::testing::Test
{
};

using Recs = ::testing::Types<WENO3Reconstruction<Layout1D>, WENOZReconstruction<Layout1D>>;
TYPED_TEST_SUITE(SharedBetas, Recs);

} // namespace


// WENOZ: betas computed from the field itself must reproduce the per-variable reconstruction
// exactly, since the shared path reuses the same stencils and weights. (Shared WENO3 uses Z-type
// weights instead of the per-variable JS ones, so it has no such identity.)
TYPED_TEST(SharedBetas, ownBetasReproducePerVariableReconstruction)
{
    using Rec = TypeParam;
    if constexpr (Rec::stencil_size != 5)
        GTEST_SKIP() << "shared WENO3 weights differ from per-variable ones by design";

    auto const f       = make_field();
    auto constexpr dir = Direction::X;

    for (std::size_t i = Rec::nghosts + 1; i + Rec::nghosts < f.v.size(); ++i)
    {
        MeshIndex<1> const face{i};
        auto const betasL = Rec::compute_IS(
            Rec::template centered_stencil<dir>(f, Layout1D::previous<dir>(face)));
        auto const betasR = Rec::compute_IS(Rec::template centered_stencil<dir>(f, face));

        auto const [L, R]   = Rec::template reconstruct<dir>(f, face);
        auto const [Ls, Rs] = Rec::template reconstruct<dir>(f, face, betasL, betasR);
        EXPECT_EQ(L, Ls) << "face " << i;
        EXPECT_EQ(R, Rs) << "face " << i;

        auto const [cL, cR] = Rec::template center_reconstruct<dir, 0, 0>(f, face);
        auto const [cLs, cRs]
            = Rec::template center_reconstruct<dir, 0, 0>(f, face, betasL, betasR);
        EXPECT_EQ(cL, cLs) << "face " << i;
        EXPECT_EQ(cR, cRs) << "face " << i;
    }
}


// betas from another field steer the stencil choice: a jump seen only in the reference field
// must make a smooth field use the same one-sided substencil
TYPED_TEST(SharedBetas, foreignBetasSelectTheReferenceStencil)
{
    using Rec          = TypeParam;
    auto constexpr dir = Direction::X;

    Field1D ref, cubic;
    for (std::size_t i = 0; i < 20; ++i)
    {
        ref.v.push_back(i >= 10 ? 1.0 : 0.0);
        cubic.v.push_back(static_cast<double>(i * i * i));
    }
    auto const q = [&](std::size_t i) { return cubic.v[i]; };

    // the face between cells 9 and 10 sits on the reference jump: the left state must come from
    // the substencil ending at cell 9, the right state from the one starting at cell 10.
    // A cubic tells the substencils apart: none of them reproduces it exactly.
    MeshIndex<1> const face{10};
    auto const betasL
        = Rec::compute_IS(Rec::template centered_stencil<dir>(ref, Layout1D::previous<dir>(face)));
    auto const betasR = Rec::compute_IS(Rec::template centered_stencil<dir>(ref, face));

    auto const [L, R] = Rec::template reconstruct<dir>(cubic, face, betasL, betasR);

    if constexpr (Rec::stencil_size == 5)
    {
        EXPECT_NEAR(L, (1. / 3.) * q(7) - (7. / 6.) * q(8) + (11. / 6.) * q(9), 1e-9 * std::abs(L));
        EXPECT_NEAR(R, (1. / 3.) * q(12) - (7. / 6.) * q(11) + (11. / 6.) * q(10), 1e-9 * std::abs(R));
    }
    else
    {
        EXPECT_NEAR(L, -0.5 * q(8) + 1.5 * q(9), 1e-9 * std::abs(L));
        EXPECT_NEAR(R, -0.5 * q(11) + 1.5 * q(10), 1e-9 * std::abs(R));
    }

    // its own betas see no jump, so the per-variable path blends substencils and differs
    auto const [ownL, ownR] = Rec::template reconstruct<dir>(cubic, face);
    EXPECT_GT(std::abs(ownL - L), 1e-3);
}


// equal betas (e.g. all floored at round-off) must give the linear, optimal-order reconstruction
TYPED_TEST(SharedBetas, equalBetasGiveLinearWeights)
{
    using Rec          = TypeParam;
    auto constexpr dir = Direction::X;
    auto const f       = make_field();

    typename Rec::Betas equal;
    equal.fill(1.e-28);

    for (std::size_t i = Rec::nghosts + 1; i + Rec::nghosts < f.v.size(); ++i)
    {
        auto const q        = [&](int o) { return f.v[i + o]; };
        auto const [L, R]   = Rec::template reconstruct<dir>(f, MeshIndex<1>{i}, equal, equal);
        auto const [eL, eR] = [&]() {
            if constexpr (Rec::stencil_size == 5)
                return std::make_pair(
                    (2. * q(-3) - 13. * q(-2) + 47. * q(-1) + 27. * q(0) - 3. * q(1)) / 60.,
                    (2. * q(2) - 13. * q(1) + 47. * q(0) + 27. * q(-1) - 3. * q(-2)) / 60.);
            else
                return std::make_pair(-q(-2) / 6. + 5. * q(-1) / 6. + q(0) / 3.,
                                      -q(1) / 6. + 5. * q(0) / 6. + q(-1) / 3.);
        }();
        EXPECT_NEAR(L, eL, 1e-12) << "face " << i;
        EXPECT_NEAR(R, eR, 1e-12) << "face " << i;
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
