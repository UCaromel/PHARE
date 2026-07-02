// Unit test for amr::CrossModelFillContext (cross-coupling-boundary recursive ghost fill).
//
//   - PairKind classification for every level pair of a 2mhd+3hyb stack (and the 1mhd case)
//   - presence hook application gating: MHD levels / firstHybridLevel / hybrid levels above
//   - EM mirror gating: fires on MHD levels only, no-op when unset
//   - nested spawn fragment registry: nullptr when unset, per-population lookup
//   - crossing-prim registrar: no-op when unset, forwarded when set

#include "core/def/phare_mpi.hpp"

#include "amr/messengers/cross_model_fill_context.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/BlockId.h>
#include <SAMRAI/hier/Index.h>
#include <SAMRAI/hier/IntVector.h>
#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/hier/PatchDescriptor.h>
#include <SAMRAI/tbox/Dimension.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/xfer/RefineAlgorithm.h>

#include "gtest/gtest.h"

#include <memory>

using namespace PHARE::amr;


namespace
{
SAMRAI::hier::Patch makePatch(SAMRAI::tbox::Dimension const& dim)
{
    SAMRAI::hier::Box box{SAMRAI::hier::Index{dim, 0}, SAMRAI::hier::Index{dim, 10},
                          SAMRAI::hier::BlockId{0}};
    return SAMRAI::hier::Patch{box, std::make_shared<SAMRAI::hier::PatchDescriptor>()};
}


struct StubRefinePatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch&, double,
                                       SAMRAI::hier::IntVector const&) override
    {
    }
    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector::getZero(dim);
    }
    void preprocessRefine(SAMRAI::hier::Patch&, SAMRAI::hier::Patch const&,
                          SAMRAI::hier::Box const&, SAMRAI::hier::IntVector const&) override
    {
    }
    void postprocessRefine(SAMRAI::hier::Patch&, SAMRAI::hier::Patch const&,
                           SAMRAI::hier::Box const&, SAMRAI::hier::IntVector const&) override
    {
    }
};
} // namespace



TEST(CrossModelFillContext, classifiesLevelsAndPairsOfA2mhd3hybStack)
{
    // levels 0,1 = MHD ; levels 2,3,4 = hybrid
    CrossModelFillContext ctx;
    EXPECT_FALSE(ctx.hasFirstHybridLevel());
    ctx.setFirstHybridLevel(2);
    ASSERT_TRUE(ctx.hasFirstHybridLevel());

    EXPECT_EQ(ModelKind::MHD, ctx.kindOf(0));
    EXPECT_EQ(ModelKind::MHD, ctx.kindOf(1));
    EXPECT_EQ(ModelKind::Hybrid, ctx.kindOf(2));
    EXPECT_EQ(ModelKind::Hybrid, ctx.kindOf(3));
    EXPECT_EQ(ModelKind::Hybrid, ctx.kindOf(4));

    EXPECT_EQ(PairKind::MHD_MHD, ctx.pairKind(0, 1));
    EXPECT_EQ(PairKind::MHD_Hyb, ctx.pairKind(1, 2));
    EXPECT_EQ(PairKind::Hyb_Hyb, ctx.pairKind(2, 3));
    EXPECT_EQ(PairKind::Hyb_Hyb, ctx.pairKind(3, 4));

    // non-adjacent pairs (recursion may skip levels via grandparent chains)
    EXPECT_EQ(PairKind::MHD_MHD, ctx.pairKind(0, 1));
    EXPECT_EQ(PairKind::MHD_Hyb, ctx.pairKind(0, 2));
    EXPECT_EQ(PairKind::MHD_Hyb, ctx.pairKind(1, 4));
    EXPECT_EQ(PairKind::Hyb_Hyb, ctx.pairKind(2, 4));
}


TEST(CrossModelFillContext, classifiesA1mhd2hybStack)
{
    CrossModelFillContext ctx;
    ctx.setFirstHybridLevel(1);

    EXPECT_EQ(ModelKind::MHD, ctx.kindOf(0));
    EXPECT_EQ(PairKind::MHD_Hyb, ctx.pairKind(0, 1));
    EXPECT_EQ(PairKind::Hyb_Hyb, ctx.pairKind(1, 2));
}


TEST(CrossModelFillContext, appliesPresenceHooksPerModelSideOfTheBoundary)
{
    SAMRAI::tbox::Dimension const dim{1};
    auto patch = makePatch(dim);

    CrossModelFillContext ctx;
    ctx.setFirstHybridLevel(2); // 2mhd + hybrid above

    int mhdCalls    = 0;
    int hybridCalls = 0;
    double lastTime = -1.;
    ctx.addMHDLevelPresence([&](SAMRAI::hier::Patch&, double t) {
        ++mhdCalls;
        lastTime = t;
    });
    ctx.addMHDLevelPresence([&](SAMRAI::hier::Patch&, double) { ++mhdCalls; });
    ctx.addHybridLevelPresence([&](SAMRAI::hier::Patch&, double t) {
        ++hybridCalls;
        lastTime = t;
    });

    ctx.applyPresence(patch, 0, 0.5); // MHD level: both MHD hooks fire
    EXPECT_EQ(2, mhdCalls);
    EXPECT_EQ(0, hybridCalls);
    EXPECT_DOUBLE_EQ(0.5, lastTime);

    ctx.applyPresence(patch, 2, 1.0); // firstHybridLevel: nothing fires
    EXPECT_EQ(2, mhdCalls);
    EXPECT_EQ(0, hybridCalls);

    ctx.applyPresence(patch, 3, 2.0); // hybrid above the boundary: hybrid hooks fire
    EXPECT_EQ(2, mhdCalls);
    EXPECT_EQ(1, hybridCalls);
    EXPECT_DOUBLE_EQ(2.0, lastTime);
}


TEST(CrossModelFillContext, electromagMirrorFiresOnMHDLevelsOnly)
{
    SAMRAI::tbox::Dimension const dim{1};
    auto patch = makePatch(dim);

    CrossModelFillContext ctx;
    ctx.setFirstHybridLevel(1);

    ctx.applyMHDElectromagMirror(patch, 0, 0.); // unset: no-op, no crash

    int mirrorCalls = 0;
    ctx.setMHDElectromagMirror([&](SAMRAI::hier::Patch&, double) { ++mirrorCalls; });

    ctx.applyMHDElectromagMirror(patch, 1, 0.); // hybrid level: gated out
    ctx.applyMHDElectromagMirror(patch, 2, 0.);
    EXPECT_EQ(0, mirrorCalls);

    ctx.applyMHDElectromagMirror(patch, 0, 0.);
    EXPECT_EQ(1, mirrorCalls);
}


TEST(CrossModelFillContext, nestedSpawnFragmentRegistryIsPerPopulation)
{
    CrossModelFillContext ctx;

    EXPECT_EQ(nullptr, ctx.nestedSpawnFragment("protons"));

    auto protons = std::make_shared<StubRefinePatchStrategy>();
    auto alphas  = std::make_shared<StubRefinePatchStrategy>();
    ctx.setNestedSpawnFragment("protons", protons);
    ctx.setNestedSpawnFragment("alphas", alphas);

    EXPECT_EQ(protons.get(), ctx.nestedSpawnFragment("protons"));
    EXPECT_EQ(alphas.get(), ctx.nestedSpawnFragment("alphas"));
    EXPECT_EQ(nullptr, ctx.nestedSpawnFragment("electrons"));
}


TEST(CrossModelFillContext, crossingPrimRegistrarNoOpWhenUnsetForwardedWhenSet)
{
    CrossModelFillContext ctx;
    SAMRAI::xfer::RefineAlgorithm algorithm;

    ctx.applyCrossingPrimItems(algorithm); // unset: no-op, no crash

    int registrarCalls = 0;
    ctx.setCrossingPrimRegistrar([&](SAMRAI::xfer::RefineAlgorithm&) { ++registrarCalls; });
    ctx.applyCrossingPrimItems(algorithm);
    EXPECT_EQ(1, registrarCalls);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();

    int const testResult = RUN_ALL_TESTS();

    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();

    return testResult;
}
