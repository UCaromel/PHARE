// Functional unit test for amr::MomentumTensorBorderFiller (phase 10).
//
// The filler must produce, on a single level, an ion momentum tensor whose values at
// patch/sibling-patch seam nodes are *border-complete* — the additive PlusEquals seam-sum
// of both patches' partial particle deposits. The defining property is PARTITION INVARIANCE:
// depositing the same global particle set on a 1-patch level vs. a 2-patch level (split at a
// seam) and border-completing the latter must yield identical M at every shared physical node.
// Particle iCell is stored in AMR index space (interpolator.hpp:396 AMRToLocal), so a particle
// placed in a given AMR cell deposits onto the same AMR node regardless of which patch's layout
// processes it — the identity is exact up to floating-point summation order.
//
//   A1 partitionInvarianceAcrossSeam : split-fill == single-fill on all shared nodes
//   A2 singlePatchFillIsDepositOnly  : single-patch fill == raw deposit (border-fill is a no-op)

#include "core/def/phare_mpi.hpp"

#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_impl.hpp"
#include "core/data/ions/ions.hpp"
#include "core/data/ions/ion_population/ion_population.hpp"
#include "core/data/particles/particle.hpp"
#include "core/data/particles/particle_array.hpp"
#include "core/data/tensorfield/tensorfield.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/interpolator/interpolator.hpp"
#include "core/utilities/point/point.hpp"

#include "initializer/data_provider.hpp"

#include "amr/resources_manager/resources_manager.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/momentum_tensor_border_filler.hpp"

#include "input_config.h"
#include "tests/initializer/init_functions.hpp"

#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/tbox/InputManager.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

#include "gtest/gtest.h"

#include <array>
#include <map>
#include <string>
#include <vector>

using namespace PHARE;
using namespace PHARE::core;
using namespace PHARE::amr;

static constexpr std::size_t dim         = 1;
static constexpr std::size_t interpOrder = 1;

using GridYee1D        = GridLayout<GridLayoutImplYee<dim, interpOrder>>;
using Field1D          = Field<dim, PhysicalQuantity::Scalar>;
using Grid1D           = Grid<NdArrayVector<dim>, PhysicalQuantity::Scalar>;
using VecField1D       = VecField<Field1D, PhysicalQuantity>;
using SymTensorField1D = SymTensorField<Field1D, PhysicalQuantity>;
using IonPopulation1D  = IonPopulation<ParticleArray<dim>, VecField1D, SymTensorField1D>;
using Ions1D           = Ions<IonPopulation1D, GridYee1D>;
using ResourcesManager1D = ResourcesManager<GridYee1D, Grid1D>;
using Filler1D = MomentumTensorBorderFiller<ResourcesManager1D, GridYee1D, Ions1D, dim, interpOrder>;
using InitFunctionT = PHARE::initializer::InitFunction<dim>;

// ----------------------------------------------------------------------------------------------
// fixed, deterministic particle set. cells 31/32 straddle the split seam (AMR primal node 32);
// every component v_i*v_j is exercised with distinct nonzero velocities. cells kept away from the
// periodic domain boundary (0 / 64) so the only relevant seam is the internal split at node 32.
struct PInfo
{
    int cell;
    double delta;
    double vx, vy, vz;
};

static std::vector<PInfo> const PARTS = {
    {5, 0.25, 1.0, 0.5, -0.3},   {15, 0.75, -0.7, 0.2, 0.9}, {30, 0.10, 0.4, -0.6, 0.8},
    {31, 0.80, 1.2, 0.3, -0.5},  {31, 0.20, -0.9, 1.1, 0.4}, {32, 0.30, 0.6, -0.2, 1.0},
    {32, 0.90, -0.4, 0.7, -0.8}, {40, 0.50, 0.3, 0.3, 0.3},  {55, 0.65, -1.0, 0.5, 0.2},
};

static PHARE::initializer::PHAREDict makeIonsDict()
{
    // init functions are required by the IonPopulation ctor but never invoked here
    // (particles are pushed manually, loadParticles is not called).
    using namespace PHARE::initializer::test_fn::func_1d;

    PHARE::initializer::PHAREDict d;
    d["ions"]["nbrPopulations"]    = std::size_t{1};
    d["ions"]["pop0"]["name"]      = std::string{"protons"};
    d["ions"]["pop0"]["mass"]      = 1.;
    auto& pi                       = d["ions"]["pop0"]["particle_initializer"];
    pi["name"]                     = std::string{"MaxwellianParticleInitializer"};
    pi["density"]                  = static_cast<InitFunctionT>(density);
    pi["bulk_velocity_x"]          = static_cast<InitFunctionT>(vx);
    pi["bulk_velocity_y"]          = static_cast<InitFunctionT>(vy);
    pi["bulk_velocity_z"]          = static_cast<InitFunctionT>(vz);
    pi["thermal_velocity_x"]       = static_cast<InitFunctionT>(vthx);
    pi["thermal_velocity_y"]       = static_cast<InitFunctionT>(vthy);
    pi["thermal_velocity_z"]       = static_cast<InitFunctionT>(vthz);
    pi["nbrPartPerCell"]           = int{0};
    pi["charge"]                   = 1.;
    pi["basis"]                    = std::string{"Cartesian"};
    return d;
}

using AmrBox = std::pair<int, int>; // inclusive cell lower / upper

static std::shared_ptr<SAMRAI::hier::PatchHierarchy> makeHierarchy(std::vector<AmrBox> const& boxes)
{
    static int nameCounter = 0;
    int const tag          = nameCounter++;

    auto inputDb = SAMRAI::tbox::InputManager::getManager()->parseInputFile(
        inputBase + std::string("input/input_db_1d"));

    SAMRAI::tbox::Dimension dimension{static_cast<unsigned short>(dim)};

    auto gridGeometry = std::make_shared<SAMRAI::geom::CartesianGridGeometry>(
        dimension, "cartesian_" + std::to_string(tag),
        inputDb->getDatabase("CartesianGridGeometry"));

    auto hierarchy = std::make_shared<SAMRAI::hier::PatchHierarchy>(
        "PatchHierarchy_" + std::to_string(tag), gridGeometry,
        inputDb->getDatabase("PatchHierarchy"));

    int const ownerRank = SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld().getRank();
    static int counterId = 0;

    SAMRAI::hier::BoxContainer container;
    for (auto const& b : boxes)
    {
        SAMRAI::hier::Box box{dimension};
        box.setBlockId(SAMRAI::hier::BlockId{0});
        box.setId(SAMRAI::hier::BoxId{
            SAMRAI::hier::GlobalId{SAMRAI::hier::LocalId{counterId++}, ownerRank}});
        box.setLower(SAMRAI::hier::Index(std::vector<int>{b.first}));
        box.setUpper(SAMRAI::hier::Index(std::vector<int>{b.second}));
        container.push_back(box);
    }

    SAMRAI::hier::BoxLevel level0{container, SAMRAI::hier::IntVector::getOne(dimension),
                                  gridGeometry};
    hierarchy->makeNewPatchLevel(0, level0);
    return hierarchy;
}

// Build a level from `boxes`, deposit PARTS into pop0, produce the ions momentum tensor either
// via the filler (border-complete) or via a raw per-patch deposit (no border-fill), and return
// the 6 tensor components keyed by AMR primal node index.
static std::map<int, std::array<double, 6>> computeM(std::vector<AmrBox> const& boxes,
                                                     bool useFiller)
{
    ResourcesManager1D rm;
    auto dict = makeIonsDict();
    Ions1D ions{dict["ions"]};
    Filler1D filler;

    rm.registerResources(ions);
    if (useFiller)
        rm.registerResources(filler.stagingField());

    auto hierarchy = makeHierarchy(boxes);
    auto level     = hierarchy->getPatchLevel(0);

    for (auto& patch : *level)
    {
        rm.allocate(ions, *patch, 0.);
        if (useFiller)
            rm.allocate(filler.stagingField(), *patch, 0.);
    }

    // push the deterministic particle set into pop0 (per patch, by AMR cell ownership)
    for (auto& patch : *level)
    {
        auto onPatch  = rm.setOnPatch(*patch, ions);
        int const lo  = patch->getBox().lower()(0);
        int const hi  = patch->getBox().upper()(0);
        for (auto& pop : ions)
        {
            auto& domain = pop.domainParticles();
            for (auto const& p : PARTS)
                if (p.cell >= lo && p.cell <= hi)
                    domain.push_back(Particle<dim>{0.1, 1.0, {{p.cell}}, {{p.delta}},
                                                   {{p.vx, p.vy, p.vz}}});
            break; // pop0 only
        }
    }

    if (useFiller)
    {
        filler.declareAlgos(ions, rm);
        filler.fillCompleteMomentumTensor(ions, level, rm, 0., /*levelGhostTimeCoef*/ 1.);
    }
    else
    {
        MomentumTensorInterpolator<dim, interpOrder> interp;
        for (auto& patch : *level)
        {
            auto onPatch = rm.setOnPatch(*patch, ions);
            auto layout  = layoutFromPatch<GridYee1D>(*patch);
            for (auto& pop : ions)
            {
                auto& m = pop.momentumTensor();
                m.zero();
                interp(pop.domainParticles(), m, layout, pop.mass());
            }
            ions.computeFullMomentumTensor();
        }
    }

    std::map<int, std::array<double, 6>> out;
    for (auto& patch : *level)
    {
        auto onPatch = rm.setOnPatch(*patch, ions);
        auto layout  = layoutFromPatch<GridYee1D>(*patch);
        auto& M      = ions.momentumTensor();
        auto const& [xx, xy, xz, yy, yz, zz] = M();
        int const lo = patch->getBox().lower()(0);
        int const hi = patch->getBox().upper()(0);
        for (int n = lo; n <= hi + 1; ++n) // primal nodes span [cellLo, cellHi+1]
        {
            auto const lcl = layout.AMRToLocal(Point<int, dim>{n});
            out[n]         = {xx(lcl), xy(lcl), xz(lcl), yy(lcl), yz(lcl), zz(lcl)};
        }
    }
    return out;
}

TEST(MomentumTensorBorderFiller, partitionInvarianceAcrossSeam)
{
    auto const single = computeM({{0, 64}}, /*useFiller=*/true);
    auto const split  = computeM({{0, 31}, {32, 64}}, /*useFiller=*/true);

    // every node owned by the split config must match the single-patch (un-partitioned) truth,
    // including the shared seam node 32 (border-completed on both split patches).
    std::size_t comparedSeam = 0;
    for (auto const& [node, vals] : split)
    {
        ASSERT_TRUE(single.count(node)) << "node " << node << " absent from single-patch ref";
        for (std::size_t c = 0; c < 6; ++c)
            EXPECT_NEAR(vals[c], single.at(node)[c], 1e-11)
                << "node " << node << " component " << c;
        if (node == 32)
            ++comparedSeam;
    }
    EXPECT_GT(comparedSeam, 0u) << "seam node 32 never compared — geometry assumption wrong";
}

TEST(MomentumTensorBorderFiller, singlePatchFillIsDepositOnly)
{
    auto const filled = computeM({{0, 64}}, /*useFiller=*/true);
    auto const raw    = computeM({{0, 64}}, /*useFiller=*/false);

    ASSERT_EQ(filled.size(), raw.size());
    for (auto const& [node, vals] : filled)
        for (std::size_t c = 0; c < 6; ++c)
            EXPECT_DOUBLE_EQ(vals[c], raw.at(node)[c]) << "node " << node << " component " << c;
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
