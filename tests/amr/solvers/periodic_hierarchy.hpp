#ifndef PHARE_TESTS_AMR_SOLVERS_PERIODIC_HIERARCHY_HPP
#define PHARE_TESTS_AMR_SOLVERS_PERIODIC_HIERARCHY_HPP

#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/BoxLevel.h>
#include <SAMRAI/hier/BoxLevelConnectorUtils.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/tbox/MemoryDatabase.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

#include <memory>
#include <vector>

// single-level, single-patch, fully periodic 3D hierarchy on [0,1]^3
inline std::shared_ptr<SAMRAI::hier::PatchHierarchy> makePeriodicHierarchy3D(int nCells)
{
    auto const dim = SAMRAI::tbox::Dimension{3};

    auto geomDB  = std::make_shared<SAMRAI::tbox::MemoryDatabase>("Hall3DGeomDB");
    int lower[3] = {0, 0, 0};
    int upper[3] = {nCells - 1, nCells - 1, nCells - 1};
    std::vector<SAMRAI::tbox::DatabaseBox> dbBoxes;
    dbBoxes.emplace_back(dim, lower, upper);
    geomDB->putDatabaseBoxVector("domain_boxes", dbBoxes);

    double x_lo[3] = {0.0, 0.0, 0.0};
    double x_up[3] = {1.0, 1.0, 1.0};
    geomDB->putDoubleArray("x_lo", x_lo, 3);
    geomDB->putDoubleArray("x_up", x_up, 3);

    int periodicity[3] = {1, 1, 1};
    geomDB->putIntegerArray("periodic_dimension", periodicity, 3);

    auto hierDB = std::make_shared<SAMRAI::tbox::MemoryDatabase>("Hall3DHierarchyDB");
    hierDB->putInteger("max_levels", 1);

    auto geometry
        = std::make_shared<SAMRAI::geom::CartesianGridGeometry>(dim, "Hall3DGeom", geomDB);
    auto hierarchy
        = std::make_shared<SAMRAI::hier::PatchHierarchy>("Hall3DHierarchy", geometry, hierDB);

    SAMRAI::hier::Box domain{dim};
    static int boxCounter = 0;
    auto const rank       = SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld().getRank();
    auto const gid        = SAMRAI::hier::GlobalId{SAMRAI::hier::LocalId{boxCounter++}, rank};
    domain.setBlockId(SAMRAI::hier::BlockId{0});
    domain.setId(SAMRAI::hier::BoxId{gid});
    domain.setLower(SAMRAI::hier::Index{std::vector<int>{0, 0, 0}});
    domain.setUpper(SAMRAI::hier::Index{std::vector<int>{nCells - 1, nCells - 1, nCells - 1}});

    SAMRAI::hier::BoxContainer levelBoxes;
    levelBoxes.push_back(domain);
    SAMRAI::hier::BoxLevel level0{levelBoxes, SAMRAI::hier::IntVector::getOne(dim), geometry};

    // A hand-built BoxLevel carries no periodic image boxes: "periodic_dimension" only sets up
    // the grid geometry's shift catalog, and the sole place SAMRAI populates the coarsest
    // level's images from it is mesh::GriddingAlgorithm::makeCoarsestLevel(), through
    // hier::BoxLevelConnectorUtils::addPeriodicImages() -- the path a real hierarchy takes and
    // this fixture bypasses. Without the images, every dst<->src Connector a RefineSchedule
    // builds on this level sees one isolated patch, so the entire ghost layer comes back
    // unfilled and RefineSchedule escalates it to a coarser level that does not exist.
    // Add them here, exactly as makeCoarsestLevel() does, so periodicity is real for
    // communication and not only for the geometry.
    SAMRAI::hier::BoxLevelConnectorUtils blcu;
    blcu.addPeriodicImages(
        level0, geometry->getDomainSearchTree(),
        SAMRAI::hier::IntVector::max(hierarchy->getRequiredConnectorWidth(0, 0, true),
                                     SAMRAI::hier::IntVector::getOne(dim)));

    hierarchy->makeNewPatchLevel(0, level0);

    return hierarchy;
}

#endif
