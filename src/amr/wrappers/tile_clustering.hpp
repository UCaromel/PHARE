#ifndef PHARE_AMR_WRAPPERS_TILE_CLUSTERING_HPP
#define PHARE_AMR_WRAPPERS_TILE_CLUSTERING_HPP

#include <SAMRAI/hier/BoxLevel.h>
#include <SAMRAI/hier/Connector.h>
#include <SAMRAI/hier/IntVector.h>
#include <SAMRAI/hier/PatchLevel.h>
#include <SAMRAI/mesh/BoxGeneratorStrategy.h>
#include <SAMRAI/mesh/TileClustering.h>
#include <SAMRAI/tbox/Dimension.h>
#include <SAMRAI/tbox/MemoryDatabase.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace PHARE::amr
{
/** SAMRAI's TileClustering, leaving the tagged level's connectors as it found them.
 *
 * When a tile holds tags owned by several processes, TileClustering::detectSemilocalEdges takes
 * the tagged level's cached self connector of width tileSize - 1, copies it, and removes the
 * periodic relationships of the copy and of its transpose. The transpose of a cached self
 * connector is that connector itself, and a copy shares it: the cached connector loses its
 * periodic neighbours for the rest of the run.
 *
 * Refine schedules built afterwards look for coarse data through that connector, or through
 * narrower ones extracted from it. A fine patch whose ghosts cross a periodic boundary into a
 * region that is not refined then has no source, and SAMRAI aborts in
 * RefineSchedule::finishScheduleConstruction ("No coarser levels...will not fill from coarser").
 *
 * This puts the connector back as it was before clustering.
 */
class TileClustering : public SAMRAI::mesh::BoxGeneratorStrategy
{
public:
    explicit TileClustering(SAMRAI::tbox::Dimension const& dim)
        : selfConnectorWidth_{dim, tileSize - 1}
        , tiles_{dim, inputDatabase_(dim)}
    {
    }

    void findBoxesContainingTags(std::shared_ptr<SAMRAI::hier::BoxLevel>& newBoxLevel,
                                 std::shared_ptr<SAMRAI::hier::Connector>& tagToNew,
                                 std::shared_ptr<SAMRAI::hier::PatchLevel> const& tagLevel,
                                 int const tagDataIndex, int const tagValue,
                                 SAMRAI::hier::BoxContainer const& boundBoxes,
                                 SAMRAI::hier::IntVector const& minBox,
                                 SAMRAI::hier::IntVector const& tagToNewWidth) override
    {
        auto const& tagBoxLevel = *tagLevel->getBoxLevel();

        // the connector TileClustering asks for: same level, same width, exact width
        auto& cached = const_cast<SAMRAI::hier::Connector&>(
            tagBoxLevel.findConnector(tagBoxLevel, selfConnectorWidth_,
                                      SAMRAI::hier::CONNECTOR_IMPLICIT_CREATION_RULE, true));
        SAMRAI::hier::Connector const beforeClustering{cached};

        tiles_.findBoxesContainingTags(newBoxLevel, tagToNew, tagLevel, tagDataIndex, tagValue,
                                       boundBoxes, minBox, tagToNewWidth);

        cached = beforeClustering;
    }

    void setMinimumCellRequest(std::size_t const numCells) override
    {
        tiles_.setMinimumCellRequest(numCells);
    }

    void setRatioToNewLevel(SAMRAI::hier::IntVector const& ratio) override
    {
        tiles_.setRatioToNewLevel(ratio);
    }

private:
    // SAMRAI's default, given explicitly since the width of the connector to protect follows it
    static constexpr int tileSize = 8;

    static std::shared_ptr<SAMRAI::tbox::Database>
    inputDatabase_(SAMRAI::tbox::Dimension const& dim)
    {
        auto db = std::make_shared<SAMRAI::tbox::MemoryDatabase>("TileClustering");
        db->putIntegerVector("tile_size", std::vector<int>(dim.getValue(), tileSize));
        return db;
    }

    SAMRAI::hier::IntVector selfConnectorWidth_;
    SAMRAI::mesh::TileClustering tiles_;
};

} // namespace PHARE::amr

#endif
