#ifndef PHARE_HYBRID_REFLUX_COMMS_HPP
#define PHARE_HYBRID_REFLUX_COMMS_HPP

#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/messenger_utils.hpp"

#include "SAMRAI/hier/CoarsenOperator.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/RefineOperator.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace PHARE::amr
{

template<typename HybridModel>
struct HybridRefluxComms
{
    using GridLayoutT   = typename HybridModel::gridlayout_type;
    using rm_t          = typename HybridModel::resources_manager_type;
    using RefOp_ptr     = std::shared_ptr<SAMRAI::hier::RefineOperator>;
    using CoarsenOp_ptr = std::shared_ptr<SAMRAI::hier::CoarsenOperator>;

    static constexpr std::size_t dimension       = GridLayoutT::dimension;
    static constexpr std::size_t rootLevelNumber = 0;

    using TFfillPattern = TensorFieldFillPattern<dimension>;

    RefluxChannel reflux_{static_cast<int>(dimension)};

    void registerQuantities(HybridMessengerInfo const& info,
                            rm_t& rm,
                            CoarsenOp_ptr electricFieldCoarseningOp,
                            RefOp_ptr EfieldRefineOp,
                            std::shared_ptr<TFfillPattern> nonOverwriteInteriorTFfillPattern)
    {
        auto e_fluxsum_id = rm.getID(info.fluxSumElectric);

        if (!e_fluxsum_id)
            throw std::runtime_error(
                "HybridRefluxComms: missing electric refluxing field variable IDs");

        // textbook refluxing: the fine fluxSum is coarsened onto the coarse fluxSum
        // (dst == src), and the coarse solver applies the correction itself via
        // reflux_geometry; ghosts are then refilled so they agree with refluxed cells.
        reflux_.coarsenAlgo.registerCoarsen(*e_fluxsum_id, *e_fluxsum_id,
                                            electricFieldCoarseningOp);
        reflux_.refineAlgo.registerRefine(*e_fluxsum_id, *e_fluxsum_id, *e_fluxsum_id,
                                          EfieldRefineOp, nonOverwriteInteriorTFfillPattern);
    }

    void registerLevel(int levelNumber,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy)
    {
        auto const level = hierarchy->getPatchLevel(levelNumber);
        reflux_.registerLevel(hierarchy, level, levelNumber, rootLevelNumber);
    }

    void reflux(int fineLevelNumber, int coarserLevelNumber, double syncTime)
    {
        reflux_.reflux(fineLevelNumber, coarserLevelNumber, syncTime);
    }
};

} // namespace PHARE::amr

#endif
