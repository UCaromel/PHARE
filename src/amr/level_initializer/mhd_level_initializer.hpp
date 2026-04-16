#ifndef PHARE_AMR_MHD_LEVEL_INITIALIZER_HPP
#define PHARE_AMR_MHD_LEVEL_INITIALIZER_HPP

#include "amr/level_initializer/level_initializer.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "core/utilities/index/index.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"


namespace PHARE::solver
{
template<typename MHDModel>
class MHDLevelInitializer : public LevelInitializer<typename MHDModel::amr_types>
{
    using amr_types                    = typename MHDModel::amr_types;
    using hierarchy_t                  = typename amr_types::hierarchy_t;
    using level_t                      = typename amr_types::level_t;
    using patch_t                      = typename amr_types::patch_t;
    using IPhysicalModelT              = IPhysicalModel<amr_types>;
    using IMessengerT                  = amr::IMessenger<IPhysicalModelT>;
    using MHDMessenger                 = amr::MHDMessenger<MHDModel>;
    using GridLayoutT                  = MHDModel::gridlayout_type;
    static constexpr auto dimension    = GridLayoutT::dimension;
    static constexpr auto interp_order = GridLayoutT::interp_order;

    inline bool isRootLevel(int levelNumber) const { return levelNumber == 0; }

public:
    MHDLevelInitializer() = default;

    void initialize(std::shared_ptr<hierarchy_t> const& hierarchy, int levelNumber,
                    std::shared_ptr<level_t> const& oldLevel, IPhysicalModelT& model,
                    amr::IMessenger<IPhysicalModelT>& messenger, double initDataTime,
                    bool isRegridding) override
    {
        auto& mhdModel = static_cast<MHDModel&>(model);
        auto& level    = amr_types::getLevel(*hierarchy, levelNumber);

        if (isRegridding)
        {
            PHARE_LOG_LINE_STR("regriding level " + std::to_string(levelNumber));
            PHARE_LOG_START(3, "mhdLevelInitializer::initialize : regriding block");
            messenger.regrid(hierarchy, levelNumber, oldLevel, model, initDataTime);
            PHARE_LOG_STOP(3, "mhdLevelInitializer::initialize : regriding block");
        }
        else
        {
            if (isRootLevel(levelNumber))
            {
                PHARE_LOG_START(3, "mhdLevelInitializer::initialize : root level init");
                model.initialize(level);
                messenger.fillRootGhosts(model, level, initDataTime);
                PHARE_LOG_STOP(3, "mhdLevelInitializer::initialize : root level init");
            }
            else
            {
                PHARE_LOG_START(3, "mhdLevelInitializer::initialize : initlevel");
                messenger.initLevel(model, level, initDataTime);
                PHARE_LOG_STOP(3, "mhdLevelInitializer::initialize : initlevel");
            }
        }

        if (mhdModel.hasInnerBoundary())
        {
            for (auto& patch : level)
            {
                auto layout = amr::layoutFromPatch<GridLayoutT>(*patch);
                auto _
                    = mhdModel.resourcesManager->setOnPatch(*patch, *mhdModel.innerBoundaryManager);
                mhdModel.innerBoundaryManager->classify(layout);
            }

            // Set inactive/ghost cells to a safe physical state so the Riemann solver
            // never receives pathological input (negative or zero rho/P) from them.
            for (auto& patch : level)
            {
                auto layout = amr::layoutFromPatch<GridLayoutT>(*patch);
                auto _guard = mhdModel.resourcesManager->setOnPatch(
                    *patch, *mhdModel.innerBoundaryManager, mhdModel.state);

                auto& meshData   = mhdModel.innerBoundaryManager->getMeshData();
                auto& cellStatus = meshData.cellStatusField();

                layout.evalOnBox(mhdModel.state.rho, [&](auto&... args) {
                    auto idx = core::MeshIndex<dimension>{args...};
                    if (cellStatus(idx) > core::toDouble(core::ElemStatus::Cut))
                    {
                        constexpr double safeRho = 1.0;
                        constexpr double safeP   = 1.0;

                        mhdModel.state.rho(idx) = safeRho;
                        mhdModel.state.P(idx)   = safeP;

                        mhdModel.state.V(core::Component::X)(idx) = 0.0;
                        mhdModel.state.V(core::Component::Y)(idx) = 0.0;
                        mhdModel.state.V(core::Component::Z)(idx) = 0.0;

                        mhdModel.state.rhoV(core::Component::X)(idx) = 0.0;
                        mhdModel.state.rhoV(core::Component::Y)(idx) = 0.0;
                        mhdModel.state.rhoV(core::Component::Z)(idx) = 0.0;

                        // Keep B as-is (avoids introducing div(B) ≠ 0); project to cell centre
                        // for Etot computation only.
                        auto const bx = GridLayoutT::project(mhdModel.state.B(core::Component::X),
                                                             idx, GridLayoutT::faceXToCellCenter());
                        auto const by = GridLayoutT::project(mhdModel.state.B(core::Component::Y),
                                                             idx, GridLayoutT::faceYToCellCenter());
                        auto const bz = GridLayoutT::project(mhdModel.state.B(core::Component::Z),
                                                             idx, GridLayoutT::faceZToCellCenter());

                        mhdModel.thermo->setState_DP(safeRho, safeP);

                        auto const e_int         = safeRho * mhdModel.thermo->internalEnergy();
                        mhdModel.state.Etot(idx) = core::totalEnergyFromInternalEnergy(
                            e_int, safeRho, 0., 0., 0., bx, by, bz);
                    }
                });
            }
        }
    }
};

} // namespace PHARE::solver


#endif
