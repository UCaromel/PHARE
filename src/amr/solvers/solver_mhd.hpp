#ifndef PHARE_SOLVER_MHD_HPP
#define PHARE_SOLVER_MHD_HPP

#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/data/vecfield/vecfield.hpp"
#include "core/errors.hpp"
#include "core/numerics/finite_volume_euler/finite_volume_euler.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/utilities/index/index.hpp"
#include "initializer/data_provider.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/solvers/solver.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"

namespace PHARE::solver
{
template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy,
         typename Messenger    = amr::MHDMessenger<MHDModel>,
         typename ModelViews_t = MHDModelView<MHDModel>>
class SolverMHD : public ISolver<AMR_Types>
{
private:
    static constexpr auto dimension = MHDModel::dimension;

    using patch_t     = typename AMR_Types::patch_t;
    using level_t     = typename AMR_Types::level_t;
    using hierarchy_t = typename AMR_Types::hierarchy_t;

    using FieldT      = typename MHDModel::field_type;
    using VecFieldT   = typename MHDModel::vecfield_type;
    using GridLayout  = typename MHDModel::gridlayout_type;
    using MHDQuantity = core::MHDQuantity;

    using IPhysicalModel_t = IPhysicalModel<AMR_Types>;
    using IMessenger       = amr::IMessenger<IPhysicalModel_t>;

    core::AllFluxes<FieldT, VecFieldT> fluxes_;

    TimeIntegratorStrategy evolve_;

    // Refluxing
    core::AllFluxes<FieldT, VecFieldT> fluxSum_;
    VecFieldT fluxSumE_{this->name() + "_fluxSumE", MHDQuantity::Vector::E};

    std::unordered_map<std::size_t, double> oldTime_;

public:
    SolverMHD(PHARE::initializer::PHAREDict const& dict)
        : ISolver<AMR_Types>{"MHDSolver"}
        , fluxes_{{"rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                  {"rhoV_fx", MHDQuantity::Vector::VecFlux_x},
                  {"B_fx", MHDQuantity::Vector::VecFlux_x},
                  {"Etot_fx", MHDQuantity::Scalar::ScalarFlux_x},

                  {"rho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                  {"rhoV_fy", MHDQuantity::Vector::VecFlux_y},
                  {"B_fy", MHDQuantity::Vector::VecFlux_y},
                  {"Etot_fy", MHDQuantity::Scalar::ScalarFlux_y},

                  {"rho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                  {"rhoV_fz", MHDQuantity::Vector::VecFlux_z},
                  {"B_fz", MHDQuantity::Vector::VecFlux_z},
                  {"Etot_fz", MHDQuantity::Scalar::ScalarFlux_z}}
        , evolve_{dict}
        , fluxSum_{{"sumRho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                   {"sumRhoV_fx", MHDQuantity::Vector::VecFlux_x},
                   {"sumB_fx", MHDQuantity::Vector::VecFlux_x},
                   {"sumEtot_fx", MHDQuantity::Scalar::ScalarFlux_x},

                   {"sumRho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                   {"sumRhoV_fy", MHDQuantity::Vector::VecFlux_y},
                   {"sumB_fy", MHDQuantity::Vector::VecFlux_y},
                   {"sumEtot_fy", MHDQuantity::Scalar::ScalarFlux_y},

                   {"sumRho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                   {"sumRhoV_fz", MHDQuantity::Vector::VecFlux_z},
                   {"sumB_fz", MHDQuantity::Vector::VecFlux_z},
                   {"sumEtot_fz", MHDQuantity::Scalar::ScalarFlux_z}}
    {
    }

    virtual ~SolverMHD() = default;

    std::string modelName() const override { return MHDModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;

    void registerResources(IPhysicalModel<AMR_Types>& model) override;

    // TODO make this a resourcesUser
    void allocate(IPhysicalModel<AMR_Types>& model, patch_t& patch,
                  double const allocateTime) const override;

    void prepareStep(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level,
                     double const currentTime) override;

    void accumulateFluxSum(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level,
                           double const coef,
                           SAMRAI::hier::CoarseFineBoundary const& cfBoundary) override;

    void resetFluxSum(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level) override;

    void reflux(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level, IMessenger& messenger,
                double const time,
                std::vector<SAMRAI::hier::BoundaryBox> const& cfFaceBoundaries) override;

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
                      IMessenger& fromCoarserMessenger, double const currentTime,
                      double const newTime) override;

    void onRegrid() override {}

    std::shared_ptr<ISolverModelView> make_view(level_t& level, IPhysicalModel_t& model) override
    {
        return std::make_shared<ModelViews_t>(level, dynamic_cast<MHDModel&>(model));
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(fluxes_, fluxSum_, fluxSumE_, evolve_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(fluxes_, fluxSum_, fluxSumE_, evolve_);
    }

private:
    void mhdNaNCheck_(MHDModel& state, level_t const& level, double time);

    struct TimeSetter
    {
        template<typename QuantityAccessor>
        void operator()(QuantityAccessor accessor)
        {
            for (auto& state : views)
                views.model().resourcesManager->setTime(accessor(state), *state.patch, newTime);
        }

        ModelViews_t& views;
        double newTime;
    };
};

// -----------------------------------------------------------------------------

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::registerResources(IPhysicalModel_t& model)
{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->registerResources(fluxes_.rho_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.B_fx);
    mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(fluxes_.rho_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.B_fy);
        mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(fluxes_.rho_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.rhoV_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.B_fz);
            mhdmodel.resourcesManager->registerResources(fluxes_.Etot_fz);
        }
    }

    mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.B_fx);
    mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fx);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.B_fy);
        mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fy);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->registerResources(fluxSum_.rho_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.rhoV_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.B_fz);
            mhdmodel.resourcesManager->registerResources(fluxSum_.Etot_fz);
        }
    }
    mhdmodel.resourcesManager->registerResources(fluxSumE_);

    evolve_.registerResources(mhdmodel);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::allocate(
    IPhysicalModel_t& model, patch_t& patch, double const allocateTime) const

{
    auto& mhdmodel = dynamic_cast<MHDModel&>(model);

    mhdmodel.resourcesManager->allocate(fluxes_.rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxes_.Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(fluxes_.rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxes_.Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(fluxes_.rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxes_.Etot_fz, patch, allocateTime);
        }
    }

    mhdmodel.resourcesManager->allocate(fluxSum_.rho_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.B_fx, patch, allocateTime);
    mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fx, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        mhdmodel.resourcesManager->allocate(fluxSum_.rho_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.B_fy, patch, allocateTime);
        mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fy, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            mhdmodel.resourcesManager->allocate(fluxSum_.rho_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.rhoV_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.B_fz, patch, allocateTime);
            mhdmodel.resourcesManager->allocate(fluxSum_.Etot_fz, patch, allocateTime);
        }
    }
    mhdmodel.resourcesManager->allocate(fluxSumE_, patch, allocateTime);

    evolve_.allocate(mhdmodel, patch, allocateTime);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info)
    const

{
    auto& mhdInfo = dynamic_cast<amr::MHDMessengerInfo&>(*info);

    mhdInfo.ghostMagneticFluxesX.emplace_back(fluxes_.B_fx.name());

    if constexpr (dimension >= 2)
    {
        mhdInfo.ghostMagneticFluxesY.emplace_back(fluxes_.B_fy.name());

        if constexpr (dimension == 3)
        {
            mhdInfo.ghostMagneticFluxesZ.emplace_back(fluxes_.B_fz.name());
        }
    }

    evolve_.fillMessengerInfo(mhdInfo);

    auto&& [timeFluxes, timeElectric] = evolve_.exposeFluxes();

    mhdInfo.reflux          = core::AllFluxesNames{timeFluxes};
    mhdInfo.refluxElectric  = timeElectric.name();
    mhdInfo.fluxSum         = core::AllFluxesNames{fluxSum_};
    mhdInfo.fluxSumElectric = fluxSumE_.name();

    // for the faraday in reflux
    mhdInfo.ghostElectric.emplace_back(timeElectric.name());
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::prepareStep(
    IPhysicalModel_t&, SAMRAI::hier::PatchLevel& level, double const currentTime)
{
    oldTime_[level.getLevelNumber()] = currentTime;
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger,
               ModelViews_t>::accumulateFluxSum(IPhysicalModel_t& model,
                                                SAMRAI::hier::PatchLevel& level, double const coef,
                                                SAMRAI::hier::CoarseFineBoundary const& cfBoundary)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::accumulateFluxSum");

    bool const debugRefluxTrace = std::getenv("PHARE_REFLUX_DEBUG_TRACE") != nullptr;

    auto& mhdModel = dynamic_cast<MHDModel&>(model);

    for (auto& patch : level)
    {
        auto&& tf          = evolve_.exposeFluxes();
        auto& timeFluxes   = std::get<0>(tf);
        auto& timeElectric = std::get<1>(tf);

        auto const& layout = amr::layoutFromPatch<GridLayout>(*patch);
        auto _ = mhdModel.resourcesManager->setOnPatch(*patch, fluxSum_, fluxSumE_, timeFluxes,
                                                       timeElectric);

        std::array<std::size_t, 6> locationHits{};
        std::array<double, 6> sumAddedEx{};
        std::array<double, 6> sumAddedEz{};
        std::array<std::string, 6> firstKey{};
        std::array<std::string, 6> lastKey{};
        std::array<bool, 6> seenLocation{};
        std::array<core::Point<int, dimension>, 6> firstAmrIdx{};
        std::array<core::Point<int, dimension>, 6> lastAmrIdx{};

        auto const keyFor = [&](int location, auto const& idx) {
            if constexpr (dimension == 1)
                return std::to_string(location) + ":" + std::to_string(idx[0]);
            if constexpr (dimension == 2)
                return std::to_string(location) + ":" + std::to_string(idx[0]) + ":"
                       + std::to_string(idx[1]);
            return std::to_string(location) + ":" + std::to_string(idx[0]) + ":"
                   + std::to_string(idx[1]) + ":" + std::to_string(idx[2]);
        };

        auto const addScalar = [&](auto& left, auto const& right,
                                   core::Point<int, dimension> const& amrIdx) {
            auto const idx = layout.AMRToLocal(amrIdx);
            left(idx) += right(idx) * coef;
        };

        auto const addVector = [&](auto& left, auto const& right,
                                   core::Point<int, dimension> const& amrIdx) {
            auto const idx = layout.AMRToLocal(amrIdx);
            left(core::Component::X)(idx) += right(core::Component::X)(idx) * coef;
            left(core::Component::Y)(idx) += right(core::Component::Y)(idx) * coef;
            left(core::Component::Z)(idx) += right(core::Component::Z)(idx) * coef;
        };

        auto const& boundaries = cfBoundary.getBoundaries(patch->getGlobalId(), 1);

        if (debugRefluxTrace)
        {
            if constexpr (dimension >= 2)
            {
                auto const exPhysStartY
                    = layout.physicalStartIndex(timeElectric(core::Component::X), core::Direction::Y);
                auto const exPhysEndY
                    = layout.physicalEndIndex(timeElectric(core::Component::X), core::Direction::Y);
                std::cerr << "[PATCH-INFO] patch=" << patch->getGlobalId()
                          << " patchAMRBox=" << patch->getBox()
                          << " Ex_physY=[" << exPhysStartY << "," << exPhysEndY << "]"
                          << " numBdry=" << boundaries.size() << std::endl;
            }
            for (auto const& _bb : boundaries)
                std::cerr << "[BB-EXTENTS] patch=" << patch->getGlobalId()
                          << " loc=" << _bb.getLocationIndex()
                          << " box=" << _bb.getBox() << std::endl;
        }

        auto const& patchCellBox = patch->getBox();

        for (auto const& bb : boundaries)
        {
            auto const location = bb.getLocationIndex();
            bool const isLower  = (location % 2 == 0);
            int const normalDir = (location < 2) ? core::dirX : (location < 4 ? core::dirY : core::dirZ);

            for (auto const& amrIdx : amr::phare_box_from<dimension>(bb.getBox()))
            {
                // Skip cells outside the patch in transverse directions.
                // The CF boundary box extends one ghost cell beyond the patch in each transverse
                // direction; those ghost cells are NaN sentinels and must not be accumulated.
                bool inPatch = true;
                for (int d = 0; d < static_cast<int>(dimension); ++d)
                {
                    if (d == normalDir) continue;
                    if (amrIdx[d] < patchCellBox.lower(d) || amrIdx[d] > patchCellBox.upper(d))
                    {
                        inPatch = false;
                        break;
                    }
                }
                if (!inPatch) continue;

                // For lower boundaries, the SAMRAI CF boundary box cell index is 1 below the
                // physical CF interface face. Shift by +1 in the normal direction to land on
                // the correct flux face (physicalStartIndex of the face-centred quantity).
                auto readIdx = amrIdx;
                if (isLower) readIdx[normalDir] += 1;

                if (location == 0 || location == 1)
                {
                    addScalar(fluxSum_.rho_fx, timeFluxes.rho_fx, readIdx);
                    addVector(fluxSum_.rhoV_fx, timeFluxes.rhoV_fx, readIdx);
                    addVector(fluxSum_.B_fx, timeFluxes.B_fx, readIdx);
                    addScalar(fluxSum_.Etot_fx, timeFluxes.Etot_fx, readIdx);

                    addScalar(fluxSumE_(core::Component::Y), timeElectric(core::Component::Y),
                              readIdx);
                    addScalar(fluxSumE_(core::Component::Z), timeElectric(core::Component::Z),
                              readIdx);
                }
                else if (location == 2 || location == 3)
                {
                    auto const loc = location;
                    if (debugRefluxTrace && loc >= 0 && loc < static_cast<int>(locationHits.size()))
                    {
                        locationHits[loc]++;
                        auto const ridx = layout.AMRToLocal(readIdx);
                        sumAddedEx[loc] += timeElectric(core::Component::X)(ridx);
                        sumAddedEz[loc] += timeElectric(core::Component::Z)(ridx);
                        if (!seenLocation[loc])
                        {
                            firstKey[loc]    = keyFor(loc, amrIdx);
                            firstAmrIdx[loc] = amrIdx;
                            seenLocation[loc] = true;
                        }
                        lastKey[loc]    = keyFor(loc, amrIdx);
                        lastAmrIdx[loc] = amrIdx;
                    }

                    addScalar(fluxSum_.rho_fy, timeFluxes.rho_fy, readIdx);
                    addVector(fluxSum_.rhoV_fy, timeFluxes.rhoV_fy, readIdx);
                    addVector(fluxSum_.B_fy, timeFluxes.B_fy, readIdx);
                    addScalar(fluxSum_.Etot_fy, timeFluxes.Etot_fy, readIdx);

                    addScalar(fluxSumE_(core::Component::X), timeElectric(core::Component::X),
                              readIdx);
                    addScalar(fluxSumE_(core::Component::Z), timeElectric(core::Component::Z),
                              readIdx);

                    if (debugRefluxTrace && (amrIdx[0] == 50 || amrIdx[0] == 51))
                    {
                        auto const idx = layout.AMRToLocal(readIdx);
                        std::cerr << "[ACCUM-SAMPLE] patch=" << patch->getGlobalId()
                                  << " loc=" << location << " amr=" << amrIdx
                                  << " readAmr=" << readIdx << " idx=" << idx
                                  << " timeE=("
                                  << timeElectric(core::Component::X)(idx) << ","
                                  << timeElectric(core::Component::Y)(idx) << ","
                                  << timeElectric(core::Component::Z)(idx) << ")"
                                  << " fluxSumE=(" << fluxSumE_(core::Component::X)(idx) << ","
                                  << fluxSumE_(core::Component::Y)(idx) << ","
                                  << fluxSumE_(core::Component::Z)(idx) << ")"
                                  << std::endl;
                    }
                }
                else if constexpr (dimension == 3)
                {
                    addScalar(fluxSum_.rho_fz, timeFluxes.rho_fz, readIdx);
                    addVector(fluxSum_.rhoV_fz, timeFluxes.rhoV_fz, readIdx);
                    addVector(fluxSum_.B_fz, timeFluxes.B_fz, readIdx);
                    addScalar(fluxSum_.Etot_fz, timeFluxes.Etot_fz, readIdx);

                    addScalar(fluxSumE_(core::Component::X), timeElectric(core::Component::X),
                              readIdx);
                    addScalar(fluxSumE_(core::Component::Y), timeElectric(core::Component::Y),
                              readIdx);
                }
            }
        }

        if (debugRefluxTrace)
        {
            std::cerr << "[ACCUM-DBG] patch=" << patch->getGlobalId() << " loc2_hits="
                      << locationHits[2] << " loc2_sumEx=" << sumAddedEx[2]
                      << " loc2_sumEz=" << sumAddedEz[2] << " loc2_first=" << firstKey[2]
                      << " loc2_firstIdx=" << layout.AMRToLocal(firstAmrIdx[2])
                      << " loc2_last=" << lastKey[2]
                      << " loc2_lastIdx=" << layout.AMRToLocal(lastAmrIdx[2])
                      << " loc3_hits=" << locationHits[3]
                      << " loc3_sumEx=" << sumAddedEx[3] << " loc3_sumEz=" << sumAddedEz[3]
                      << " loc3_first=" << firstKey[3]
                      << " loc3_firstIdx=" << layout.AMRToLocal(firstAmrIdx[3])
                      << " loc3_last=" << lastKey[3]
                      << " loc3_lastIdx=" << layout.AMRToLocal(lastAmrIdx[3])
                      << std::endl;
        }
    }
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::resetFluxSum(
    IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level)
{
    auto& mhdModel = dynamic_cast<MHDModel&>(model);

    for (auto& patch : level)
    {
        auto const& layout = amr::layoutFromPatch<GridLayout>(*patch);
        auto _             = mhdModel.resourcesManager->setOnPatch(*patch, fluxSum_, fluxSumE_);

        evalFluxesOnGhostBox(
            layout, [&](auto& left, auto const&... args) mutable { left(args...) = 0.0; },
            fluxSum_);

        layout.evalOnGhostBox(fluxSumE_(core::Component::X), [&](auto const&... args) mutable {
            fluxSumE_(core::Component::X)(args...) = 0.0;
        });

        layout.evalOnGhostBox(fluxSumE_(core::Component::Y), [&](auto const&... args) mutable {
            fluxSumE_(core::Component::Y)(args...) = 0.0;
        });

        layout.evalOnGhostBox(fluxSumE_(core::Component::Z), [&](auto const&... args) mutable {
            fluxSumE_(core::Component::Z)(args...) = 0.0;
        });
    }
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::reflux(
    IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level, IMessenger& messenger,
    double const time, std::vector<SAMRAI::hier::BoundaryBox> const& cfFaceBoundaries)
{
    auto& bc           = dynamic_cast<Messenger&>(messenger);
    auto& mhdModel     = dynamic_cast<MHDModel&>(model);
    auto&& tf          = evolve_.exposeFluxes();
    auto& timeFluxes   = std::get<0>(tf);
    auto& timeElectric = std::get<1>(tf);
    auto& state        = mhdModel.state;
    double const dt    = time - oldTime_[level.getLevelNumber()];
    constexpr auto dirX = core::dirX;
    constexpr auto dirY = core::dirY;
    constexpr auto dirZ = core::dirZ;

    bool const debugReflux      = std::getenv("PHARE_REFLUX_DEBUG") != nullptr;
    bool const debugRefluxTrace = std::getenv("PHARE_REFLUX_DEBUG_TRACE") != nullptr;
    std::array<std::size_t, 6> hitPerLocation{};
    std::unordered_map<std::string, std::size_t> hitPerCoarseLocation;
    std::array<double, 6> sumDBxPerLocation{};
    std::array<double, 6> sumDByPerLocation{};
    std::array<double, 6> sumDBzPerLocation{};
    std::array<double, 6> maxAbsDEPerLocation{};
    std::size_t maxDuplicateHits = 0;
    std::size_t skippedDuplicates = 0;
    double maxAbsDE              = 0.0;
    double maxAbsDeltaB          = 0.0;

    auto const keyFor = [&](int location, auto const& idx) {
        if constexpr (dimension == 1)
            return std::to_string(location) + ":" + std::to_string(idx[0]);
        if constexpr (dimension == 2)
            return std::to_string(location) + ":" + std::to_string(idx[0]) + ":"
                   + std::to_string(idx[1]);
        return std::to_string(location) + ":" + std::to_string(idx[0]) + ":"
               + std::to_string(idx[1]) + ":" + std::to_string(idx[2]);
    };

    for (auto& patch : level)
    {
        std::unordered_set<std::string> seenCoarseLocation;

        auto const& patchAMRBox = patch->getBox();
        auto const& layout      = amr::layoutFromPatch<GridLayout>(*patch);
        auto _                  = mhdModel.resourcesManager->setOnPatch(
            *patch, state.rho, state.rhoV, state.Etot, state.B, fluxSum_, fluxSumE_,
            timeFluxes, timeElectric);

        for (auto const& bb : cfFaceBoundaries)
        {
            auto const location = bb.getLocationIndex();
            // SAMRAI convention assumed here: lower side = even, upper side = odd.
            // If convention differs, hydro and magnetic reflux sign/order are wrong.
            int const sign      = (location % 2 == 0) ? 1 : -1;
            bool const isLower  = (location % 2 == 0);

            auto const dir = [&]() {
                if (location == 0 || location == 1)
                    return dirX;
                if (location == 2 || location == 3)
                    return dirY;
                return dirZ;
            }();

            auto const scale = sign * dt / layout.meshSize()[dir];

            for (auto const& fineIdx : amr::phare_box_from<dimension>(bb.getBox()))
            {
                auto const coarseIdx = amr::toCoarseIndex(fineIdx);
                auto const inPatch = [&]() {
                    if (coarseIdx[dirX] < patchAMRBox.lower(dirX)
                        || coarseIdx[dirX] > patchAMRBox.upper(dirX))
                        return false;
                    if constexpr (dimension > 1)
                        if (coarseIdx[dirY] < patchAMRBox.lower(dirY)
                            || coarseIdx[dirY] > patchAMRBox.upper(dirY))
                            return false;
                    if constexpr (dimension > 2)
                        if (coarseIdx[dirZ] < patchAMRBox.lower(dirZ)
                            || coarseIdx[dirZ] > patchAMRBox.upper(dirZ))
                            return false;
                    return true;
                }();
                if (!inPatch)
                    continue;

                auto const key = keyFor(location, coarseIdx);
                if (!seenCoarseLocation.insert(key).second)
                {
                    if (debugReflux)
                        skippedDuplicates++;
                    continue;
                }

                auto const idx = layout.AMRToLocal(coarseIdx);
                // The CF boundary box is in cell coordinates; for lower boundaries the flux face
                // is one step into the interior (coarseIdx+1 in the normal direction).
                // Accumulation wrote at that shifted position; read from the same face here.
                auto fluxCoarseIdx = coarseIdx;
                if (isLower) fluxCoarseIdx[dir] += 1;
                auto const idxFlux = layout.AMRToLocal(fluxCoarseIdx);

                if (debugReflux)
                {
                    if (location >= 0 && location < static_cast<int>(hitPerLocation.size()))
                        hitPerLocation[location]++;
                    auto const n = ++hitPerCoarseLocation[key];
                    maxDuplicateHits = std::max(maxDuplicateHits, n);
                }

                if (dir == dirX)
                {
                    state.rho(idx) += scale * (fluxSum_.rho_fx(idxFlux) - timeFluxes.rho_fx(idxFlux));
                    state.rhoV(core::Component::X)(idx)
                        += scale
                           * (fluxSum_.rhoV_fx(core::Component::X)(idxFlux)
                              - timeFluxes.rhoV_fx(core::Component::X)(idxFlux));
                    state.rhoV(core::Component::Y)(idx)
                        += scale
                           * (fluxSum_.rhoV_fx(core::Component::Y)(idxFlux)
                              - timeFluxes.rhoV_fx(core::Component::Y)(idxFlux));
                    state.rhoV(core::Component::Z)(idx)
                        += scale
                           * (fluxSum_.rhoV_fx(core::Component::Z)(idxFlux)
                              - timeFluxes.rhoV_fx(core::Component::Z)(idxFlux));
                    state.Etot(idx)
                        += scale * (fluxSum_.Etot_fx(idxFlux) - timeFluxes.Etot_fx(idxFlux));

                    auto const dEx = 0.0;
                    auto const dEy
                        = fluxSumE_(core::Component::Y)(idxFlux) - timeElectric(core::Component::Y)(idxFlux);
                    auto const dEz
                        = fluxSumE_(core::Component::Z)(idxFlux) - timeElectric(core::Component::Z)(idxFlux);
                    auto const magSign = -sign;
                    auto const dBy = magSign * (-dt / layout.meshSize()[dirX] * dEz);
                    auto const dBz = magSign * (+dt / layout.meshSize()[dirX] * dEy);
                    state.B(core::Component::Y)(idx) += dBy;
                    state.B(core::Component::Z)(idx) += dBz;

                    if (debugReflux)
                    {
                        auto const absDE = std::max(std::abs(dEy), std::abs(dEz));
                        maxAbsDE         = std::max(maxAbsDE, absDE);
                        maxAbsDeltaB = std::max(maxAbsDeltaB, std::max(std::abs(dBy), std::abs(dBz)));
                        if (location >= 0 && location < 6)
                        {
                            sumDByPerLocation[location] += dBy;
                            sumDBzPerLocation[location] += dBz;
                            maxAbsDEPerLocation[location]
                                = std::max(maxAbsDEPerLocation[location], absDE);
                        }

                        if (debugRefluxTrace)
                        {
                            std::cerr << "[REFLUX-DBG] level=" << level.getLevelNumber()
                                      << " patch=" << patch->getGlobalId() << " location="
                                      << location << " side=" << ((location % 2 == 0) ? "lower" : "upper")
                                      << " dir=x coarseKey=" << keyFor(location, coarseIdx)
                                      << " dE=(" << dEx << "," << dEy << "," << dEz << ")"
                                      << " dB=(0," << dBy << "," << dBz << ")"
                                      << " Eavg=(0," << fluxSumE_(core::Component::Y)(idxFlux) << ","
                                      << fluxSumE_(core::Component::Z)(idxFlux) << ")"
                                      << " timeE=(0," << timeElectric(core::Component::Y)(idxFlux) << ","
                                      << timeElectric(core::Component::Z)(idxFlux) << ")"
                                      << std::endl;
                        }
                    }
                }
                else if (dir == dirY)
                {
                    state.rho(idx) += scale * (fluxSum_.rho_fy(idxFlux) - timeFluxes.rho_fy(idxFlux));
                    state.rhoV(core::Component::X)(idx)
                        += scale
                           * (fluxSum_.rhoV_fy(core::Component::X)(idxFlux)
                              - timeFluxes.rhoV_fy(core::Component::X)(idxFlux));
                    state.rhoV(core::Component::Y)(idx)
                        += scale
                           * (fluxSum_.rhoV_fy(core::Component::Y)(idxFlux)
                              - timeFluxes.rhoV_fy(core::Component::Y)(idxFlux));
                    state.rhoV(core::Component::Z)(idx)
                        += scale
                           * (fluxSum_.rhoV_fy(core::Component::Z)(idxFlux)
                              - timeFluxes.rhoV_fy(core::Component::Z)(idxFlux));
                    state.Etot(idx)
                        += scale * (fluxSum_.Etot_fy(idxFlux) - timeFluxes.Etot_fy(idxFlux));

                    auto const dEx
                        = fluxSumE_(core::Component::X)(idxFlux) - timeElectric(core::Component::X)(idxFlux);
                    auto const dEy = 0.0;
                    auto const dEz
                        = fluxSumE_(core::Component::Z)(idxFlux) - timeElectric(core::Component::Z)(idxFlux);
                    auto const magSign = -sign;
                    auto const dBx = magSign * (+dt / layout.meshSize()[dirY] * dEz);
                    auto const dBz = magSign * (-dt / layout.meshSize()[dirY] * dEx);
                    state.B(core::Component::X)(idx) += dBx;
                    state.B(core::Component::Z)(idx) += dBz;

                    if (debugReflux)
                    {
                        auto const absDE = std::max(std::abs(dEx), std::abs(dEz));
                        maxAbsDE         = std::max(maxAbsDE, absDE);
                        maxAbsDeltaB = std::max(maxAbsDeltaB, std::max(std::abs(dBx), std::abs(dBz)));
                        if (location >= 0 && location < 6)
                        {
                            sumDBxPerLocation[location] += dBx;
                            sumDBzPerLocation[location] += dBz;
                            maxAbsDEPerLocation[location]
                                = std::max(maxAbsDEPerLocation[location], absDE);
                        }

                        if (debugRefluxTrace)
                        {
                            std::cerr << "[REFLUX-DBG] level=" << level.getLevelNumber()
                                      << " patch=" << patch->getGlobalId() << " location="
                                      << location << " side=" << ((location % 2 == 0) ? "lower" : "upper")
                                      << " dir=y coarseKey=" << keyFor(location, coarseIdx)
                                      << " dE=(" << dEx << "," << dEy << "," << dEz << ")"
                                      << " dB=(" << dBx << ",0," << dBz << ")"
                                      << " Eavg=(" << fluxSumE_(core::Component::X)(idxFlux) << ",0,"
                                      << fluxSumE_(core::Component::Z)(idxFlux) << ")"
                                      << " timeE=(" << timeElectric(core::Component::X)(idxFlux) << ",0,"
                                      << timeElectric(core::Component::Z)(idxFlux) << ")"
                                      << std::endl;

                            if (coarseIdx[0] == 25 || coarseIdx[0] == 26)
                                std::cerr << "[REFLUX-SAMPLE-Y] patch=" << patch->getGlobalId()
                                          << " loc=" << location
                                          << " fineIdx=" << fineIdx
                                          << " coarseIdx=" << coarseIdx
                                          << " fluxCoarseIdx=" << fluxCoarseIdx
                                          << " idxFlux=" << idxFlux
                                          << " fluxSumEx=" << fluxSumE_(core::Component::X)(idxFlux)
                                          << " timeEx=" << timeElectric(core::Component::X)(idxFlux)
                                          << " dEx=" << dEx
                                          << " fluxSumEz=" << fluxSumE_(core::Component::Z)(idxFlux)
                                          << " timeEz=" << timeElectric(core::Component::Z)(idxFlux)
                                          << " dEz=" << dEz
                                          << std::endl;
                        }
                    }
                }
                else if constexpr (dimension == 3)
                {
                    state.rho(idx) += scale * (fluxSum_.rho_fz(idxFlux) - timeFluxes.rho_fz(idxFlux));
                    state.rhoV(core::Component::X)(idx)
                        += scale
                           * (fluxSum_.rhoV_fz(core::Component::X)(idxFlux)
                              - timeFluxes.rhoV_fz(core::Component::X)(idxFlux));
                    state.rhoV(core::Component::Y)(idx)
                        += scale
                           * (fluxSum_.rhoV_fz(core::Component::Y)(idxFlux)
                              - timeFluxes.rhoV_fz(core::Component::Y)(idxFlux));
                    state.rhoV(core::Component::Z)(idx)
                        += scale
                           * (fluxSum_.rhoV_fz(core::Component::Z)(idxFlux)
                              - timeFluxes.rhoV_fz(core::Component::Z)(idxFlux));
                    state.Etot(idx)
                        += scale * (fluxSum_.Etot_fz(idxFlux) - timeFluxes.Etot_fz(idxFlux));

                    auto const dEx
                        = fluxSumE_(core::Component::X)(idxFlux) - timeElectric(core::Component::X)(idxFlux);
                    auto const dEy
                        = fluxSumE_(core::Component::Y)(idxFlux) - timeElectric(core::Component::Y)(idxFlux);
                    auto const dEz = 0.0;
                    auto const magSign = -sign;
                    auto const dBx = magSign * (-dt / layout.meshSize()[dirZ] * dEy);
                    auto const dBy = magSign * (+dt / layout.meshSize()[dirZ] * dEx);
                    state.B(core::Component::X)(idx) += dBx;
                    state.B(core::Component::Y)(idx) += dBy;

                    if (debugReflux)
                    {
                        auto const absDE = std::max(std::abs(dEx), std::abs(dEy));
                        maxAbsDE         = std::max(maxAbsDE, absDE);
                        maxAbsDeltaB = std::max(maxAbsDeltaB, std::max(std::abs(dBx), std::abs(dBy)));
                        if (location >= 0 && location < 6)
                        {
                            sumDBxPerLocation[location] += dBx;
                            sumDByPerLocation[location] += dBy;
                            maxAbsDEPerLocation[location]
                                = std::max(maxAbsDEPerLocation[location], absDE);
                        }

                        if (debugRefluxTrace)
                        {
                            std::cerr << "[REFLUX-DBG] level=" << level.getLevelNumber()
                                      << " patch=" << patch->getGlobalId() << " location="
                                      << location << " side=" << ((location % 2 == 0) ? "lower" : "upper")
                                      << " dir=z coarseKey=" << keyFor(location, coarseIdx)
                                      << " dE=(" << dEx << "," << dEy << "," << dEz << ")"
                                      << " dB=(" << dBx << "," << dBy << ",0)"
                                      << " Eavg=(" << fluxSumE_(core::Component::X)(idxFlux) << ","
                                      << fluxSumE_(core::Component::Y)(idxFlux) << ",0)"
                                      << " timeE=(" << timeElectric(core::Component::X)(idxFlux) << ","
                                      << timeElectric(core::Component::Y)(idxFlux) << ",0)"
                                      << std::endl;
                        }
                    }
                }
            }
        }
    }

    if (debugReflux)
    {
        std::cerr << "[REFLUX-DBG] level=" << level.getLevelNumber() << " hits(loc0..5)="
                  << hitPerLocation[0] << "," << hitPerLocation[1] << "," << hitPerLocation[2]
                  << "," << hitPerLocation[3] << "," << hitPerLocation[4] << ","
                  << hitPerLocation[5] << " unique=" << hitPerCoarseLocation.size()
                  << " maxDup=" << maxDuplicateHits << " skippedDup=" << skippedDuplicates
                  << " max|dE|=" << maxAbsDE << " max|dBcorr|=" << maxAbsDeltaB
                  << " sumdBx(loc0..5)=" << sumDBxPerLocation[0] << "," << sumDBxPerLocation[1]
                  << "," << sumDBxPerLocation[2] << "," << sumDBxPerLocation[3] << ","
                  << sumDBxPerLocation[4] << "," << sumDBxPerLocation[5]
                  << " sumdBy(loc0..5)=" << sumDByPerLocation[0] << "," << sumDByPerLocation[1]
                  << "," << sumDByPerLocation[2] << "," << sumDByPerLocation[3] << ","
                  << sumDByPerLocation[4] << "," << sumDByPerLocation[5]
                  << " sumdBz(loc0..5)=" << sumDBzPerLocation[0] << "," << sumDBzPerLocation[1]
                  << "," << sumDBzPerLocation[2] << "," << sumDBzPerLocation[3] << ","
                  << sumDBzPerLocation[4] << "," << sumDBzPerLocation[5]
                  << " max|dE|perloc=" << maxAbsDEPerLocation[0] << "," << maxAbsDEPerLocation[1]
                  << "," << maxAbsDEPerLocation[2] << "," << maxAbsDEPerLocation[3] << ","
                  << maxAbsDEPerLocation[4] << "," << maxAbsDEPerLocation[5] << std::endl;
    }

    bc.fillMomentsGhosts(state, level, time);
    bc.fillMagneticGhosts(state.B, level, time);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::advanceLevel(
    hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& view,
    IMessenger& fromCoarserMessenger, double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::advanceLevel");

    auto& modelView   = dynamic_cast<ModelViews_t&>(view);
    auto& fromCoarser = dynamic_cast<Messenger&>(fromCoarserMessenger);
    auto level        = hierarchy.getPatchLevel(levelNumber);

    try
    {
        evolve_(modelView.model(), modelView.model().state, fluxes_, fromCoarser, *level,
                currentTime, newTime);

        mhdNaNCheck_(modelView.model(), *level, currentTime);
    }
    catch (core::DictionaryException& ex)
    {
        PHARE_LOG_ERROR(ex());
    }

    if (core::mpi::any(core::Errors::instance().any()))
        throw core::DictionaryException{}("ID", "SolverMHD::advanceLevel");
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy, typename Messenger,
         typename ModelViews_t>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy, Messenger, ModelViews_t>::mhdNaNCheck_(
    MHDModel& model, level_t const& level, double time)
{
    auto& rm  = model.resourcesManager;
    auto& rho = model.state.rho;

    auto check_nans = [&](auto const& field, auto const& origin,
                          core::MeshIndex<MHDModel::dimension> const& index) {
        if (std::isnan(field(index)))
        {
            std::stringstream ss;
            ss << "NaN detected in MHD field at index " << index << " on patch of origin " << origin
               << " on level " << level.getLevelNumber() << " at time " << time;
            core::DictionaryException ex{"cause", ss.str()};
            throw ex;
        }
    };

    for (auto const& patch : rm->enumerate(level, rho))
    {
        auto layout = amr::layoutFromPatch<GridLayout>(*patch);
        layout.evalOnGhostBox(
            rho, [&](auto const&... args) { check_nans(rho, layout.origin(), {args...}); });
    }
}

} // namespace PHARE::solver

#endif
