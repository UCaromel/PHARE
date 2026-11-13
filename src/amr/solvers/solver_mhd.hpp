#ifndef PHARE_SOLVER_MHD_HPP
#define PHARE_SOLVER_MHD_HPP


#include "amr/messengers/messenger.hpp"
#include "core/numerics/primite_conservative_converter/face_energy.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/solvers/solver.hpp"
#include "amr/solvers/time_integrator/euler_using_computed_flux.hpp"

#include "core/errors.hpp"
#include "core/logger.hpp"
#include "core/models/quantities/mhd_quantities.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/riemann_solvers/mhd_speeds.hpp"
#include "core/utilities/index/index.hpp"

#include "initializer/data_provider.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <cmath>
#include <tuple>
#include <vector>

namespace PHARE::solver
{
template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
class SolverMHD : public ISolver<AMR_Types>
{
private:
    static constexpr auto dimension = MHDModel::dimension;

    using patch_t     = AMR_Types::patch_t;
    using level_t     = AMR_Types::level_t;
    using hierarchy_t = AMR_Types::hierarchy_t;

    using FieldT      = MHDModel::field_type;
    using VecFieldT   = MHDModel::vecfield_type;
    using MHDStateT   = MHDModel::state_type;
    using GridLayout  = MHDModel::gridlayout_type;
    using MHDQuantity = core::MHDQuantity;
    using Messenger   = TimeIntegratorStrategy::Messenger;

    using IPhysicalModel_t = IPhysicalModel<AMR_Types>;
    using IMessenger       = amr::IMessenger<IPhysicalModel_t>;

    core::AllFluxes<FieldT, VecFieldT> fluxes_;

    TimeIntegratorStrategy evolve_;

    // Refluxing
    MHDStateT stateOld_{this->name() + "_stateOld"};

    core::AllFluxes<FieldT, VecFieldT> fluxSum_;
    VecFieldT fluxSumE_{this->name() + "_fluxSumE", MHDQuantity::Vector::E};
    EulerUsingComputedFlux<MHDModel> reflux_euler_;

    std::unordered_map<std::size_t, double> oldTime_;
    // experiment PHARE_REFLUX_DIAG: per level, coarse time fluxes (Etot_fx, Etot_fy, rho_fx, rho_fy per
    // patch) and state (Etot, rho per patch) at the end of advanceLevel
    std::unordered_map<int, std::vector<std::array<std::vector<double>, 6>>> dgSnap_;
    static bool refluxDiagOn_() { return std::getenv("PHARE_REFLUX_DIAG") != nullptr; }

    double const gamma_; // adiabatic index
    double const eta_;   // resistivity
    bool const isHall_;

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
        , gamma_{dict["to_primitive"]["heat_capacity_ratio"].template to<double>()}
        , eta_{dict["constrained_transport"]["resistivity"].template to<double>()}
        , isHall_{cppdict::get_value(dict, "fv_method/hall", false)}
    {
    }

    virtual ~SolverMHD() = default;

    std::string modelName() const override { return MHDModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;

    void registerResources(IPhysicalModel<AMR_Types>& model) override;

    // TODO make this a resourcesUser
    void allocate(IPhysicalModel<AMR_Types>& model, patch_t& patch,
                  double const allocateTime) const override;

    void prepareStep(IPhysicalModel_t& model, level_t& level, double const currentTime) override;

    void accumulateFluxSum(IPhysicalModel_t& model, level_t& level, double const coef) override;

    void resetFluxSum(IPhysicalModel_t& model, level_t& level) override;

    void reflux(IPhysicalModel_t& model, level_t& level, IMessenger& messenger,
                double const time) override;

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, IPhysicalModel_t& model,
                      IMessenger& fromCoarserMessenger, double const currentTime,
                      double const newTime) override;

    double computeStableDt(IPhysicalModel_t& model, level_t& level,
                           CFLNumbers const& cflNumbers) override;

    void onRegrid() override {}


    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(fluxes_, fluxSum_, fluxSumE_, stateOld_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(fluxes_, fluxSum_, fluxSumE_, stateOld_);
    }

private:
    void mhdNaNCheck_(MHDModel& state, level_t const& level, double time);
};

// -----------------------------------------------------------------------------

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::registerResources(
    IPhysicalModel_t& model)
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

    mhdmodel.resourcesManager->registerResources(stateOld_);

    evolve_.registerResources(mhdmodel);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::allocate(
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

    mhdmodel.resourcesManager->allocate(stateOld_, patch, allocateTime);

    evolve_.allocate(mhdmodel, patch, allocateTime);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::fillMessengerInfo(
    std::unique_ptr<amr::IMessengerInfo> const& info) const

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
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::prepareStep(IPhysicalModel_t& model,
                                                                         level_t& level,
                                                                         double const currentTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::prepareStep");

    oldTime_[level.getLevelNumber()] = currentTime;

    auto& mhdModel = dynamic_cast<MHDModel&>(model);

    auto& rho  = mhdModel.state.rho;
    auto& rhoV = mhdModel.state.rhoV;
    auto& B    = mhdModel.state.B;
    auto& Etot = mhdModel.state.Etot;

    for (auto& patch : level)
    {
        auto dataOnPatch
            = mhdModel.resourcesManager->setOnPatch(*patch, rho, rhoV, B, Etot, stateOld_);

        mhdModel.resourcesManager->setTime(stateOld_.rho, *patch, currentTime);
        mhdModel.resourcesManager->setTime(stateOld_.rhoV, *patch, currentTime);
        mhdModel.resourcesManager->setTime(stateOld_.B, *patch, currentTime);
        mhdModel.resourcesManager->setTime(stateOld_.Etot, *patch, currentTime);

        stateOld_.rho.copyData(rho);
        stateOld_.rhoV.copyData(rhoV);
        stateOld_.B.copyData(B);
        stateOld_.Etot.copyData(Etot);
    }
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::accumulateFluxSum(
    IPhysicalModel_t& model, level_t& level, double const coef)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::accumulateFluxSum");

    auto& mhdModel = dynamic_cast<MHDModel&>(model);
    auto& rm       = *mhdModel.resourcesManager;

    // MacOS clang still unhappy with structured bindings captures in lambdas
    auto&& tf          = evolve_.exposeFluxes();
    auto& timeFluxes   = std::get<0>(tf);
    auto& timeElectric = std::get<1>(tf);

    for (auto& _ : rm.enumerate(level, fluxSum_, fluxSumE_, timeFluxes, timeElectric))
    {
        core::operate<core::PlusEqualsProduct>(fluxSum_, timeFluxes, coef);
        core::operate<core::PlusEqualsProduct>(fluxSumE_, timeElectric, coef);
    }
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::resetFluxSum(IPhysicalModel_t& model,
                                                                          level_t& level)
{
    auto& mhdModel = dynamic_cast<MHDModel&>(model);
    auto& rm       = *mhdModel.resourcesManager;

    for (auto& _ : rm.enumerate(level, fluxSum_, fluxSumE_))
    {
        fluxSum_.zero();
        fluxSumE_.zero();
    }
}


template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::reflux(IPhysicalModel_t& model,
                                                                    level_t& level,
                                                                    IMessenger& messenger,
                                                                    double const time)
{
    auto& bc                          = dynamic_cast<Messenger&>(messenger);
    auto& mhdModel                    = dynamic_cast<MHDModel&>(model);
    auto&& [timeFluxes, timeElectric] = evolve_.exposeFluxes();

    // experiment PHARE_REFLUX_DIAG=x0,x1,y0,y1 (2D; physical box of the next finer level). Diagnostic only.
    static auto const refluxBox = [] {
        std::array<double, 4> b{0, 0, 0, 0};
        auto const* s = std::getenv("PHARE_REFLUX_DIAG");
        if (!s || std::sscanf(s, "%lf,%lf,%lf,%lf", &b[0], &b[1], &b[2], &b[3]) != 4)
            return std::optional<std::array<double, 4>>{};
        return std::optional<std::array<double, 4>>{b};
    }();
    // face measure: time-averaged coarse fluxes after the messenger's reflux overwrite minus the ones
    // the coarse step used. Faces on the box boundary: the reflux defect |dF|/h (a cell rate); faces
    // elsewhere outside the box must not change. Also the state change since the end of the step.
    if constexpr (dimension == 2)
        if (refluxBox && dgSnap_.count(level.getLevelNumber()))
        {
            auto const& bx   = *refluxBox;
            auto const& snap = dgSnap_[level.getLevelNumber()];
            double ifE = 0, ifEsum = 0, ifR = 0, farF = 0, dSt = 0;
            std::size_t nif = 0, ip = 0;
            for (auto& patch :
                 mhdModel.resourcesManager->enumerate(level, mhdModel.state, timeFluxes))
            {
                auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                double const hx   = 1. / layout.inverseMeshSize(core::Direction::X);
                double const hy   = 1. / layout.inverseMeshSize(core::Direction::Y);
                int const lo[2]   = {static_cast<int>(std::lround(bx[0] / hx)),
                                     static_cast<int>(std::lround(bx[2] / hy))};
                int const hi[2]   = {static_cast<int>(std::lround(bx[1] / hx)) - 1,
                                     static_cast<int>(std::lround(bx[3] / hy)) - 1};
                auto const& a     = snap[ip++];
                for (int d = 0; d < 2; ++d)
                {
                    auto const& FE = d == 0 ? timeFluxes.Etot_fx : timeFluxes.Etot_fy;
                    auto const& FR = d == 0 ? timeFluxes.rho_fx : timeFluxes.rho_fy;
                    double const h = d == 0 ? hx : hy;
                    int const t    = 1 - d;
                    std::size_t k  = 0;
                    layout.evalOnBox(FE, [&](auto... i) {
                        auto const g = layout.localToAMR(
                            core::Point<int, dimension>{static_cast<int>(i)...});
                        double const dE = std::abs(FE(i...) - a[d][k]);
                        double const dR = std::abs(FR(i...) - a[2 + d][k]);
                        ++k;
                        bool const tin = g[t] >= lo[t] && g[t] <= hi[t];
                        bool const onb = tin && (g[d] == lo[d] || g[d] == hi[d] + 1);
                        bool const inn = tin && g[d] > lo[d] && g[d] <= hi[d];
                        if (onb)
                        {
                            ifE = std::max(ifE, dE / h);
                            ifR = std::max(ifR, dR / h);
                            ifEsum += dE / h;
                            ++nif;
                        }
                        else if (!inn)
                            farF = std::max(farF, std::max(dE, dR));
                    });
                }
                std::size_t k = 0;
                layout.evalOnBox(mhdModel.state.rho, [&](auto... i) {
                    dSt = std::max({dSt, std::abs(mhdModel.state.Etot(i...) - a[4][k]),
                                    std::abs(mhdModel.state.rho(i...) - a[5][k])});
                    ++k;
                });
            }
            auto const* rk = std::getenv("OMPI_COMM_WORLD_RANK");
            std::fprintf(stderr,
                         "REFLUXFACE r=%s lvl=%d t=%.9e ifE_max=%.6e ifE_sum=%.6e nif=%zu "
                         "ifRho_max=%.6e farF_max=%.6e stateChange_max=%.6e\n",
                         rk ? rk : "0", level.getLevelNumber(), time, ifE, ifEsum, nif, ifR, farF,
                         dSt);
        }

    double const dt = time - oldTime_[level.getLevelNumber()];
    reflux_euler_(mhdModel, stateOld_, mhdModel.state, timeElectric, timeFluxes, bc, level, time,
                  dt);
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::advanceLevel(
    hierarchy_t const& hierarchy, int const levelNumber, IPhysicalModel_t& model,
    IMessenger& fromCoarserMessenger, double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::advanceLevel");

    auto& mhdModel    = dynamic_cast<MHDModel&>(model);
    auto& fromCoarser = dynamic_cast<Messenger&>(fromCoarserMessenger);
    auto level        = hierarchy.getPatchLevel(levelNumber);

    try
    {
        evolve_(mhdModel, mhdModel.state, fluxes_, fromCoarser, *level, currentTime, newTime);

        if constexpr (dimension == 2)
            if (refluxDiagOn_())
            {
                auto&& [tF, tE] = evolve_.exposeFluxes();
                auto& snap      = dgSnap_[levelNumber];
                snap.clear();
                for (auto& patch : mhdModel.resourcesManager->enumerate(*level, mhdModel.state, tF))
                {
                    auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
                    auto& a           = snap.emplace_back();
                    layout.evalOnBox(tF.Etot_fx, [&](auto... i) {
                        a[0].push_back(tF.Etot_fx(i...));
                        a[2].push_back(tF.rho_fx(i...));
                    });
                    layout.evalOnBox(tF.Etot_fy, [&](auto... i) {
                        a[1].push_back(tF.Etot_fy(i...));
                        a[3].push_back(tF.rho_fy(i...));
                    });
                    layout.evalOnBox(mhdModel.state.rho, [&](auto... i) {
                        a[4].push_back(mhdModel.state.Etot(i...));
                        a[5].push_back(mhdModel.state.rho(i...));
                    });
                }
            }

        PHARE_DEBUG_DO({ mhdNaNCheck_(mhdModel, *level, currentTime); })
    }
    catch (core::DictionaryException& ex)
    {
        PHARE_LOG_ERROR(ex());
    }

    if (mpi::any_errors())
        throw core::DictionaryException{}("ID", "SolverMHD::advanceLevel");
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
double SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::computeStableDt(
    IPhysicalModel_t& model, level_t& level, CFLNumbers const& cflNumbers)
{
    PHARE_LOG_SCOPE(1, "SolverMHD::computeStableDt");

    auto const [wave, diffusive] = cflNumbers;

    auto& mhdModel = dynamic_cast<MHDModel&>(model);
    auto& rho      = mhdModel.state.rho;
    auto& rhoV     = mhdModel.state.rhoV;
    auto& B        = mhdModel.state.B;
    auto& Etot     = mhdModel.state.Etot;

    double dt = std::numeric_limits<double>::max();

    for (auto& patch : level)
    {
        auto const& layout = amr::layoutFromPatch<GridLayout>(*patch);
        auto _             = mhdModel.resourcesManager->setOnPatch(*patch, rho, rhoV, B, Etot);

        auto const meshSize = layout.meshSize();

        // diffusive cfl constraint
        if (eta_ > 0)
        {
            double invdx2 = 0;
            for (std::size_t d = 0; d < dimension; ++d)
                invdx2 += 1.0 / (meshSize[d] * meshSize[d]);
            dt = std::min(dt, diffusive / (2.0 * eta_ * invdx2));
        }

        auto const& rhoVx = rhoV(core::Component::X);
        auto const& rhoVy = rhoV(core::Component::Y);
        auto const& rhoVz = rhoV(core::Component::Z);
        auto const& Bx    = B(core::Component::X);
        auto const& By    = B(core::Component::Y);
        auto const& Bz    = B(core::Component::Z);

        // wave cfl constraint
        layout.evalOnBox(rho, [&](auto&... args) mutable {
            core::MeshIndex<dimension> const index{args...};

            auto const r  = rho(index);
            auto const vx = rhoVx(index) / r;
            auto const vy = rhoVy(index) / r;
            auto const vz = rhoVz(index) / r;

            auto const bx
                = GridLayout::template project<GridLayout::implT::faceXToCellCenter>(Bx, index);
            auto const by
                = GridLayout::template project<GridLayout::implT::faceYToCellCenter>(By, index);
            auto const bz
                = GridLayout::template project<GridLayout::implT::faceZToCellCenter>(Bz, index);

            auto const P     = core::useFaceEnergy()
                                   ? (gamma_ - 1.0)
                                         * (Etot(index) - 0.5 * r * (vx * vx + vy * vy + vz * vz)
                                            - 0.5 * core::faceMagneticSquare<GridLayout>(B, index))
                                   : core::eosEtotToP(gamma_, r, vx, vy, vz, bx, by, bz, Etot(index));
            auto const BdotB = bx * bx + by * by + bz * bz;

            std::array<double, 3> const v{vx, vy, vz};
            std::array<double, 3> const b{bx, by, bz};

            // sum_d (|v_d| + c_fast_d + c_whistler_d) / dx_d over simulated directions
            double invDtWave = 0;
            for (std::size_t d = 0; d < dimension; ++d)
            {
                auto const cfast = core::compute_fast_magnetosonic_(gamma_, r, b[d], BdotB, P);
                // Hall whistler
                auto const cw = isHall_ ? std::numbers::pi
                                              * core::compute_whistler_(1.0 / meshSize[d], r, BdotB)
                                        : 0.0;
                invDtWave += (std::abs(v[d]) + cfast + cw) / meshSize[d];
            }
            dt = std::min(dt, wave / invDtWave);
        });
    }

    return dt;
}

template<typename MHDModel, typename AMR_Types, typename TimeIntegratorStrategy>
void SolverMHD<MHDModel, AMR_Types, TimeIntegratorStrategy>::mhdNaNCheck_(MHDModel& model,
                                                                          level_t const& level,
                                                                          double time)
{
    auto& rm  = model.resourcesManager;
    auto& rho = model.state.rho;

    auto check_nans = [&](auto const& field, auto const& layout,
                          core::MeshIndex<MHDModel::dimension> const& index) {
        if (std::isnan(field(index)))
        {
            std::stringstream ss;
            ss << "NaN detected in MHD field at index " << index << " on patch of origin "
               << layout.origin() << " on level " << level.getLevelNumber() << " at time " << time;
            // experiment: the AMR cell index and centre, ghost width read from the layout
            core::Point<int, MHDModel::dimension> lp;
            for (std::size_t d = 0; d < MHDModel::dimension; ++d)
                lp[d] = static_cast<int>(index[d]);
            auto const amr = layout.localToAMR(lp);
            ss << " | ghosts(dual)=" << layout.physicalStartIndex(core::QtyCentering::dual,
                                                                 core::Direction::X)
               << " AMR index=" << amr << " cell centre=" << layout.cellCenteredCoordinates(amr);
            core::DictionaryException ex{"cause", ss.str()};
            throw ex;
        }
    };

    for (auto const& patch : rm->enumerate(level, rho))
    {
        auto layout = amr::layoutFromPatch<GridLayout>(*patch);
        layout.evalOnGhostBox(
            rho, [&](auto const&... args) { check_nans(rho, layout, {args...}); });
    }
}

} // namespace PHARE::solver

#endif
