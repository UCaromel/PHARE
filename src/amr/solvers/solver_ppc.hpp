#ifndef PHARE_SOLVER_PPC_HPP
#define PHARE_SOLVER_PPC_HPP

#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "core/numerics/ohm/ohm.hpp"
#include "core/utilities/mpi_utils.hpp"
#include "core/numerics/ampere/ampere.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/faraday/faraday.hpp"
#include "core/data/grid/gridlayout_utils.hpp"
#include "core/utilities/index/index.hpp"
#include "core/numerics/ion_updater/ion_updater.hpp"
#include "core/numerics/interpolator/interpolator.hpp"

#include "amr/solvers/solver.hpp"
#include "amr/solvers/reflux_geometry.hpp"
#include "amr/solvers/momentum_tensor_border_filler.hpp"
#include "amr/messengers/hybrid_messenger.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "amr/solvers/solver_ppc_model_view.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/messengers/hybrid_messenger_info.hpp"

#include <SAMRAI/hier/Patch.h>
#include "SAMRAI/hier/PatchLevel.h"

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>


namespace PHARE::solver
{
// -----------------------------------------------------------------------------

template<typename HybridModel, typename AMR_Types>
class SolverPPC : public ISolver<AMR_Types>
{
private:
    static constexpr auto dimension    = HybridModel::dimension;
    static constexpr auto interp_order = HybridModel::gridlayout_type::interp_order;

    using Electromag       = HybridModel::electromag_type;
    using Ions             = HybridModel::ions_type;
    using ParticleArray    = Ions::particle_array_type;
    using VecFieldT        = HybridModel::vecfield_type;
    using GridLayout       = HybridModel::gridlayout_type;
    using ResourcesManager = HybridModel::resources_manager_type;
    using IPhysicalModel_t = IPhysicalModel<AMR_Types>;
    using IMessenger       = amr::IMessenger<IPhysicalModel_t>;
    using HybridMessenger  = amr::HybridMessenger<HybridModel>;

    using ModelViews_t = HybridPPCModelView<HybridModel>;
    using Faraday_t    = ModelViews_t::Faraday_t;
    using Ampere_t     = ModelViews_t::Ampere_t;
    using Ohm_t        = ModelViews_t::Ohm_t;
    using IonUpdater_t = PHARE::core::IonUpdater<Ions, Electromag, GridLayout>;
    using FieldT       = HybridModel::field_type;

    Electromag electromagPred_{"EMPred"};
    Electromag electromagAvg_{"EMAvg"};

    VecFieldT Bold_{this->name() + "_Bold", core::PhysicalQuantity::Vector::B};
    VecFieldT fluxSumE_{this->name() + "_fluxSumE", core::PhysicalQuantity::Vector::E};

    // Flux accumulators for MHD-Hybrid coupling reflux (accumulated over Hybrid subcycle).
    // Per-direction layout mirrors MHD AllFluxes:
    //   fluxSumRho_fd:  scalar FieldT at d-face  (ScalarFlux_x/y/z centering)
    //   fluxSumRhoV_fd: VecFieldT at d-face, all 3 momentum components (VecFlux_x/y/z centering)
    //   fluxSumEtot_fd: scalar FieldT at d-face  (ScalarFlux_x/y/z centering)
    FieldT    fluxSumRho_fx_{this->name() + "_fluxSumRho_fx",   core::PhysicalQuantity::Scalar::ScalarFlux_x};
    FieldT    fluxSumRho_fy_{this->name() + "_fluxSumRho_fy",   core::PhysicalQuantity::Scalar::ScalarFlux_y};
    FieldT    fluxSumRho_fz_{this->name() + "_fluxSumRho_fz",   core::PhysicalQuantity::Scalar::ScalarFlux_z};
    VecFieldT fluxSumRhoV_fx_{this->name() + "_fluxSumRhoV_fx", core::PhysicalQuantity::Vector::VecFlux_x};
    VecFieldT fluxSumRhoV_fy_{this->name() + "_fluxSumRhoV_fy", core::PhysicalQuantity::Vector::VecFlux_y};
    VecFieldT fluxSumRhoV_fz_{this->name() + "_fluxSumRhoV_fz", core::PhysicalQuantity::Vector::VecFlux_z};
    FieldT    fluxSumEtot_fx_{this->name() + "_fluxSumEtot_fx", core::PhysicalQuantity::Scalar::ScalarFlux_x};
    FieldT    fluxSumEtot_fy_{this->name() + "_fluxSumEtot_fy", core::PhysicalQuantity::Scalar::ScalarFlux_y};
    FieldT    fluxSumEtot_fz_{this->name() + "_fluxSumEtot_fz", core::PhysicalQuantity::Scalar::ScalarFlux_z};

    std::unordered_map<std::size_t, double> oldTime_;

    Faraday_t faraday_;
    Ampere_t ampere_;
    Ohm_t ohm_;

    IonUpdater_t ionUpdater_;

    double const gamma_i_; // ion heat capacity ratio (same dict key as MHD: "heat_capacity_ratio")

    // levelNumber → coarser model is MHD; set in advanceLevel before any accumulateFluxSum
    // read for that level. Pure-hybrid AMR refluxes only electromag: hydro flux sums and the
    // momentum-tensor deposit are skipped when not coupled.
    std::unordered_map<int, bool> coupledHydroFluxSum_;

    // Border-completes the ion momentum tensor on a level before the coupled momentum/energy
    // flux consumes it (coupled-path only). Owns deposit + per-pop seam-sum + aggregate; its
    // schedule cache is cleared on regrid.
    amr::MomentumTensorBorderFiller<ResourcesManager, GridLayout, Ions, dimension, interp_order>
        momentumTensorFiller_;


public:
    using patch_t     = AMR_Types::patch_t;
    using level_t     = AMR_Types::level_t;
    using hierarchy_t = AMR_Types::hierarchy_t;



    explicit SolverPPC(PHARE::initializer::PHAREDict const& dict)
        : ISolver<AMR_Types>{"PPC"}
        , ohm_{dict["ohm"]}
        , ionUpdater_{dict["ion_updater"]}
        , gamma_i_{dict["heat_capacity_ratio"].template to<double>()}
    {
    }

    ~SolverPPC() override = default;


    std::string modelName() const override { return HybridModel::model_name; }

    void fillMessengerInfo(std::unique_ptr<amr::IMessengerInfo> const& info) const override;


    void registerResources(IPhysicalModel_t& model) override;


    void allocate(IPhysicalModel_t& model, SAMRAI::hier::Patch& patch,
                  double const allocateTime) const override;

    void prepareStep(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level,
                     double const currentTime) override;

    void accumulateFluxSum(IPhysicalModel_t& model,
                           std::shared_ptr<SAMRAI::hier::PatchLevel> const& level,
                           double const coef,
                           SAMRAI::hier::CoarseFineBoundary const& cfBoundary) override;

    void resetFluxSum(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level) override;

    void reflux(IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level, IMessenger& messenger,
                double const time,
                SAMRAI::hier::CoarseFineBoundary const& fineCfBdry,
                SAMRAI::hier::PatchLevel const& fineLevel) override;

    void advanceLevel(hierarchy_t const& hierarchy, int const levelNumber, ISolverModelView& views,
                      IMessenger& fromCoarserMessenger, double const currentTime,
                      double const newTime) override;


    void onRegrid() override
    {
        boxing.clear();
        ionUpdater_.reset();
        coupledHydroFluxSum_.clear();
        momentumTensorFiller_.onRegrid();
    }


    std::shared_ptr<ISolverModelView> make_view(level_t& level, IPhysicalModel_t& model) override
    {
        return std::make_shared<ModelViews_t>(level, dynamic_cast<HybridModel&>(model));
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(Bold_, fluxSumE_,
                                     fluxSumRho_fx_, fluxSumRho_fy_, fluxSumRho_fz_,
                                     fluxSumRhoV_fx_, fluxSumRhoV_fy_, fluxSumRhoV_fz_,
                                     fluxSumEtot_fx_, fluxSumEtot_fy_, fluxSumEtot_fz_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(Bold_, fluxSumE_,
                                     fluxSumRho_fx_, fluxSumRho_fy_, fluxSumRho_fz_,
                                     fluxSumRhoV_fx_, fluxSumRhoV_fy_, fluxSumRhoV_fz_,
                                     fluxSumEtot_fx_, fluxSumEtot_fy_, fluxSumEtot_fz_);
    }


private:
    using Messenger = amr::HybridMessenger<HybridModel>;


    void predictor1_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                     double const currentTime, double const newTime);


    void predictor2_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                     double const currentTime, double const newTime);


    void corrector_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                    double const currentTime, double const newTime);


    void average_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                  double const newTime);


    void moveIons_(level_t& level, ModelViews_t& views, Messenger& fromCoarser,
                   double const currentTime, double const newTime, core::UpdaterMode mode);


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


    void make_boxes(hierarchy_t const& hierarchy, level_t& level)
    {
        int const lvlNbr = level.getLevelNumber();
        if (boxing.count(lvlNbr))
            return;

        auto& levelBoxing = boxing[lvlNbr]; // creates if missing

        for (auto const& patch : level)
            if (auto [it, suc] = levelBoxing.try_emplace(
                    amr::to_string(patch->getGlobalId()),
                    Boxing_t{amr::layoutFromPatch<GridLayout>(*patch),
                             amr::makeNonLevelGhostBoxFor<GridLayout>(*patch, hierarchy)});
                !suc)
                throw std::runtime_error("boxing map insertion failure");
    }

    auto& setup_level(hierarchy_t const& hierarchy, int const levelNumber)
    {
        auto level = hierarchy.getPatchLevel(levelNumber);
        if (boxing.count(levelNumber) == 0)
            make_boxes(hierarchy, *level);
        return *level;
    }


    using Boxing_t = core::UpdaterSelectionBoxing<IonUpdater_t, GridLayout>;
    std::unordered_map<int /*level*/, std::unordered_map<std::string /*patchid*/, Boxing_t>> boxing;


}; // end solverPPC



// -----------------------------------------------------------------------------



template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::registerResources(IPhysicalModel_t& model)
{
    auto& hmodel = dynamic_cast<HybridModel&>(model);
    hmodel.resourcesManager->registerResources(electromagPred_);
    hmodel.resourcesManager->registerResources(electromagAvg_);

    hmodel.resourcesManager->registerResources(Bold_);
    hmodel.resourcesManager->registerResources(fluxSumE_);
    hmodel.resourcesManager->registerResources(fluxSumRho_fx_);
    hmodel.resourcesManager->registerResources(fluxSumRhoV_fx_);
    hmodel.resourcesManager->registerResources(fluxSumEtot_fx_);

    if constexpr (dimension >= 2)
    {
        hmodel.resourcesManager->registerResources(fluxSumRho_fy_);
        hmodel.resourcesManager->registerResources(fluxSumRhoV_fy_);
        hmodel.resourcesManager->registerResources(fluxSumEtot_fy_);

        if constexpr (dimension == 3)
        {
            hmodel.resourcesManager->registerResources(fluxSumRho_fz_);
            hmodel.resourcesManager->registerResources(fluxSumRhoV_fz_);
            hmodel.resourcesManager->registerResources(fluxSumEtot_fz_);
        }
    }

    hmodel.resourcesManager->registerResources(momentumTensorFiller_.stagingField());
    momentumTensorFiller_.declareAlgos(hmodel.state.ions, *hmodel.resourcesManager);
}




template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::allocate(IPhysicalModel_t& model,
                                                 SAMRAI::hier::Patch& patch,
                                                 double const allocateTime) const
{
    auto& hmodel = dynamic_cast<HybridModel&>(model);
    hmodel.resourcesManager->allocate(electromagPred_, patch, allocateTime);
    hmodel.resourcesManager->allocate(electromagAvg_, patch, allocateTime);

    hmodel.resourcesManager->allocate(Bold_, patch, allocateTime);
    hmodel.resourcesManager->allocate(fluxSumE_, patch, allocateTime);
    hmodel.resourcesManager->allocate(fluxSumRho_fx_, patch, allocateTime);
    hmodel.resourcesManager->allocate(fluxSumRhoV_fx_, patch, allocateTime);
    hmodel.resourcesManager->allocate(fluxSumEtot_fx_, patch, allocateTime);

    if constexpr (dimension >= 2)
    {
        hmodel.resourcesManager->allocate(fluxSumRho_fy_, patch, allocateTime);
        hmodel.resourcesManager->allocate(fluxSumRhoV_fy_, patch, allocateTime);
        hmodel.resourcesManager->allocate(fluxSumEtot_fy_, patch, allocateTime);

        if constexpr (dimension == 3)
        {
            hmodel.resourcesManager->allocate(fluxSumRho_fz_, patch, allocateTime);
            hmodel.resourcesManager->allocate(fluxSumRhoV_fz_, patch, allocateTime);
            hmodel.resourcesManager->allocate(fluxSumEtot_fz_, patch, allocateTime);
        }
    }

    hmodel.resourcesManager->allocate(momentumTensorFiller_.stagingField(), patch, allocateTime);
}




template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::fillMessengerInfo(
    std::unique_ptr<amr::IMessengerInfo> const& info) const
{
    auto& hybridInfo = dynamic_cast<amr::HybridMessengerInfo&>(*info);

    auto const& Eavg  = electromagAvg_.E;
    auto const& Bpred = electromagPred_.B;

    hybridInfo.ghostElectric.emplace_back(Eavg.name());
    hybridInfo.initMagnetic.emplace_back(Bpred.name());
    hybridInfo.ghostMagnetic.emplace_back(Bpred.name());
    hybridInfo.refluxElectric  = Eavg.name();
    hybridInfo.fluxSumElectric = fluxSumE_.name();
    hybridInfo.fluxSumRho_fx   = fluxSumRho_fx_.name();
    hybridInfo.fluxSumRhoV_fx  = fluxSumRhoV_fx_.name();
    hybridInfo.fluxSumEtot_fx  = fluxSumEtot_fx_.name();

    if constexpr (dimension >= 2)
    {
        hybridInfo.fluxSumRho_fy  = fluxSumRho_fy_.name();
        hybridInfo.fluxSumRhoV_fy = fluxSumRhoV_fy_.name();
        hybridInfo.fluxSumEtot_fy = fluxSumEtot_fy_.name();

        if constexpr (dimension == 3)
        {
            hybridInfo.fluxSumRho_fz  = fluxSumRho_fz_.name();
            hybridInfo.fluxSumRhoV_fz = fluxSumRhoV_fz_.name();
            hybridInfo.fluxSumEtot_fz = fluxSumEtot_fz_.name();
        }
    }
}


template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::prepareStep(IPhysicalModel_t& model,
                                                    SAMRAI::hier::PatchLevel& level,
                                                    double const currentTime)
{
    oldTime_[level.getLevelNumber()] = currentTime;

    auto& hybridModel = dynamic_cast<HybridModel&>(model);
    auto& B           = hybridModel.state.electromag.B;

    for (auto& patch : level)
    {
        auto dataOnPatch = hybridModel.resourcesManager->setOnPatch(*patch, B, Bold_);

        hybridModel.resourcesManager->setTime(Bold_, *patch, currentTime);

        Bold_.copyData(B);
    }
}


template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::accumulateFluxSum(
    IPhysicalModel_t& model, std::shared_ptr<SAMRAI::hier::PatchLevel> const& level,
    double const coef, SAMRAI::hier::CoarseFineBoundary const& cfBoundary)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::accumulateFluxSum");

    auto& hybridModel = dynamic_cast<HybridModel&>(model);

    bool const coupled = [&] {
        auto const it = coupledHydroFluxSum_.find(level->getLevelNumber());
        return it != coupledHydroFluxSum_.end() and it->second;
    }();

    // Pass 1: electromagnetic flux accumulation — unconditional (pure-hybrid AMR refluxes
    // only electromag). Hydro + momentum tensor are coupled-path only and handled below.
    for (auto& patch : *level)
    {
        auto& Eavg         = electromagAvg_.E;
        auto const& layout = amr::layoutFromPatch<GridLayout>(*patch);

        auto _ = hybridModel.resourcesManager->setOnPatch(*patch, fluxSumE_, Eavg);

        auto const addScalar = [&](auto& left, auto const& right,
                                   core::Point<int, dimension> const& amrIdx) {
            auto const idx = layout.AMRToLocal(amrIdx);
            left(idx) += right(idx) * coef;
        };

        // E field accumulation. Geometry (codim-1 vs codim-2, transverse clipping, Ez
        // primal-endpoint patching) lives in the dim-generic enumerator shared with MHD,
        // which returns read-shifted, box-deduped containers per E component.
        auto const& patchCellBox = patch->getBox();
        auto const eBoxes = reflux_geometry::cfElectricBoxes<dimension>(
            cfBoundary, patch->getGlobalId(), patchCellBox);

        auto const accumulateE = [&](SAMRAI::hier::BoxContainer const& boxes,
                                     core::Component comp) {
            for (auto const& box : boxes)
                for (auto const& amrIdx : amr::phare_box_from<dimension>(box))
                    addScalar(fluxSumE_(comp), Eavg(comp), amrIdx);
        };

        accumulateE(eBoxes.ex, core::Component::X);
        accumulateE(eBoxes.ey, core::Component::Y);
        accumulateE(eBoxes.ez, core::Component::Z);
    }

    // Hydro flux sums + momentum tensor feed only the MHD-Hybrid reflux path; pure-hybrid AMR
    // refluxes only electromag, so skip the rest when the coarser model is hybrid.
    if (not coupled)
        return;

    auto& ions = hybridModel.state.ions;

    // Border-complete the ion momentum tensor on this level before the momentum/energy flux
    // consumes it. M was previously deposited per-patch with no border completion, leaving
    // partial sums at patch/sibling-patch seams (which lie on the CFB in a multi-patch fine
    // level) for the flux to read. The filler deposits from particles, additively seam-sums
    // each population tensor across borders, then aggregates into ions.momentumTensor().
    momentumTensorFiller_.fillCompleteMomentumTensor(ions, level, *hybridModel.resourcesManager,
                                                     /*fillTime*/ 0.);

    // Pass 2: hydro + Poynting flux sums (coupled levels only), reading the now
    // border-complete momentum tensor.
    for (auto& patch : *level)
    {
        auto const& layout = amr::layoutFromPatch<GridLayout>(*patch);
        auto& EM           = hybridModel.state.electromag;
        auto& B            = EM.B;
        auto& E            = EM.E;

        auto _ = hybridModel.resourcesManager->setOnPatch(
            *patch, fluxSumRho_fx_, fluxSumRhoV_fx_, fluxSumEtot_fx_, ions, B, E,
            hybridModel.state.electrons);

        auto const& Pe      = hybridModel.state.electrons.pressure();
        auto const& rho     = ions.massDensity();
        auto const& Vi      = ions.velocity();
        auto const& Vx      = Vi(core::Component::X);
        auto const& Vy      = Vi(core::Component::Y);
        auto const& Vz      = Vi(core::Component::Z);
        auto const& MT      = ions.momentumTensor();
        auto const& Mxx     = MT(core::Component::XX);
        auto const& Mxy     = MT(core::Component::XY);
        auto const& Mxz     = MT(core::Component::XZ);
        auto const& Myy     = MT(core::Component::YY);
        auto const& Myz     = MT(core::Component::YZ);
        auto const& Mzz     = MT(core::Component::ZZ);
        auto const& Bx      = B(core::Component::X);
        auto const& By      = B(core::Component::Y);
        auto const& Bz      = B(core::Component::Z);
        auto const& Ex      = E(core::Component::X);
        auto const& Ey      = E(core::Component::Y);
        auto const& Ez      = E(core::Component::Z);

        // ---- x-face (pdd): mass flux, full momentum tensor, Poynting ----

        layout.evalOnBox(fluxSumRho_fx_, [&](auto const&... args) mutable {
            core::MeshIndex<dimension> idx{args...};
            auto constexpr s = GridLayout::momentsToBx();
            fluxSumRho_fx_(args...)
                += coef * GridLayout::project(rho, idx, s) * GridLayout::project(Vx, idx, s);
        });

        // rhoV_fx(X) = Mxx + (By² + Bz² − Bx²)/2  [Maxwell: (B²/2)δxx − BxBx]
        // Mxx = Σ m vx² is the full kinetic stress (ρVx² + Pxx), tensor form replacing ρVx²+Pi_iso.
        layout.evalOnBox(fluxSumRhoV_fx_(core::Component::X), [&](auto const&... args) mutable {
            core::MeshIndex<dimension> idx{args...};
            auto constexpr s  = GridLayout::momentsToBx();
            auto const Mxx_a  = GridLayout::project(Mxx, idx, s);
            auto const Bx_a   = Bx(args...);
            auto const By_a   = GridLayout::project(By, idx, GridLayout::ByToBx());
            auto const Bz_a   = GridLayout::project(Bz, idx, GridLayout::BzToBx());
            fluxSumRhoV_fx_(core::Component::X)(args...)
                += coef * (Mxx_a + (By_a*By_a + Bz_a*Bz_a - Bx_a*Bx_a) / 2.0);
        });

        // rhoV_fx(Y) = Mxy − BxBy
        layout.evalOnBox(fluxSumRhoV_fx_(core::Component::Y), [&](auto const&... args) mutable {
            core::MeshIndex<dimension> idx{args...};
            auto constexpr s = GridLayout::momentsToBx();
            auto const Mxy_a = GridLayout::project(Mxy, idx, s);
            auto const Bx_a  = Bx(args...);
            auto const By_a  = GridLayout::project(By, idx, GridLayout::ByToBx());
            fluxSumRhoV_fx_(core::Component::Y)(args...)
                += coef * (Mxy_a - Bx_a*By_a);
        });

        // rhoV_fx(Z) = Mxz − BxBz
        layout.evalOnBox(fluxSumRhoV_fx_(core::Component::Z), [&](auto const&... args) mutable {
            core::MeshIndex<dimension> idx{args...};
            auto constexpr s = GridLayout::momentsToBx();
            auto const Mxz_a = GridLayout::project(Mxz, idx, s);
            auto const Bx_a  = Bx(args...);
            auto const Bz_a  = GridLayout::project(Bz, idx, GridLayout::BzToBx());
            fluxSumRhoV_fx_(core::Component::Z)(args...)
                += coef * (Mxz_a - Bx_a*Bz_a);
        });

        // Etot_fx: Poynting S_x = EyBz − EzBy  (all averaged to x-face pdd, μ₀=1)
        layout.evalOnBox(fluxSumEtot_fx_, [&](auto const&... args) mutable {
            core::MeshIndex<dimension> idx{args...};
            auto const Ey_a = GridLayout::project(Ey, idx, GridLayout::EyToBx());
            auto const Bz_a = GridLayout::project(Bz, idx, GridLayout::BzToBx());
            auto const Ez_a = GridLayout::project(Ez, idx, GridLayout::EzToBx());
            auto const By_a = GridLayout::project(By, idx, GridLayout::ByToBx());
            fluxSumEtot_fx_(args...) += coef * (Ey_a*Bz_a - Ez_a*By_a);
        });

        // Etot_fx: enthalpy flux (½ρV² + γ_i/(γ_i-1)·(Pi_iso + Pe))·Vx
        // Pi_iso = isotropic ion pressure (same approximation as momentum flux off-diagonal = 0).
        // Pe from electron closure (general: any closure exposing pressure() works here).
        // Electron enthalpy factor matches the Etot density convention Pe/(γ−1) + Pe work term.
        layout.evalOnBox(fluxSumEtot_fx_, [&, hi = gamma_i_ / (gamma_i_ - 1.0)](auto const&... args) mutable {
            core::MeshIndex<dimension> idx{args...};
            auto constexpr s  = GridLayout::momentsToBx();
            auto const rho_a  = GridLayout::project(rho, idx, s);
            auto const Vx_a   = GridLayout::project(Vx,  idx, s);
            auto const Vy_a   = GridLayout::project(Vy,  idx, s);
            auto const Vz_a   = GridLayout::project(Vz,  idx, s);
            auto const Mxx_a  = GridLayout::project(Mxx, idx, s);
            auto const Myy_a  = GridLayout::project(Myy, idx, s);
            auto const Mzz_a  = GridLayout::project(Mzz, idx, s);
            auto const Pe_a   = GridLayout::project(Pe,  idx, s);
            auto const V2_a   = Vx_a*Vx_a + Vy_a*Vy_a + Vz_a*Vz_a;
            auto const Pi_a   = (Mxx_a + Myy_a + Mzz_a - rho_a*V2_a) / 3.0;
            fluxSumEtot_fx_(args...) += coef * (0.5*rho_a*V2_a + hi*(Pi_a + Pe_a)) * Vx_a;
        });

        if constexpr (dimension >= 2)
        {
            auto _y = hybridModel.resourcesManager->setOnPatch(
                *patch, fluxSumRho_fy_, fluxSumRhoV_fy_, fluxSumEtot_fy_);

            // ---- y-face (dpd): mass flux, full momentum tensor, Poynting ----

            layout.evalOnBox(fluxSumRho_fy_, [&](auto const&... args) mutable {
                core::MeshIndex<dimension> idx{args...};
                auto constexpr s = GridLayout::momentsToBy();
                fluxSumRho_fy_(args...)
                    += coef * GridLayout::project(rho, idx, s) * GridLayout::project(Vy, idx, s);
            });

            // rhoV_fy(X) = Myx − ByBx   (Myx = Mxy, symmetric)
            layout.evalOnBox(fluxSumRhoV_fy_(core::Component::X), [&](auto const&... args) mutable {
                core::MeshIndex<dimension> idx{args...};
                auto constexpr s = GridLayout::momentsToBy();
                auto const Mxy_a = GridLayout::project(Mxy, idx, s);
                auto const By_a  = By(args...);
                auto const Bx_a  = GridLayout::project(Bx, idx, GridLayout::BxToBy());
                fluxSumRhoV_fy_(core::Component::X)(args...)
                    += coef * (Mxy_a - By_a*Bx_a);
            });

            // rhoV_fy(Y) = Myy + (Bx² + Bz² − By²)/2
            layout.evalOnBox(fluxSumRhoV_fy_(core::Component::Y), [&](auto const&... args) mutable {
                core::MeshIndex<dimension> idx{args...};
                auto constexpr s = GridLayout::momentsToBy();
                auto const Myy_a = GridLayout::project(Myy, idx, s);
                auto const By_a  = By(args...);
                auto const Bx_a  = GridLayout::project(Bx, idx, GridLayout::BxToBy());
                auto const Bz_a  = GridLayout::project(Bz, idx, GridLayout::BzToBy());
                fluxSumRhoV_fy_(core::Component::Y)(args...)
                    += coef * (Myy_a + (Bx_a*Bx_a + Bz_a*Bz_a - By_a*By_a) / 2.0);
            });

            // rhoV_fy(Z) = Myz − ByBz
            layout.evalOnBox(fluxSumRhoV_fy_(core::Component::Z), [&](auto const&... args) mutable {
                core::MeshIndex<dimension> idx{args...};
                auto constexpr s = GridLayout::momentsToBy();
                auto const Myz_a = GridLayout::project(Myz, idx, s);
                auto const By_a  = By(args...);
                auto const Bz_a  = GridLayout::project(Bz, idx, GridLayout::BzToBy());
                fluxSumRhoV_fy_(core::Component::Z)(args...)
                    += coef * (Myz_a - By_a*Bz_a);
            });

            // Etot_fy: Poynting S_y = EzBx − ExBz
            layout.evalOnBox(fluxSumEtot_fy_, [&](auto const&... args) mutable {
                core::MeshIndex<dimension> idx{args...};
                auto const Ez_a = GridLayout::project(Ez, idx, GridLayout::EzToBy());
                auto const Bx_a = GridLayout::project(Bx, idx, GridLayout::BxToBy());
                auto const Ex_a = GridLayout::project(Ex, idx, GridLayout::ExToBy());
                auto const Bz_a = GridLayout::project(Bz, idx, GridLayout::BzToBy());
                fluxSumEtot_fy_(args...) += coef * (Ez_a*Bx_a - Ex_a*Bz_a);
            });

            // Etot_fy: enthalpy flux (½ρV² + γ_i/(γ_i-1)·(Pi_iso + Pe))·Vy
            layout.evalOnBox(fluxSumEtot_fy_, [&, hi = gamma_i_ / (gamma_i_ - 1.0)](auto const&... args) mutable {
                core::MeshIndex<dimension> idx{args...};
                auto constexpr s  = GridLayout::momentsToBy();
                auto const rho_a  = GridLayout::project(rho, idx, s);
                auto const Vx_a   = GridLayout::project(Vx,  idx, s);
                auto const Vy_a   = GridLayout::project(Vy,  idx, s);
                auto const Vz_a   = GridLayout::project(Vz,  idx, s);
                auto const Mxx_a  = GridLayout::project(Mxx, idx, s);
                auto const Myy_a  = GridLayout::project(Myy, idx, s);
                auto const Mzz_a  = GridLayout::project(Mzz, idx, s);
                auto const Pe_a   = GridLayout::project(Pe,  idx, s);
                auto const V2_a   = Vx_a*Vx_a + Vy_a*Vy_a + Vz_a*Vz_a;
                auto const Pi_a   = (Mxx_a + Myy_a + Mzz_a - rho_a*V2_a) / 3.0;
                fluxSumEtot_fy_(args...) += coef * (0.5*rho_a*V2_a + hi*(Pi_a + Pe_a)) * Vy_a;
            });

            if constexpr (dimension == 3)
            {
                auto _z = hybridModel.resourcesManager->setOnPatch(
                    *patch, fluxSumRho_fz_, fluxSumRhoV_fz_, fluxSumEtot_fz_);

                // ---- z-face (ddp): mass flux, full momentum tensor, Poynting ----

                layout.evalOnBox(fluxSumRho_fz_, [&](auto const&... args) mutable {
                    core::MeshIndex<dimension> idx{args...};
                    auto constexpr s = GridLayout::momentsToBz();
                    fluxSumRho_fz_(args...)
                        += coef * GridLayout::project(rho, idx, s) * GridLayout::project(Vz, idx, s);
                });

                // rhoV_fz(X) = Mzx − BzBx   (Mzx = Mxz, symmetric)
                layout.evalOnBox(fluxSumRhoV_fz_(core::Component::X), [&](auto const&... args) mutable {
                    core::MeshIndex<dimension> idx{args...};
                    auto constexpr s = GridLayout::momentsToBz();
                    auto const Mxz_a = GridLayout::project(Mxz, idx, s);
                    auto const Bz_a  = Bz(args...);
                    auto const Bx_a  = GridLayout::project(Bx, idx, GridLayout::BxToBz());
                    fluxSumRhoV_fz_(core::Component::X)(args...)
                        += coef * (Mxz_a - Bz_a*Bx_a);
                });

                // rhoV_fz(Y) = Mzy − BzBy   (Mzy = Myz, symmetric)
                layout.evalOnBox(fluxSumRhoV_fz_(core::Component::Y), [&](auto const&... args) mutable {
                    core::MeshIndex<dimension> idx{args...};
                    auto constexpr s = GridLayout::momentsToBz();
                    auto const Myz_a = GridLayout::project(Myz, idx, s);
                    auto const Bz_a  = Bz(args...);
                    auto const By_a  = GridLayout::project(By, idx, GridLayout::ByToBz());
                    fluxSumRhoV_fz_(core::Component::Y)(args...)
                        += coef * (Myz_a - Bz_a*By_a);
                });

                // rhoV_fz(Z) = Mzz + (Bx² + By² − Bz²)/2
                layout.evalOnBox(fluxSumRhoV_fz_(core::Component::Z), [&](auto const&... args) mutable {
                    core::MeshIndex<dimension> idx{args...};
                    auto constexpr s = GridLayout::momentsToBz();
                    auto const Mzz_a = GridLayout::project(Mzz, idx, s);
                    auto const Bz_a  = Bz(args...);
                    auto const Bx_a  = GridLayout::project(Bx, idx, GridLayout::BxToBz());
                    auto const By_a  = GridLayout::project(By, idx, GridLayout::ByToBz());
                    fluxSumRhoV_fz_(core::Component::Z)(args...)
                        += coef * (Mzz_a + (Bx_a*Bx_a + By_a*By_a - Bz_a*Bz_a) / 2.0);
                });

                // Etot_fz: Poynting S_z = ExBy − EyBx
                layout.evalOnBox(fluxSumEtot_fz_, [&](auto const&... args) mutable {
                    core::MeshIndex<dimension> idx{args...};
                    auto const Ex_a = GridLayout::project(Ex, idx, GridLayout::ExToBz());
                    auto const By_a = GridLayout::project(By, idx, GridLayout::ByToBz());
                    auto const Ey_a = GridLayout::project(Ey, idx, GridLayout::EyToBz());
                    auto const Bx_a = GridLayout::project(Bx, idx, GridLayout::BxToBz());
                    fluxSumEtot_fz_(args...) += coef * (Ex_a*By_a - Ey_a*Bx_a);
                });

                // Etot_fz: enthalpy flux (½ρV² + γ_i/(γ_i-1)·(Pi_iso + Pe))·Vz
                layout.evalOnBox(fluxSumEtot_fz_, [&, hi = gamma_i_ / (gamma_i_ - 1.0)](auto const&... args) mutable {
                    core::MeshIndex<dimension> idx{args...};
                    auto constexpr s  = GridLayout::momentsToBz();
                    auto const rho_a  = GridLayout::project(rho, idx, s);
                    auto const Vx_a   = GridLayout::project(Vx,  idx, s);
                    auto const Vy_a   = GridLayout::project(Vy,  idx, s);
                    auto const Vz_a   = GridLayout::project(Vz,  idx, s);
                    auto const Mxx_a  = GridLayout::project(Mxx, idx, s);
                    auto const Myy_a  = GridLayout::project(Myy, idx, s);
                    auto const Mzz_a  = GridLayout::project(Mzz, idx, s);
                    auto const Pe_a   = GridLayout::project(Pe,  idx, s);
                    auto const V2_a   = Vx_a*Vx_a + Vy_a*Vy_a + Vz_a*Vz_a;
                    auto const Pi_a   = (Mxx_a + Myy_a + Mzz_a - rho_a*V2_a) / 3.0;
                    fluxSumEtot_fz_(args...) += coef * (0.5*rho_a*V2_a + hi*(Pi_a + Pe_a)) * Vz_a;
                });
            }
        }
    }
}


template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::resetFluxSum(IPhysicalModel_t& model,
                                                     SAMRAI::hier::PatchLevel& level)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::resetFluxSum");

    auto& hybridModel = dynamic_cast<HybridModel&>(model);

    for (auto& patch : level)
    {
        auto _ = hybridModel.resourcesManager->setOnPatch(
            *patch, fluxSumE_, fluxSumRho_fx_, fluxSumRhoV_fx_, fluxSumEtot_fx_);

        fluxSumE_.zero();
        fluxSumRho_fx_.zero();
        fluxSumRhoV_fx_.zero();
        fluxSumEtot_fx_.zero();

        if constexpr (dimension >= 2)
        {
            auto _y = hybridModel.resourcesManager->setOnPatch(
                *patch, fluxSumRho_fy_, fluxSumRhoV_fy_, fluxSumEtot_fy_);

            fluxSumRho_fy_.zero();
            fluxSumRhoV_fy_.zero();
            fluxSumEtot_fy_.zero();

            if constexpr (dimension == 3)
            {
                auto _z = hybridModel.resourcesManager->setOnPatch(
                    *patch, fluxSumRho_fz_, fluxSumRhoV_fz_, fluxSumEtot_fz_);

                fluxSumRho_fz_.zero();
                fluxSumRhoV_fz_.zero();
                fluxSumEtot_fz_.zero();
            }
        }
    }
}


template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::reflux(
    IPhysicalModel_t& model, SAMRAI::hier::PatchLevel& level, IMessenger& messenger,
    double const time, SAMRAI::hier::CoarseFineBoundary const& /*fineCfBdry*/,
    SAMRAI::hier::PatchLevel const& fineLevel)
{
    auto& hybridModel     = dynamic_cast<HybridModel&>(model);
    auto& hybridMessenger = dynamic_cast<HybridMessenger&>(messenger);
    auto& B               = hybridModel.state.electromag.B;
    auto& Eavg            = electromagAvg_.E;

    auto const dt = time - oldTime_[level.getLevelNumber()];

    // Build coarsened fine domain from global fine boxes (MPI-collective, done once per call)
    auto const& globalFineBoxes = fineLevel.getBoxLevel()->getGlobalizedVersion().getGlobalBoxes();
    auto const ratio            = fineLevel.getRatioToCoarserLevel();

    std::vector<SAMRAI::hier::Box> coarsenedFine;
    for (auto const& box : globalFineBoxes)
        coarsenedFine.push_back(SAMRAI::hier::Box::coarsen(box, ratio));

    for (auto& coarsePatch : level)
    {
        auto const& patchAMRBox = coarsePatch->getBox();
        auto const& layout      = amr::layoutFromPatch<GridLayout>(*coarsePatch);
        auto _ = hybridModel.resourcesManager->setOnPatch(*coarsePatch, B, fluxSumE_, Eavg);

        // B correction via Faraday. Coarse Yee B-faces on the CF boundary are enumerated and
        // box-deduped per (component, dir, side) across all coarsened-fine boxes by the shared
        // geometry primitives. The normal-direction E read coordinate is recovered per face
        // from amrIdx[dir]. E source is the time-averaged Eavg (electromagAvg_.E); no hydro.
        for (int dir = 0; dir < static_cast<int>(dimension); ++dir)
            for (int side = 0; side < 2; ++side)
            {
                bool const isLower  = (side == 0);
                int const sign      = isLower ? +1 : -1;
                double const bScale = -sign * dt / layout.meshSize()[dir];

                for (auto const& t : reflux_geometry::faradayTerms<core::PhysicalQuantity::Scalar>(dir))
                {
                    auto const faces = reflux_geometry::cfBFaceBoxes(
                        layout, t.bQty, dir, side, patchAMRBox, coarsenedFine);

                    for (auto const& box : faces)
                        for (auto const& amrIdx : amr::phare_box_from<dimension>(box))
                        {
                            if (reflux_geometry::bFaceInsideFine(layout, t.bQty, dir, amrIdx,
                                                                 coarsenedFine))
                                continue;

                            auto eReadIdx   = amrIdx;
                            eReadIdx[dir]   = isLower ? amrIdx[dir] + 1 : amrIdx[dir];
                            auto const idxE = layout.AMRToLocal(eReadIdx);
                            auto const idx  = layout.AMRToLocal(amrIdx);

                            auto const tE = Eavg(t.eComp)(idxE);
                            auto const fE = fluxSumE_(t.eComp)(idxE);
                            B(t.bComp)(idx) += t.eSign * bScale * (tE - fE);
                        }
                }
            }
    }

    hybridMessenger.fillMagneticGhosts(B, level, time);
}



template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::advanceLevel(hierarchy_t const& hierarchy,
                                                     int const levelNumber, ISolverModelView& views,
                                                     IMessenger& fromCoarserMessenger,
                                                     double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::advanceLevel");

    auto& modelView   = dynamic_cast<ModelViews_t&>(views);
    auto& fromCoarser = dynamic_cast<HybridMessenger&>(fromCoarserMessenger);
    auto& level       = setup_level(hierarchy, levelNumber);

    coupledHydroFluxSum_[levelNumber]
        = fromCoarser.coarseModelName() != fromCoarser.fineModelName();

    predictor1_(level, modelView, fromCoarser, currentTime, newTime);

    average_(level, modelView, fromCoarser, newTime);

    moveIons_(level, modelView, fromCoarser, currentTime, newTime, core::UpdaterMode::domain_only);

    predictor2_(level, modelView, fromCoarser, currentTime, newTime);


    average_(level, modelView, fromCoarser, newTime);

    moveIons_(level, modelView, fromCoarser, currentTime, newTime, core::UpdaterMode::all);

    corrector_(level, modelView, fromCoarser, currentTime, newTime);
}




template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::predictor1_(level_t& level, ModelViews_t& views,
                                                    Messenger& fromCoarser,
                                                    double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::predictor1_");

    TimeSetter setTime{views, newTime};

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::predictor1_.faraday");
        auto dt = newTime - currentTime;
        faraday_(views.layouts, views.electromag_B, views.electromag_E, views.electromagPred_B, dt);
        setTime([](auto& state) -> auto& { return state.electromagPred.B; });
        fromCoarser.fillMagneticGhosts(electromagPred_.B, level, newTime);
    }

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::predictor1_.ampere");
        ampere_(views.layouts, views.electromagPred_B, views.J);
        setTime([](auto& state) -> auto& { return state.J; });
        fromCoarser.fillCurrentGhosts(views.model().state.J, level, newTime);
    }

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::predictor1_.ohm");
        for (auto& state : views)
            state.electrons.update(state.layout);
        ohm_(views.layouts, views.N, views.Ve, views.Pe, views.electromagPred_B, views.J,
             views.electromagPred_E);
        setTime([](auto& state) -> auto& { return state.electromagPred.E; });
    }
}


template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::predictor2_(level_t& level, ModelViews_t& views,
                                                    Messenger& fromCoarser,
                                                    double const currentTime, double const newTime)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::predictor2_");

    TimeSetter setTime{views, newTime};

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::predictor2_.faraday");
        auto dt = newTime - currentTime;
        faraday_(views.layouts, views.electromag_B, views.electromagAvg_E, views.electromagPred_B,
                 dt);
        setTime([](auto& state) -> auto& { return state.electromagPred.B; });
        fromCoarser.fillMagneticGhosts(electromagPred_.B, level, newTime);
    }

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::predictor2_.ampere");
        ampere_(views.layouts, views.electromagPred_B, views.J);
        setTime([](auto& state) -> auto& { return state.J; });
        fromCoarser.fillCurrentGhosts(views.model().state.J, level, newTime);
    }

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::predictor2_.ohm");
        for (auto& state : views)
            state.electrons.update(state.layout);
        ohm_(views.layouts, views.N, views.Ve, views.Pe, views.electromagPred_B, views.J,
             views.electromagPred_E);
        setTime([](auto& state) -> auto& { return state.electromagPred.E; });
    }
}




template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::corrector_(level_t& level, ModelViews_t& views,
                                                   Messenger& fromCoarser, double const currentTime,
                                                   double const newTime)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::corrector_");

    auto levelNumber = level.getLevelNumber();
    TimeSetter setTime{views, newTime};

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::corrector_.faraday");
        auto dt = newTime - currentTime;
        faraday_(views.layouts, views.electromag_B, views.electromagAvg_E, views.electromag_B, dt);
        setTime([](auto& state) -> auto& { return state.electromag.B; });
        fromCoarser.fillMagneticGhosts(views.model().state.electromag.B, level, newTime);
    }

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::corrector_.ampere");
        ampere_(views.layouts, views.electromag_B, views.J);
        setTime([](auto& state) -> auto& { return state.J; });
        fromCoarser.fillCurrentGhosts(views.model().state.J, level, newTime);
    }

    {
        PHARE_LOG_SCOPE(3, "SolverPPC::corrector_.ohm");
        for (auto& state : views)
            state.electrons.update(state.layout);
        ohm_(views.layouts, views.N, views.Ve, views.Pe, views.electromag_B, views.J,
             views.electromag_E);
        setTime([](auto& state) -> auto& { return state.electromag.E; });

        fromCoarser.fillElectricGhosts(views.model().state.electromag.E, level, newTime);
    }
}



template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::average_(level_t& level, ModelViews_t& views,
                                                 Messenger& fromCoarser, double const newTime)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::average_");

    TimeSetter setTime{views, newTime};

    for (auto& state : views)
    {
        PHARE::core::average(state.electromag.B, state.electromagPred.B, state.electromagAvg.B);
        PHARE::core::average(state.electromag.E, state.electromagPred.E, state.electromagAvg.E);
    }

    setTime([](auto& state) -> auto& { return state.electromagAvg.B; });
    setTime([](auto& state) -> auto& { return state.electromagAvg.E; });

    // the following will fill E on all edges of all ghost cells, including those
    // on domain border. For level ghosts, electric field will be obtained from
    // next coarser level E average
    fromCoarser.fillElectricGhosts(electromagAvg_.E, level, newTime);
}


template<typename... Args>
void _debug_log_move_ions(Args const&... args)
{
    auto const& [views] = std::forward_as_tuple(args...);

    std::size_t nbrLevelGhostOldParticles = 0;
    std::size_t nbrLevelGhostParticles    = 0;
    for (auto& state : views)
    {
        for (auto& pop : state.ions)
        {
            nbrLevelGhostOldParticles += pop.levelGhostParticlesOld().size();
            nbrLevelGhostParticles += pop.levelGhostParticles().size();

            if (nbrLevelGhostOldParticles < nbrLevelGhostParticles
                and nbrLevelGhostOldParticles > 0)
                throw std::runtime_error("Error - there are less old level ghost particles ("
                                         + std::to_string(nbrLevelGhostOldParticles)
                                         + ") than pushable ("
                                         + std::to_string(nbrLevelGhostParticles) + ")");
        }
    }
}


template<typename HybridModel, typename AMR_Types>
void SolverPPC<HybridModel, AMR_Types>::moveIons_(level_t& level, ModelViews_t& views,
                                                  Messenger& fromCoarser, double const currentTime,
                                                  double const newTime, core::UpdaterMode mode)
{
    PHARE_LOG_SCOPE(3, "SolverPPC::moveIons_");
    PHARE_DEBUG_DO(_debug_log_move_ions(views);)

    TimeSetter setTime{views, newTime};
    auto const& levelBoxing = boxing[level.getLevelNumber()];

    try
    {
        auto dt = newTime - currentTime;
        for (auto& state : views)
            ionUpdater_.updatePopulations(
                state.ions, state.electromagAvg,
                levelBoxing.at(amr::to_string(state.patch->getGlobalId())), dt, mode);
    }
    catch (core::DictionaryException const& ex)
    {
        PHARE_LOG_ERROR(ex());
    }
    if (core::mpi::any(core::Errors::instance().any()))
        throw core::DictionaryException{}("ID", "Updater::updatePopulations");

    // this needs to be done before calling the messenger
    setTime([](auto& state) -> auto& { return state.ions; });

    fromCoarser.fillFluxBorders(views.model().state.ions, level, newTime);
    fromCoarser.fillDensityBorders(views.model().state.ions, level, newTime);
    fromCoarser.fillIonPopMomentGhosts(views.model().state.ions, level, newTime);
    fromCoarser.fillIonGhostParticles(views.model().state.ions, level, newTime);

    for (auto& state : views)
        ionUpdater_.updateIons(state.ions);

    fromCoarser.fillIonBorders(views.model().state.ions, level, newTime);

    // no need to update time, since it has been done before
    // now Ni and Vi are calculated we can fill pure ghost nodes
    // these were not completed by the deposition of patch and levelghost particles
    // fromCoarser.fillIonMomentGhosts(views.model().state.ions, level, newTime);
}



} // namespace PHARE::solver




#endif
