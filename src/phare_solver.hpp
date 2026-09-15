
#ifndef PHARE_SOLVER_INCLUDE_HPP
#define PHARE_SOLVER_INCLUDE_HPP

#include "phare_amr.hpp" // IWYU pragma: keep

#include "amr/solvers/solver_mhd.hpp"
#include "amr/solvers/solver_ppc.hpp"
#include "amr/multiphysics_integrator.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/messengers/messenger_factory.hpp"
#include "amr/messengers/hybrid_hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mc2011_temporal_transfer.hpp"
#include "amr/physical_models/hybrid_model.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/data/particles/refine/splitter.hpp"
#include "amr/data/particles/refine/particles_data_split.hpp"
#include "amr/level_initializer/level_initializer_factory.hpp"
#include "amr/level_initializer/hybrid_level_initializer.hpp"
#include "amr/level_initializer/mhd_level_initializer.hpp"
#include "amr/solvers/mhd_resolver.hpp"

#include <memory>

namespace PHARE::solver
{
// Bool-specialized holders: compile-time model enabling, to avoid expanding templates and
// emitting binary symbols for a model the opts value does not ask for. The `false`
// specialization is empty, so hybrid- (or mhd-) disabled opts never name the corresponding
// Model_t/Solver_t/etc. as a type.
// The `enabled` default is on the primary template because partial specializations may not
// carry default template arguments.
template<auto opts, typename CoreTypes, bool enabled = has_hybrid_v<opts>>
struct HybridStack;

template<auto opts, typename CoreTypes>
struct HybridStack<opts, CoreTypes, false>
{
};

template<auto opts, typename CoreTypes>
struct HybridStack<opts, CoreTypes, true>
{
    using GridLayout_t = CoreTypes::Hybrid::GridLayout_t;
    using Model_t
        = HybridModel<GridLayout_t, typename CoreTypes::Hybrid::Electromag_t,
                      typename CoreTypes::Hybrid::Ions_t, typename CoreTypes::Hybrid::Electrons_t,
                      amr::SAMRAI_Types, typename CoreTypes::Hybrid::Grid_t>;
    using Solver_t = PHARE::solver::SolverPPC<Model_t, PHARE::amr::SAMRAI_Types>;

    using Splitter_t = PHARE::amr::Splitter<PHARE::core::DimConst<opts.dimension>,
                                            PHARE::core::InterpConst<opts.interp_order>,
                                            PHARE::core::RefinedParticlesConst<opts.nbRefinedPart>>;
    using RefinementParams_t
        = PHARE::amr::RefinementParams<typename CoreTypes::Hybrid::ParticleArray_t, Splitter_t>;

    using LevelInitializer_t = HybridLevelInitializer<Model_t>;
};

// A fourth-order scheme stores cell averages, so its initial condition has to be integrated over
// each element rather than sampled at the node -- point values carry an O(h^2) error that caps the
// run at second order on its own. Second order samples, its representation error sitting at its own
// truncation level. This is the only place that rule is written down: anything building an MHDModel
// -- tests included -- reads it from here rather than restating it.
template<MHDOpts::MHDOrder Order>
auto constexpr initRepresentationFor
    = Order == MHDOpts::MHDOrder::O4 ? core::InitRepresentation::CellAverage
                                     : core::InitRepresentation::PointValue;

template<MHDOpts::MHDOrder Order, MHDOpts::TimeIntegratorType Integrator, typename MHDModel>
struct MHDTemporalTransferSelector
{
    using type = amr::NoCoarseFineTemporalTransfer<MHDModel>;
};

template<MHDOpts::TimeIntegratorType Integrator, typename MHDModel>
struct MHDTemporalTransferSelector<MHDOpts::MHDOrder::O2, Integrator, MHDModel>
{
    using type = amr::Linear2TemporalTransfer<MHDModel>;
};

template<typename MHDModel>
struct MHDTemporalTransferSelector<MHDOpts::MHDOrder::O4, MHDOpts::TimeIntegratorType::SSPRK4_5,
                                   MHDModel>
{
    using type = amr::MC2011TemporalTransfer<MHDModel>;
};

template<auto opts, typename CoreTypes, bool enabled = has_mhd_v<opts>>
struct MHDStack;

template<auto opts, typename CoreTypes>
struct MHDStack<opts, CoreTypes, false>
{
};

template<auto opts, typename CoreTypes>
struct MHDStack<opts, CoreTypes, true>
{
    using GridLayout_t = CoreTypes::MHD::GridLayout_t;
    using Model_t
        = MHDModel<GridLayout_t, typename CoreTypes::MHD::VecField_t, amr::SAMRAI_Types,
                   typename CoreTypes::MHD::Grid_t, initRepresentationFor<opts.mhd_order>>;
    using TemporalTransfer_t =
        typename MHDTemporalTransferSelector<opts.mhd_order, opts.time_integrator_type,
                                             Model_t>::type;
    using Messenger_t = amr::MHDMessenger<Model_t, TemporalTransfer_t>;
    using Solver_t    = PHARE::solver::SolverMHD<Model_t, PHARE::amr::SAMRAI_Types,
                                                 typename MHDResolver<opts, Model_t>::MHDTimeStepper_t,
                                                 Messenger_t>;

    using LevelInitializer_t = MHDLevelInitializer<Model_t>;
};

// One specialization per enabled-combination, so a disabled model's types are never named as
// template arguments to MessengerFactory/LevelInitializerFactory.
template<auto opts, typename Hybrid, typename MHD, bool hasHybrid, bool hasMHD>
struct FactorySelector;

// hybrid-only
template<auto opts, typename Hybrid, typename MHD>
struct FactorySelector<opts, Hybrid, MHD, true, false>
{
    using Messenger_t
        = amr::MessengerFactory<typename Hybrid::Model_t, typename Hybrid::Model_t,
                                amr::HybridHybridMessengerStrategy<
                                    typename Hybrid::Model_t, typename Hybrid::RefinementParams_t>>;
    using LevelInit_t
        = LevelInitializerFactory<amr::SAMRAI_Types, typename Hybrid::LevelInitializer_t>;
};

// mhd-only
template<auto opts, typename Hybrid, typename MHD>
struct FactorySelector<opts, Hybrid, MHD, false, true>
{
    using Messenger_t = amr::MessengerFactory<typename MHD::Model_t, typename MHD::Model_t,
                                              typename MHD::Messenger_t>;
    using LevelInit_t
        = LevelInitializerFactory<amr::SAMRAI_Types, typename MHD::LevelInitializer_t>;
};

template<auto opts>
struct PHARE_Types
{
    static_assert(opts.mhd_axes_consistent());
    static_assert(!has_mhd_v<opts> || opts.mhd_order != MHDOpts::MHDOrder::O4
                      || opts.time_integrator_type == MHDOpts::TimeIntegratorType::TVDRK3
                      || opts.time_integrator_type == MHDOpts::TimeIntegratorType::SSPRK4_5,
                  "MHD4 requires TVDRK3 or SSPRK4_5");
    static_assert(!has_mhd_v<opts> || opts.mhd_order != MHDOpts::MHDOrder::O4
                      || opts.reconstruction_type == MHDOpts::ReconstructionType::WENOZ
                      || opts.reconstruction_type == MHDOpts::ReconstructionType::MP5,
                  "MHD4 reconstruction must be WENOZ or MP5");
    // Hyper-resistivity adds a fixed second-order modification of the equations, so it caps the
    // scheme at second order whatever the reconstruction does. Fourth order dissipates the
    // dispersive branch with the upwind whistler speed in the wave fan instead.
    static_assert(!has_mhd_v<opts> || opts.mhd_order != MHDOpts::MHDOrder::O4
                      || !opts.HyperResistivity,
                  "MHD4 does not accept hyper-resistivity");
    static_assert(has_hybrid_v<opts> || has_mhd_v<opts>, "a build must enable at least one model");
    static_assert(!(has_hybrid_v<opts> && has_mhd_v<opts>),
                  "mixed Hybrid/MHD builds are not supported");

    auto static constexpr dimension = opts.dimension;

    // core deps
    using core_types = PHARE::core::PHARE_Types<opts>;

    using Hybrid = HybridStack<opts, core_types>;
    using MHD    = MHDStack<opts, core_types>;

    using IPhysicalModel = PHARE::solver::IPhysicalModel<PHARE::amr::SAMRAI_Types>;

    // amr deps
    using amr_types = PHARE::amr::PHARE_Types<opts>;

    using Selector_t = FactorySelector<opts, Hybrid, MHD, has_hybrid_v<opts>, has_mhd_v<opts>>;

    using MessengerFactory // = amr/solver bidirectional dependency
        = Selector_t::Messenger_t;
    using LevelInitializerFactory_t = Selector_t::LevelInit_t;
    // amr deps

    using MultiPhysicsIntegrator_t
        = MultiPhysicsIntegrator<MessengerFactory, LevelInitializerFactory_t,
                                 PHARE::amr::SAMRAI_Types>;
};

// Same bool-specialization trick as HybridStack/MHDStack above, for runtime storage rather than
// types: the `false` specialization is empty, so a disabled model's resource manager / model
// shared_ptrs are simply not declared. Simulator never names the disabled model as a type outside
// a lazily-instantiated (non-virtual) function body.
template<auto opts, bool enabled = has_hybrid_v<opts>>
struct HybridSimState
{
};

template<auto opts>
struct HybridSimState<opts, true>
{
    using Model_t  = PHARE_Types<opts>::Hybrid::Model_t;
    using ResMan_t = Model_t::resources_manager_type;
    std::shared_ptr<ResMan_t> resman_;
    std::shared_ptr<Model_t> model_;
};

template<auto opts, bool enabled = has_mhd_v<opts>>
struct MHDSimState
{
};

template<auto opts>
struct MHDSimState<opts, true>
{
    using Model_t  = PHARE_Types<opts>::MHD::Model_t;
    using ResMan_t = Model_t::resources_manager_type;
    std::shared_ptr<ResMan_t> resman_;
    std::shared_ptr<Model_t> model_;
};

} // namespace PHARE::solver

#endif // PHARE_SOLVER_INCLUDE_HPP
