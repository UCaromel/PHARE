
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
#include "amr/physical_models/hybrid_model.hpp"
#include "amr/physical_models/physical_model.hpp"
#include "amr/data/particles/refine/splitter.hpp"
#include "amr/data/particles/refine/particles_data_split.hpp"
#include "amr/level_initializer/level_initializer_factory.hpp"
#include "amr/level_initializer/hybrid_level_initializer.hpp"
#include "amr/level_initializer/mhd_level_initializer.hpp"
#include "python3/mhd_resolver.hpp"

namespace PHARE::solver
{
// Bool-specialized holders: the `false` specialization is empty, so an opts value with
// hybrid (or mhd) disabled never names the corresponding Model_t/Solver_t/etc. as a type —
// that is what keeps the disabled model's templates from ever being instantiated.
template<auto opts, typename CoreTypes, bool enabled>
struct HybridStack;

template<auto opts, typename CoreTypes>
struct HybridStack<opts, CoreTypes, false>
{
};

template<auto opts, typename CoreTypes>
struct HybridStack<opts, CoreTypes, true>
{
    using GridLayout_t = typename CoreTypes::Hybrid::GridLayout_t;
    using Model_t       = HybridModel<GridLayout_t, typename CoreTypes::Hybrid::Electromag_t,
                                      typename CoreTypes::Hybrid::Ions_t,
                                      typename CoreTypes::Hybrid::Electrons_t, amr::SAMRAI_Types,
                                      typename CoreTypes::Hybrid::Grid_t>;
    using Solver_t = PHARE::solver::SolverPPC<Model_t, PHARE::amr::SAMRAI_Types>;

    using Splitter_t = PHARE::amr::Splitter<PHARE::core::DimConst<opts.dimension>,
                                            PHARE::core::InterpConst<opts.interp_order>,
                                            PHARE::core::RefinedParticlesConst<opts.nbRefinedPart>>;
    using RefinementParams_t
        = PHARE::amr::RefinementParams<typename CoreTypes::Hybrid::ParticleArray_t, Splitter_t>;

    using LevelInitializer_t = HybridLevelInitializer<Model_t>;
};

template<auto opts, typename CoreTypes, bool enabled>
struct MHDStack;

template<auto opts, typename CoreTypes>
struct MHDStack<opts, CoreTypes, false>
{
};

template<auto opts, typename CoreTypes>
struct MHDStack<opts, CoreTypes, true>
{
    using GridLayout_t = typename CoreTypes::MHD::GridLayout_t;
    using Model_t
        = MHDModel<GridLayout_t, typename CoreTypes::MHD::VecField_t, amr::SAMRAI_Types,
                  typename CoreTypes::MHD::Grid_t>;
    using Solver_t = PHARE::solver::SolverMHD<Model_t, PHARE::amr::SAMRAI_Types,
                                              typename MHDResolver<opts, Model_t>::MHDTimeStepper_t>;

    using LevelInitializer_t = MHDLevelInitializer<Model_t>;
};

// One specialization per enabled-combination, so a disabled model's types are never named as
// template arguments to MessengerFactory/LevelInitializerFactory.
template<auto opts, typename H, typename M, bool hasHybrid, bool hasMHD>
struct FactorySelector;

// hybrid-only
template<auto opts, typename H, typename M>
struct FactorySelector<opts, H, M, true, false>
{
    using Messenger_t = amr::MessengerFactory<
        typename H::Model_t, typename H::Model_t,
        amr::HybridHybridMessengerStrategy<typename H::Model_t, typename H::RefinementParams_t>>;
    using LevelInit_t = LevelInitializerFactory<amr::SAMRAI_Types, typename H::LevelInitializer_t>;
};

// mhd-only
template<auto opts, typename H, typename M>
struct FactorySelector<opts, H, M, false, true>
{
    using Messenger_t = amr::MessengerFactory<typename M::Model_t, typename M::Model_t,
                                              amr::MHDMessenger<typename M::Model_t>>;
    using LevelInit_t = LevelInitializerFactory<amr::SAMRAI_Types, typename M::LevelInitializer_t>;
};

// both
template<auto opts, typename H, typename M>
struct FactorySelector<opts, H, M, true, true>
{
    using Messenger_t = amr::MessengerFactory<
        typename M::Model_t, typename H::Model_t,
        amr::HybridHybridMessengerStrategy<typename H::Model_t, typename H::RefinementParams_t>,
        amr::MHDHybridMessengerStrategy<typename M::Model_t, typename H::Model_t>,
        amr::MHDMessenger<typename M::Model_t>>;
    using LevelInit_t
        = LevelInitializerFactory<amr::SAMRAI_Types, typename H::LevelInitializer_t,
                                  typename M::LevelInitializer_t>;
};

template<auto opts>
struct PHARE_Types
{
    static_assert(opts.mhd_axes_consistent());
    static_assert(is_hybrid_v<opts> || is_mhd_v<opts>, "a build must enable at least one model");

    auto static constexpr dimension = opts.dimension;

    // core deps
    using core_types = PHARE::core::PHARE_Types<opts>;

    using Hybrid = HybridStack<opts, core_types, is_hybrid_v<opts>>;
    using MHD    = MHDStack<opts, core_types, is_mhd_v<opts>>;

    using IPhysicalModel = PHARE::solver::IPhysicalModel<PHARE::amr::SAMRAI_Types>;

    // amr deps
    using amr_types = PHARE::amr::PHARE_Types<opts>;

    using Selector_t = FactorySelector<opts, Hybrid, MHD, is_hybrid_v<opts>, is_mhd_v<opts>>;

    using MessengerFactory // = amr/solver bidirectional dependency
        = typename Selector_t::Messenger_t;
    using LevelInitializerFactory_t = typename Selector_t::LevelInit_t;
    // amr deps

    using MultiPhysicsIntegrator_t
        = MultiPhysicsIntegrator<MessengerFactory, LevelInitializerFactory_t,
                                 PHARE::amr::SAMRAI_Types>;
};

} // namespace PHARE::solver

#endif // PHARE_SOLVER_INCLUDE_HPP
