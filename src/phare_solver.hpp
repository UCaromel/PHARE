
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
#include "amr/level_initializer/level_initializer_factory.hpp"
#include <python3/mhd_resolver.hpp>

namespace PHARE::solver
{
template<auto opts, typename HybridModel, typename MHDModel, typename ParticleArray, bool hasHybrid>
struct MessengerFactorySelector;

template<auto opts, typename HybridModel, typename MHDModel, typename ParticleArray>
struct MessengerFactorySelector<opts, HybridModel, MHDModel, ParticleArray, true>
{
    using Splitter_t = PHARE::amr::Splitter<PHARE::core::DimConst<opts.dimension>,
                                            PHARE::core::InterpConst<opts.interp_order>,
                                            PHARE::core::RefinedParticlesConst<opts.nbRefinedPart>>;

    using RefinementParams_t = PHARE::amr::RefinementParams<ParticleArray, Splitter_t>;

    using type = PHARE::amr::MessengerFactory<
        MHDModel,
        HybridModel,
        PHARE::amr::HybridHybridMessengerStrategy<HybridModel, RefinementParams_t>,
        PHARE::amr::MHDHybridMessengerStrategy<MHDModel, HybridModel>,
        PHARE::amr::MHDMessenger<MHDModel>>;
};

template<auto opts, typename HybridModel, typename MHDModel, typename ParticleArray>
struct MessengerFactorySelector<opts, HybridModel, MHDModel, ParticleArray, false>
{
    using type = PHARE::amr::MessengerFactory<MHDModel,
                                              HybridModel,
                                              PHARE::amr::MHDMessenger<MHDModel>>;
};

template<auto opts>
struct PHARE_Types
{
    auto static constexpr dimension     = opts.dimension;
    auto static constexpr interp_order  = opts.interp_order;
    auto static constexpr nbRefinedPart = opts.nbRefinedPart;

    // core deps
    using core_types = PHARE::core::PHARE_Types<opts>;

    // Hybrid
    using VecField_t   = typename core_types::VecField_t;
    using Grid_t       = typename core_types::Grid_t;
    using Electromag_t = typename core_types::Electromag_t;
    using Ions_t       = typename core_types::Ions_t;
    using GridLayout_t = typename core_types::GridLayout_t;
    using Electrons_t  = typename core_types::Electrons_t;

    // MHD
    using Grid_MHD       = typename core_types::Grid_MHD;
    using VecField_MHD   = typename core_types::VecField_MHD;
    using GridLayout_MHD = typename core_types::GridLayout_MHD;
    // core deps

    using IPhysicalModel = PHARE::solver::IPhysicalModel<PHARE::amr::SAMRAI_Types>;
    using HybridModel_t  = PHARE::solver::HybridModel<GridLayout_t, Electromag_t, Ions_t,
                                                      Electrons_t, PHARE::amr::SAMRAI_Types, Grid_t>;
    using MHDModel_t
        = PHARE::solver::MHDModel<GridLayout_MHD, VecField_MHD, PHARE::amr::SAMRAI_Types, Grid_MHD>;

    using SolverPPC_t = PHARE::solver::SolverPPC<HybridModel_t, PHARE::amr::SAMRAI_Types>;
    using SolverMHD_t
        = PHARE::solver::SolverMHD<MHDModel_t, PHARE::amr::SAMRAI_Types,
                                   typename MHDResolver<opts, MHDModel_t>::MHDTimeStepper_t>;

    using LevelInitializerFactory_t
        = PHARE::solver::LevelInitializerFactory<HybridModel_t, MHDModel_t>;

    using MessengerFactory // = amr/solver bidirectional dependency
        = typename MessengerFactorySelector<opts,
                                            HybridModel_t,
                                            MHDModel_t,
                                            typename core_types::ParticleArray_t,
                                            opts.has_hybrid_model()>::type;
    // amr deps

    using MultiPhysicsIntegrator_t
        = MultiPhysicsIntegrator<MessengerFactory, LevelInitializerFactory_t,
                                 PHARE::amr::SAMRAI_Types>;
};

} // namespace PHARE::solver

#endif // PHARE_SOLVER_INCLUDE_HPP
