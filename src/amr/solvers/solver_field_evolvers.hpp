#ifndef PHARE_AMR_SOLVERS_SOLVER_FIELD_EVOLVERS_HPP
#define PHARE_AMR_SOLVERS_SOLVER_FIELD_EVOLVERS_HPP

#include "core/numerics/ampere/ampere.hpp"
#include "core/numerics/faraday/faraday.hpp"

#include "amr/resources_manager/amr_utils.hpp"

#include <type_traits>


namespace PHARE::solver
{



template<typename Model>
class FaradayLevelTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::Faraday<GridLayout>;

public:
    explicit FaradayLevelTransformer(level_t& level, auto& model)
        : level_{level}
        , model_{model}
    {
    }

    // Internal dispatch with explicit layout parameter
    template<typename VecField>
    void operator()(GridLayout& layout, VecField& B, VecField& E, VecField& Bnew, double dt)
    {
        core_type{layout}(B, E, Bnew, dt);
    }

    // Public 4-arg interface: iterate through patches and dispatch to internal layout version
    void operator()(auto& B, auto& E, auto& Bnew, auto dt)
    {
        auto& rm = *model_.resourcesManager;
        for (auto& patch : rm.enumerate(level_, B, E, Bnew))
        {
            auto layout = amr::layoutFromPatch<GridLayout>(*patch);
            (*this)(layout, B, E, Bnew, dt);
        }
    }

    level_t& level_;
    Model& model_;
};
template<typename Model>
FaradayLevelTransformer(typename Model::amr_types::level_t&, Model&)
    -> FaradayLevelTransformer<Model>;




template<typename Model, core::AmpereBox box = core::AmpereBox{}>
class AmpereLevelTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::Ampere<GridLayout, box>;

public:
    explicit AmpereLevelTransformer(level_t& level, auto& model)
        : level_{level}
        , model_{model}
    {
    }

    void operator()(GridLayout& layout, auto&&... args) { core_type{layout}(args...); }

    void operator()(auto& B, auto& J)
    {
        auto& rm = *model_.resourcesManager;
        for (auto& patch : rm.enumerate(level_, B, J))
        {
            auto layout = amr::layoutFromPatch<GridLayout>(*patch);
            (*this)(layout, B, J);
        }
    }

    level_t& level_;
    Model& model_;
};


template<typename Model>
AmpereLevelTransformer(typename Model::amr_types::level_t&, Model&)
    -> AmpereLevelTransformer<Model>;




template<typename Model, core::AmpereBox box = core::AmpereBox{}>
class AmperePVLevelTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::AmperePV<GridLayout, box>;

public:
    explicit AmperePVLevelTransformer(level_t& level, auto& model)
        : level_{level}
        , model_{model}
    {
    }

    void operator()(GridLayout& layout, auto&&... args) { core_type{layout}(args...); }

    // point-value current: J = curl_4(Bavg) + curl_2(Bpv - Bavg) (see core::AmperePV).
    void operator()(auto& Bavg, auto& Bpv, auto& J)
    {
        auto& rm = *model_.resourcesManager;
        for (auto& patch : rm.enumerate(level_, Bavg, Bpv, J))
        {
            auto layout = amr::layoutFromPatch<GridLayout>(*patch);
            (*this)(layout, Bavg, Bpv, J);
        }
    }

    level_t& level_;
    Model& model_;
};






template<typename level_t, typename Model>
struct TimeSetter
{
    void operator()(auto&... quantities)
    {
        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, quantities...))
            (model.resourcesManager->setTime(quantities, *patch, newTime), ...);
    }

    level_t& level;
    Model& model;
    double newTime;
};

template<typename level_t, typename Model>
TimeSetter(level_t&, Model&, double) -> TimeSetter<level_t, Model>;



template<typename Model>
struct FieldEvolverDispatchers
{
    // hybrid: J is never communicated — the plain 2nd-order Ampere computes it on
    // physical+1, exactly the layer Ohm reads. MHD instead uses the point-value
    // AmperePV on ghostbox-2: the split-order curl (curl_4 on avg B with full
    // ghosts, curl_2 on the point-value correction valid on ghostbox-1) makes
    // ghostbox-2 exactly the widest J box computable from allocated B — see
    // core::AmperePV.
    static constexpr bool is_hybrid = Model::model_type_name == "HybridModel";

    using Faraday_t = FaradayLevelTransformer<Model>;
    using Ampere_t  = std::conditional_t<
        is_hybrid,
        AmpereLevelTransformer<Model, core::AmpereBox{core::AmpereMode::GrownPhysical, 1}>,
        AmperePVLevelTransformer<Model, core::AmpereBox{core::AmpereMode::ShrinkedGhost, 2}>>;
};


} // namespace PHARE::solver



#endif /* PHARE_AMR_SOLVERS_SOLVER_FIELD_EVOLVERS_HPP */
