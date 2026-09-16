#ifndef PHARE_AMR_SOLVERS_TIME_INTEGRATOR_POINT_VALUE_APPROXIMATION_HPP
#define PHARE_AMR_SOLVERS_TIME_INTEGRATOR_POINT_VALUE_APPROXIMATION_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"
#include "core/numerics/point_values_handler/point_value_state.hpp"

#include <tuple>

namespace PHARE::solver
{
template<typename MHDModel>
class SecondOrderPointValueApproximation
{
    using Dispatchers_t = Dispatchers<MHDModel>;

public:
    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::tuple<>{}; }
    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::tuple<>{}; }

    template<bool ComputeCurrent>
    auto& averageStateToPointValues(MHDModel& model, auto& state, typename MHDModel::level_t& level,
                                    double const time, double const gamma)
    {
        typename Dispatchers_t::ToPrimitiveConverter_t{level, model}(state, gamma);
        TimeSetter{level, model, time}(state.rho, state.V, state.P);

        if constexpr (ComputeCurrent)
        {
            typename Dispatchers_t::Ampere_t{level, model}(state.B, state.J);
            TimeSetter{level, model, time}(state.B, state.J);
        }
        return state;
    }

    void pointValueFluxesToAverages(MHDModel&, auto&, auto&, auto&, typename MHDModel::level_t&,
                                    double) const
    {
    }
};


template<typename MHDModel>
class FourthOrderPointValueApproximation
{
    using Dispatchers_t = Dispatchers<MHDModel>;
    using VecField      = typename MHDModel::vecfield_type;

public:
    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(pointValues_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(pointValues_);
    }

    template<bool ComputeCurrent>
    auto& averageStateToPointValues(MHDModel& model, auto& state, typename MHDModel::level_t& level,
                                    double const time, double const gamma)
    {
        typename Dispatchers_t::ToPointValue_t{level, model}(pointValues_, state, time);

        if constexpr (ComputeCurrent)
        {
            typename Dispatchers_t::AmperePV_t{level, model}(state.B, pointValues_.B,
                                                             pointValues_.J);
            TimeSetter{level, model, time}(pointValues_.J);
        }

        typename Dispatchers_t::ToPrimitiveConverter_t{level, model}.onShrinkedGhostBox(
            pointValues_, gamma, core::point_value_conversion_shrink);
        TimeSetter{level, model, time}(pointValues_.V, pointValues_.P);
        return pointValues_;
    }

    void pointValueFluxesToAverages(MHDModel& model, auto& pointValues, auto& fluxes, auto& E,
                                    typename MHDModel::level_t& level, double const time) const
    {
        typename Dispatchers_t::ToPointValue_t{level, model}.pointValueFluxesToAverages(pointValues,
                                                                                        fluxes, E);
        TimeSetter{level, model, time}(E);
    }

private:
    core::PointValueState<VecField> pointValues_{};
};

} // namespace PHARE::solver

#endif
