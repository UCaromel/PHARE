#ifndef PHARE_AMR_SOLVERS_TIME_INTEGRATOR_POINT_VALUE_APPROXIMATION_HPP
#define PHARE_AMR_SOLVERS_TIME_INTEGRATOR_POINT_VALUE_APPROXIMATION_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"
#include "core/numerics/point_values_handler/point_value_state.hpp"
#include "core/numerics/point_values_handler/troubled_cell_detector.hpp"

#include "initializer/data_provider.hpp"

#include <tuple>

namespace PHARE::solver
{
template<typename MHDModel>
class SecondOrderPointValueApproximation
{
    using Dispatchers_t = Dispatchers<MHDModel>;

public:
    explicit SecondOrderPointValueApproximation(PHARE::initializer::PHAREDict const& /*dict*/) {}

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::tuple<>{}; }
    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::tuple<>{}; }

    void fillMessengerInfo(auto& /*info*/) const {}

    auto& averageStateToPointValues(MHDModel& model, auto& state, auto& /*bc*/,
                                    typename MHDModel::level_t& level, double const time,
                                    double const gamma, bool const computeCurrent)
    {
        typename Dispatchers_t::ToPrimitiveConverter_t{level, model}(state, gamma);
        TimeSetter{level, model, time}(state.rho, state.V, state.P);

        if (computeCurrent)
        {
            typename Dispatchers_t::Ampere_t{level, model}(state.B, state.J);
            TimeSetter{level, model, time}(state.B, state.J);
        }
        return state;
    }

    void pointValueFluxesToAverages(MHDModel&, auto&, auto&, typename MHDModel::level_t&,
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
    explicit FourthOrderPointValueApproximation(PHARE::initializer::PHAREDict const& dict)
        : limiter_{core::ConversionLimiterInfo::FROM(dict)}
    {
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(pointValues_);
    }

    // the messenger fills the ghosts of the troubled cells only if they are published here
    void fillMessengerInfo(auto& info) const
    {
        if (limiter_.enabled())
            info.troubledCells = pointValues_.troubled.name();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(pointValues_);
    }

    // The ghosts of state are filled. With the limiter, each patch flags its own troubled cells
    // and gets the flags of its ghost cells from the patches that own them, so that two patches
    // convert a shared cell or face the same way.
    auto& averageStateToPointValues(MHDModel& model, auto& state, auto& bc,
                                    typename MHDModel::level_t& level, double const time,
                                    double const gamma, bool const computeCurrent)
    {
        typename Dispatchers_t::PointValue_t pointValue{level, model, limiter_};

        if (limiter_.enabled())
        {
            pointValue.detectTroubledCells(pointValues_, state, gamma);
            bc.fillTroubledCellGhosts(pointValues_.troubled, level, time);
        }

        pointValue.toPointValues(pointValues_, state, time);

        if (computeCurrent)
        {
            typename Dispatchers_t::AmperePV_t{level, model}(state.B, pointValues_.B,
                                                             pointValues_.J);
            TimeSetter{level, model, time}(pointValues_.J);
        }

        typename Dispatchers_t::ToPrimitiveConverter_t{level, model}.pointValuesOnShrinkedGhostBox(
            pointValues_, gamma, core::point_value_conversion_shrink);
        TimeSetter{level, model, time}(pointValues_.V, pointValues_.P);
        return pointValues_;
    }

    void pointValueFluxesToAverages(MHDModel& model, auto& fluxes, auto& E,
                                    typename MHDModel::level_t& level, double const time)
    {
        typename Dispatchers_t::PointValue_t{level, model, limiter_}.fluxesToAverages(
            pointValues_, fluxes, E);
        TimeSetter{level, model, time}(E);
    }

private:
    core::PointValueState<VecField> pointValues_{};
    core::ConversionLimiterInfo limiter_;
};

} // namespace PHARE::solver

#endif
