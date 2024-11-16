#ifndef PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
#define PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP

#include "amr/solvers/solver.hpp"
#include "amr/solvers/solver_ppc_model_view.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/constrained_transport/constrained_transport.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "core/numerics/finite_volume_euler/finite_volume_euler.hpp"

namespace PHARE::solver
{
template<typename GridLayout>
class GodunovFluxesTransformer
{
    using core_type = PHARE::core::GodunovFluxes<GridLayout>;

public:
    template<typename Layout, typename Field, typename VecField, typename... Fluxes>
    void operator()(Layout const& layouts, Field const& rho, VecField const& V,
                    VecField const& B_FV, Field const& P, VecField const& J, Fluxes&... fluxes)
    {
        assert_equal_sizes(rho, V, B_FV, P, J, fluxes...);
        for (std::size_t i = 0; i < layouts.size(); ++i)
        {
            auto _ = core::SetLayout(layouts[i], godunov_);
            godunov_(*rho[i], *V[i], *B_FV[i], *P[i], *J[i], *fluxes[i]...);
        }
    }

    core_type godunov_;
};

template<typename GridLayout>
class FiniteVolumeEulerTransformer
{
    using core_type = PHARE::core::FiniteVolumeEuler<GridLayout>;

public:
    template<typename Layout, typename Field, typename... Fluxes>
    void operator()(Layout const& layouts, Field const& U, Field& Unew, double const& dt,
                    const Fluxes&... fluxes)
    {
        assert_equal_sizes(U, Unew, fluxes...);
        for (std::size_t i = 0; i < layouts.size(); ++i)
        {
            auto _ = core::SetLayout(layouts[i], finite_volume_euler_);
            finite_volume_euler_(*U[i], *Unew[i], *fluxes[i]...);
        }
    }

    core_type finite_volume_euler_;
};

template<typename GridLayout>
class ConstrainedTransportTransformer
{
    using core_type = PHARE::core::ConstrainedTransport<GridLayout>;

public:
    template<typename Layout, typename VecField, typename... Fluxes>
    void operator()(Layout const& layouts, VecField& E, const Fluxes&... fluxes)
    {
        assert_equal_sizes(E, fluxes...);
        for (std::size_t i = 0; i < layouts.size(); ++i)
        {
            auto _ = core::SetLayout(layouts[i], constrained_transport_);
            finite_volume_euler_(*E[i], *fluxes[i]...);
        }
    }

    core_type constrained_transport_;
};


template<typename MHDModel_>
class MHDModelView : public ISolverModelView
{
public:
    using MHDModel_t          = MHDModel_;
    using GridLayout          = typename MHDModel_t::gridlayout_type;
    using GodunovFluxes_t     = GodunovFluxesTransformer<GridLayout>;
    using Ampere_t            = AmpereTransformer<GridLayout>;
    using FiniteVolumeEuler_t = FiniteVolumeEulerTransformer<GridLayout>;
    using Faraday_t           = FaradayTransformer<GridLayout>;
};

}; // namespace PHARE::solver

#endif // PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
