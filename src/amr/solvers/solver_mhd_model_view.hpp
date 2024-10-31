#ifndef PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
#define PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP

#include "amr/solvers/solver.hpp"
#include "amr/solvers/solver_ppc_model_view.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"

namespace PHARE::solver
{
template<typename GridLayout>
class GodunovFluxesTransformer
{
    using core_type = PHARE::core::GodunovFluxes<GridLayout>;

public:
    template<typename Layout, typename Field, typename VecField, typename... Fluxes>
    void operator()(Layout const& layouts, Field const& rho, VecField const& V,
                    VecField const& B_CT, Field const& P, VecField const& J, Fluxes&... fluxes)
    {
        assert_equal_sizes(rho, V, B_CT, P, J, fluxes...);
        for (std::size_t i = 0; i < layouts.size(); ++i)
        {
            auto _ = core::SetLayout(layouts[i], godunov_);
            godunov_(*rho[i], *V[i], *B_CT[i], *P[i], *J[i], *fluxes[i]...);
        }
    }

    core_type godunov_;
};


template<typename MHDModel_>
class MHDModelView : public ISolverModelView
{
public:
    using MHDModel_t      = MHDModel_;
    using GridLayout      = typename MHDModel_t::gridlayout_type;
    using GodunovFluxes_t = GodunovFluxesTransformer<GridLayout>;
    using Ampere_t        = AmpereTransformer<GridLayout>;
};

}; // namespace PHARE::solver

#endif // PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
