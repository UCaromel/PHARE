#ifndef PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
#define PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP

#include "amr/solvers/solver.hpp"

namespace PHARE::solver {

template <typename MHDModel_>
class MHDModelView : public ISolverModelView {};

};  // namespace PHARE::solver

#endif  // PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
