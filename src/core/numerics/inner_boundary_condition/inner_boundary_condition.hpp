#ifndef PHARE_CORE_INNER_BOUNDARY_CONDITION_INNER_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_INNER_BOUNDARY_CONDITION_INNER_BOUNDARY_CONDITION_HPP


#include "core/inner_boundary/inner_boundary.hpp"

namespace PHARE::core
{

template<size_t dim>
class InnerBoundaryCondition
{
public:
    using inner_boundary_type = InnerBoundary<dim>;

    InnerBoundaryCondition() = default;

    virtual ~InnerBoundaryCondition() = default;

private:
};

} // namespace PHARE::core

#endif // PHARE_CORE_INNER_BOUNDARY_CONDITION_INNER_BOUNDARY_CONDITION_HPP
