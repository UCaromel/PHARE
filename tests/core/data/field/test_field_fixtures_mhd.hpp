#ifndef PHARE_TEST_CORE_DATA_TEST_FIELD_MHD_HPP
#define PHARE_TEST_CORE_DATA_TEST_FIELD_MHD_HPP

#include "core/data/field/field.hpp"
#include "core/physical_quantities.hpp"

namespace PHARE::core
{

template<std::size_t dim>
using FieldMHD = Field<dim, PhysicalQuantity::Scalar, double>;

} // namespace PHARE::core


#endif /*PHARE_TEST_CORE_DATA_TEST_FIELD_FIXTURES_HPP*/
