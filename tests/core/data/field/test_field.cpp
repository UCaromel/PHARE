#include "core/data/field/field.hpp"
#include "core/data/field/field_traits.hpp"
#include "core/mhd/mhd_quantities.hpp"

namespace PHARE::core
{
static_assert(IsField<Field<3, MHDQuantity::Scalar>>);
}
