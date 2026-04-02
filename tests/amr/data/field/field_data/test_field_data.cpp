#include "amr/data/field/field_data.hpp"
#include "amr/data/field/field_data_traits.hpp"

#include "simulator/phare_types.hpp"

namespace PHARE::amr
{
constexpr SimOpts opts;
using Types      = PHARE_Types<opts>::core_types;
using Grid       = Types::Grid_t;
using GridLayout = Types::GridLayout_t;

static_assert(IsFieldData<FieldData<GridLayout, Grid>>);
} // namespace PHARE::amr
