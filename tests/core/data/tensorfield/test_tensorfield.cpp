#include "core/data/tensorfield/tensorfield.hpp"
#include "core/data/tensorfield/tensorfield_traits.hpp"
#include "core/mhd/mhd_quantities.hpp"

#include "simulator/phare_types.hpp"

#include "gtest/gtest.h"

namespace PHARE::amr
{
constexpr SimOpts opts;
constexpr std::size_t rank = 1;
using Types                = PHARE_Types<opts>::core_types;
using Grid                 = Types::Grid_t;
using Field                = Grid::field_type;
using GridLayout           = Types::GridLayout_t;
using PhysicalQuantity     = MHDQuantity;

static_assert(IsTensorField<TensorField<Field, PhysicalQuantity, rank>>);
} // namespace PHARE::amr

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
