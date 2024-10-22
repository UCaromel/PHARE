#ifndef PHARE_TEST_CORE_DATA_MHDSTATE_MHDSTATE_FIXTURES_HPP
#define PHARE_TEST_CORE_DATA_MHDSTATE_MHDSTATE_FIXTURES_HPP

#include "core/mhd/mhd_quantities.hpp"
#include "core/models/mhd_state.hpp"
#include "initializer/data_provider.hpp"
#include "phare_core.hpp"
#include "tests/core/data/field/test_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"
#include "tests/core/numerics/mhd_solver/init_functions.hpp"

namespace PHARE::core
{
using namespace PHARE::initializer;
using namespace PHARE::initializer::test_fn::func_1d;

inline PHAREDict getDict()
{
    using initfunc = InitFunction<1>;
    PHAREDict dict;

    dict["density"]["initializer"] = static_cast<initfunc>(density);

    dict["velocity"]["initializer"]["x_component"] = static_cast<initfunc>(vx);
    dict["velocity"]["initializer"]["y_component"] = static_cast<initfunc>(vy);
    dict["velocity"]["initializer"]["z_component"] = static_cast<initfunc>(vz);

    dict["magnetic"]["initializer"]["x_component"] = static_cast<initfunc>(bx);
    dict["magnetic"]["initializer"]["y_component"] = static_cast<initfunc>(by);
    dict["magnetic"]["initializer"]["z_component"] = static_cast<initfunc>(bz);

    dict["pressure"]["initializer"] = static_cast<initfunc>(pressure);

    return dict;
}

template<std::size_t dim>
class UsableMHDState : public MHDState<VecFieldMHD<dim>>
{
    using Array_t = NdArrayVector<dim, double, /*c_ordering*/ true>;
    using Grid_t  = Grid<Array_t, MHDQuantity::Scalar>;
    using Super   = MHDState<VecFieldMHD<dim>>;

    void _set()
    {
        auto&& [_rho, _V, _B_FV, _P, _M, _Etot, _B_CT, _J, _E, _B_RSx, _B_RSy, _B_RSz]
            = Super::getCompileTimeResourcesViewList();
        _rho.setBuffer(&rho);
        V.set_on(_V);
        B_CT.set_on(_B_CT);
        _P.setBuffer(&P);
    }

public:
    template<typename GridLayout>
    UsableMHDState(GridLayout const& layout)
        : Super{getDict()}
        , rho{"rho", layout, MHDQuantity::Scalar::rho}
        , V{"V", layout, MHDQuantity::Vector::V}
        , B_CT{"B_FV", layout, MHDQuantity::Vector::B_CT}
        , P{"P", layout, MHDQuantity::Scalar::P}
    {
        _set();
    }

    UsableMHDState(UsableMHDState const&) = delete;

    UsableMHDState(UsableMHDState&& that)
        : Super{std::forward<Super>(that)}
        , rho{std::move(that.rho)}
        , V{std::move(that.V)}
        , B_CT{std::move(that.B_CT)}
        , P{std::move(that.P)}
    {
        _set();
    }

    Super& super() { return *this; }
    Super const& super() const { return *this; }
    auto& operator*() { return super(); }
    auto& operator*() const { return super(); }

    Grid_t rho, P;
    UsableVecFieldMHD<dim> V, B_CT;
};

} // namespace PHARE::core

#endif
