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
    using Super = MHDState<VecFieldMHD<dim>>;

    void _set()
    {
        auto&& [_rho, _V, _B, _P, _rhoV, _Etot, _J, _E] = Super::getCompileTimeResourcesViewList();
        _rho.setBuffer(&rho);
        V.set_on(_V);
        B.set_on(_B);
        _P.setBuffer(&P);
        rhoV.set_on(_rhoV);
        _Etot.setBuffer(&Etot);
        J.set_on(_J);
        E.set_on(_E);
    }

public:
    using Array_t = NdArrayVector<dim, double, /*c_ordering*/ true>;
    using Grid_t  = Grid<Array_t, MHDQuantity::Scalar>;

    template<typename GridLayout>
    UsableMHDState(GridLayout const& layout)
        : Super{getDict()}
        , rho{"rho", layout, MHDQuantity::Scalar::rho}
        , V{"V", layout, MHDQuantity::Vector::V}
        , B{"B", layout, MHDQuantity::Vector::B}
        , P{"P", layout, MHDQuantity::Scalar::P}
        , rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV}
        , Etot{"Etot", layout, MHDQuantity::Scalar::Etot}
        , J{"J", layout, MHDQuantity::Vector::J}
        , E{"E", layout, MHDQuantity::Vector::E}
    {
        _set();
    }

    UsableMHDState(UsableMHDState const&) = delete;

    UsableMHDState(UsableMHDState&& that)
        : Super{std::forward<Super>(that)}
        , rho{std::move(that.rho)}
        , V{std::move(that.V)}
        , B{std::move(that.B)}
        , P{std::move(that.P)}
        , rhoV{std::move(that.rhoV)}
        , Etot{std::move(that.Etot)}
        , J{std::move(that.J)}
        , E{std::move(that.E)}
    {
        _set();
    }

    Super& super() { return *this; }
    Super const& super() const { return *this; }
    auto& operator*() { return super(); }
    auto& operator*() const { return super(); }

    Grid_t rho, P, Etot;
    UsableVecFieldMHD<dim> V, B, rhoV, J, E;
};

} // namespace PHARE::core

#endif
