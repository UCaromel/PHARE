#ifndef PHARE_TEST_INITIALIZER_INIT_FUNCTIONS_HPP
#define PHARE_TEST_INITIALIZER_INIT_FUNCTIONS_HPP

#include <memory>
#include <vector>

#include "core/utilities/span.hpp"
#include "core/utilities/types.hpp"
#include "initializer/data_provider.hpp"

namespace PHARE::initializer::test_fn::func_1d
{
using Param  = std::vector<double> const&;
using Return = std::shared_ptr<PHARE::core::Span<double>>;

Return density(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vx(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vy(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vz(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vthx(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vthy(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vthz(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return bx(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return by(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return bz(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return pressure(Param x)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

} // namespace PHARE::initializer::test_fn::func_1d

namespace PHARE::initializer::test_fn::func_2d
{
using Param  = std::vector<double> const&;
using Return = std::shared_ptr<PHARE::core::Span<double>>;

Return density(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vx(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vy(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vz(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vthx(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vthy(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return vthz(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return bx(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return by(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return bz(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

Return pressure(Param x, Param /*y*/)
{
    return std::make_shared<core::VectorSpan<double>>(x);
}

} // namespace PHARE::initializer::test_fn::func_2d

template<std::size_t dim>
auto makeSharedPtr()
{
    using Param = std::vector<double> const&;

    if constexpr (dim == 1)
    {
        return [](Param x) { return std::make_shared<PHARE::core::VectorSpan<double>>(x); };
    }
    else if constexpr (dim == 2)
    {
        return [](Param x, Param /*y*/) {
            return std::make_shared<PHARE::core::VectorSpan<double>>(x);
        };
    }
    else if constexpr (dim == 3)
    {
        return [](Param x, Param /*y*/, Param /*z*/) {
            return std::make_shared<PHARE::core::VectorSpan<double>>(x);
        };
    }
}

// Conserved-composition helpers, mirroring the Python MHDModel composition.
// They build a new InitFunction<dim> that evaluates the primitive init functions
// at the same coordinates and combines the resulting spans element-wise.

// Element-wise product of two scalar init functions: rho * v
template<std::size_t dim>
PHARE::initializer::InitFunction<dim> mulInit(PHARE::initializer::InitFunction<dim> a,
                                              PHARE::initializer::InitFunction<dim> b)
{
    return [a = std::move(a),
            b = std::move(b)](auto const&... coords) -> std::shared_ptr<PHARE::core::Span<double>> {
        auto sa = a(coords...);
        auto sb = b(coords...);
        std::vector<double> vals(sa->size());
        for (std::size_t i = 0; i < vals.size(); ++i)
            vals[i] = (*sa)[i] * (*sb)[i];
        return std::make_shared<PHARE::core::VectorSpan<double>>(std::move(vals));
    };
}

// EOS total energy from primitive init functions:
//   Etot = p/(gamma-1) + 0.5*rho*(vx^2+vy^2+vz^2) + 0.5*(bx^2+by^2+bz^2)
template<std::size_t dim>
PHARE::initializer::InitFunction<dim>
etotInit(double gamma, PHARE::initializer::InitFunction<dim> rho,
         PHARE::initializer::InitFunction<dim> vx, PHARE::initializer::InitFunction<dim> vy,
         PHARE::initializer::InitFunction<dim> vz, PHARE::initializer::InitFunction<dim> bx,
         PHARE::initializer::InitFunction<dim> by, PHARE::initializer::InitFunction<dim> bz,
         PHARE::initializer::InitFunction<dim> p)
{
    return [=](auto const&... coords) -> std::shared_ptr<PHARE::core::Span<double>> {
        auto srho = rho(coords...);
        auto svx  = vx(coords...);
        auto svy  = vy(coords...);
        auto svz  = vz(coords...);
        auto sbx  = bx(coords...);
        auto sby  = by(coords...);
        auto sbz  = bz(coords...);
        auto sp   = p(coords...);
        std::vector<double> vals(srho->size());
        for (std::size_t i = 0; i < vals.size(); ++i)
            vals[i]
                = (*sp)[i] / (gamma - 1.0)
                  + 0.5 * (*srho)[i]
                        * ((*svx)[i] * (*svx)[i] + (*svy)[i] * (*svy)[i] + (*svz)[i] * (*svz)[i])
                  + 0.5 * ((*sbx)[i] * (*sbx)[i] + (*sby)[i] * (*sby)[i] + (*sbz)[i] * (*sbz)[i]);
        return std::make_shared<PHARE::core::VectorSpan<double>>(std::move(vals));
    };
}

#endif // PHARE_TEST_INITIALIZER_INIT_FUNCTIONS_HPP
