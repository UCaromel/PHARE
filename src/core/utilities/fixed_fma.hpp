#ifndef PHARE_CORE_UTILITIES_FIXED_FMA_HPP
#define PHARE_CORE_UTILITIES_FIXED_FMA_HPP

#include <cmath>

namespace PHARE::core
{
// A value whose products are rounded by one fixed rule instead of being left to the compiler.
// With gcc's default -ffp-contract=fast, `a*b + c*d` may fuse either product into an FMA, and gcc
// picks differently in a vectorized loop and its scalar remainder, so two copies of a node shared
// between patches differ by 1 ulp (#1611). Wrap the factors with fixed_fma() and write the formula
// as usual: in a sum of two products, the right one is fused and the left one rounded first.
template<typename T>
struct FixedFma
{
    T v;
};

// a*b, not yet rounded. Deliberately not convertible to T: mixed with a plain T, the compiler
// would be free to fuse it again.
template<typename T>
struct FixedProduct
{
    T a, b;
};

template<typename T>
auto fixed_fma(T const v)
{
    return FixedFma<T>{v};
}

template<typename T>
auto operator-(FixedFma<T> const x)
{
    return FixedFma<T>{-x.v};
}

template<typename T>
auto operator*(FixedFma<T> const x, FixedFma<T> const y)
{
    return FixedProduct<T>{x.v, y.v};
}

template<typename T>
T operator+(FixedProduct<T> const l, FixedProduct<T> const r)
{
#if defined(__FP_FAST_FMA) || defined(FP_FAST_FMA) || defined(__FMA__)
    return std::fma(r.a, r.b, l.a * l.b);
#else
    // no FMA hardware: nothing to fuse, and std::fma would be a slow libm call
    return l.a * l.b + r.a * r.b;
#endif
}

} // namespace PHARE::core

#endif
