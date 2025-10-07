#ifndef PHARE_SPLIT_3D_HPP
#define PHARE_SPLIT_3D_HPP

#include <array>
#include <cstddef>
#include "core/utilities/point/point.hpp"
#include "core/utilities/types.hpp"
#include "splitter.hpp"

namespace PHARE::amr
{
using namespace PHARE::core;

/**************************************************************************/
template<>
struct Splitter<DimConst<3>, InterpConst<1>, RefinedParticlesConst<0>>
    : public ASplitter<DimConst<3>, InterpConst<1>, RefinedParticlesConst<0>>
{
    constexpr Splitter() {}

    // mocking out patern dispatcher call op
    void operator()(auto const&, auto&, std::size_t idx = 0) const {}

    static constexpr std::array<float, 1> delta  = {0.};
    static constexpr std::array<float, 1> weight = {0.};
};


/**************************************************************************/
template<>
struct Splitter<DimConst<3>, InterpConst<2>, RefinedParticlesConst<0>>
    : public ASplitter<DimConst<3>, InterpConst<2>, RefinedParticlesConst<0>>
{
    constexpr Splitter() {}

    void operator()(auto const&, auto&, std::size_t idx = 0) const {}

    static constexpr std::array<float, 1> delta  = {0.};
    static constexpr std::array<float, 2> weight = {0., 0.};
};

/**************************************************************************/


} // namespace PHARE::amr


#endif /*PHARE_SPLIT_3D_H*/
