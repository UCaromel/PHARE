#ifndef PHARE_CORE_NUMERICS_PRIMITE_CONSERVATIVE_CONVERTER_FACE_ENERGY_HPP
#define PHARE_CORE_NUMERICS_PRIMITE_CONSERVATIVE_CONVERTER_FACE_ENERGY_HPP

#include "core/data/vecfield/vecfield_component.hpp"

#include <cstdlib>

namespace PHARE::core
{
// experiment PHARE_P_FACE_ENERGY=1: the magnetic energy used in every Etot <-> p conversion is
// 1/2 sum_d (average over the cell's two d-faces of B_d^2) ("average of squares") instead of
// 1/2 |average of B|^2. With B uniform per cell both are bitwise equal.
inline bool useFaceEnergy()
{
    static bool const v = std::getenv("PHARE_P_FACE_ENERGY") != nullptr;
    return v;
}

template<typename F>
struct SquaredField
{
    using type = double;
    F const& f;
    double operator()(auto const& i) const
    {
        double const v = f(i);
        return v * v;
    }
};

// sum_d of the face->cell projection of B_d^2 (same stencil as the B_cc projection)
template<typename GridLayout>
double faceMagneticSquare(auto const& B, auto const& index)
{
    using implT  = GridLayout::implT;
    using Bx_t   = std::decay_t<decltype(B(Component::X))>;
    double const sx = GridLayout::template project<implT::faceXToCellCenter>(
        SquaredField<Bx_t>{B(Component::X)}, index);
    double const sy = GridLayout::template project<implT::faceYToCellCenter>(
        SquaredField<Bx_t>{B(Component::Y)}, index);
    double const sz = GridLayout::template project<implT::faceZToCellCenter>(
        SquaredField<Bx_t>{B(Component::Z)}, index);
    return sx + sy + sz;
}
} // namespace PHARE::core

#endif
