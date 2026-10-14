#ifndef PHARE_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_PRIMITIVE_CONSERVATIVE_CONVERTER_HPP


#include <tuple>

namespace PHARE::core
{
inline auto vToRhoV(auto const& rho, auto const& Vx, auto const& Vy, auto const& Vz)
{
    auto const rhoVx = rho * Vx;
    auto const rhoVy = rho * Vy;
    auto const rhoVz = rho * Vz;

    return std::make_tuple(rhoVx, rhoVy, rhoVz);
}

inline auto eosPToEtot(double const gamma, auto const& rho, auto const& vx, auto const& vy,
                       auto const& vz, auto const& bx, auto const& by, auto const& bz,
                       auto const& p)
{
    auto const v2 = vx * vx + vy * vy + vz * vz;
    auto const b2 = bx * bx + by * by + bz * bz;
    return p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2;
}


} // namespace PHARE::core

#endif
