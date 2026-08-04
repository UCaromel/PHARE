#ifndef PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_TO_PRIMITIVE_CONVERTER_HPP


#include "core/utilities/index/index.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/positivity_floors/positivity_floors.hpp"

namespace PHARE::core
{
static auto const min_value = std::sqrt(1024 * std::numeric_limits<double>::min());

auto rhoVToV(auto& rho, auto const& rhoVx, auto const& rhoVy, auto const& rhoVz)
{
    auto const vx = rhoVx / rho;
    auto const vy = rhoVy / rho;
    auto const vz = rhoVz / rho;

    return std::make_tuple(vx, vy, vz);
}

auto eosEtotToP(double const gamma, auto const& rho, auto const& vx, auto const& vy, auto const& vz,
                auto const& bx, auto const& by, auto const& bz, auto& etot)
{
    auto const v2 = vx * vx + vy * vy + vz * vz;
    auto const b2 = bx * bx + by * by + bz * bz;

    auto p = (gamma - 1.0) * (etot - 0.5 * rho * v2 - 0.5 * b2);
    // p      = (p < 0.) ? 1.0e-5 : p; //tbd maybe not needed
    // etot = p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2;

    return p;
}



template<typename GridLayout>
class ToPrimitiveConverter
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    ToPrimitiveConverter(GridLayout const& layout)
        : layout_{layout}
    {
    }

    template<typename Field, typename VecField>
    void operator()(double const gamma, Field& rho, VecField& rhoV, VecField const& B, Field& Etot,
                    VecField& V, Field& P, FloorParams const& floors = {}) const
    {
        rhoVToVOnGhostBox(rho, rhoV, V, floors);

        eosEtotToPOnGhostBox(gamma, rho, rhoV, B, Etot, P, floors);
    }

    // used for diagnostics (and, with an enabled FloorParams, for S1)
    template<typename Field, typename VecField>
    void rhoVToVOnGhostBox(Field& rho, VecField& rhoV, VecField& V,
                           FloorParams const& floors = {}) const
    {
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            rhoVToV_(rho, rhoV, V, {args...}, floors, FloorSite::ToPrimitive);
        });
    }

    // S4: post-reflux floor over the interior box only (no ghost cells).
    template<typename Field, typename VecField>
    void rhoVToVOnBox(Field& rho, VecField& rhoV, VecField& V, FloorParams const& floors,
                      FloorSite site) const
    {
        layout_.evalOnBox(
            rho, [&](auto&... args) mutable { rhoVToV_(rho, rhoV, V, {args...}, floors, site); });
    }

    // used for diagnostics (and, with an enabled FloorParams, for S1)
    template<typename Field, typename VecField>
    void eosEtotToPOnGhostBox(double const gamma, Field const& rho, VecField const& rhoV,
                              VecField const& B, Field& Etot, Field& P,
                              FloorParams const& floors = {}) const
    {
        layout_.evalOnGhostBox(rho, [&](auto&... args) mutable {
            eosEtotToP_(gamma, rho, rhoV, B, Etot, P, {args...}, floors, FloorSite::ToPrimitive);
        });
    }

    // S4: post-reflux floor over the interior box only (no ghost cells).
    template<typename Field, typename VecField>
    void eosEtotToPOnBox(double const gamma, Field const& rho, VecField const& rhoV,
                         VecField const& B, Field& Etot, Field& P, FloorParams const& floors,
                         FloorSite site) const
    {
        layout_.evalOnBox(rho, [&](auto&... args) mutable {
            eosEtotToP_(gamma, rho, rhoV, B, Etot, P, {args...}, floors, site);
        });
    }

private:
    template<typename Field, typename VecField>
    static void rhoVToV_(Field& rho, VecField& rhoV, VecField& V, MeshIndex<Field::dimension> index,
                         FloorParams const& floors, FloorSite site)
    {
        auto& rhoVx = rhoV(Component::X);
        auto& rhoVy = rhoV(Component::Y);
        auto& rhoVz = rhoV(Component::Z);

        auto& Vx = V(Component::X);
        auto& Vy = V(Component::Y);
        auto& Vz = V(Component::Z);

        // Computed from the raw (unfloored) rho/rhoV, so V reflects the true velocity.
        auto&& [x, y, z] = rhoVToV(rho(index), rhoVx(index), rhoVy(index), rhoVz(index));
        Vx(index)        = x;
        Vy(index)        = y;
        Vz(index)        = z;

        // On a rho floor, preserve V and rescale rhoV = rho_floor * V (not the other way
        // around) — preserving rhoV at a floored rho would blow up |V| = rhoV/rho_floor.
        if (floorScalarInPlace(rho(index), rho(index), floors.density_floor, site, true, floors))
        {
            rhoVx(index) = rho(index) * x;
            rhoVy(index) = rho(index) * y;
            rhoVz(index) = rho(index) * z;
        }
    }

    template<typename Field, typename VecField>
    static void eosEtotToP_(double const gamma, Field const& rho, VecField const& rhoV,
                            VecField const& B, Field& Etot, Field& P,
                            MeshIndex<Field::dimension> index, FloorParams const& floors,
                            FloorSite site)
    {
        auto const& rhoVx = rhoV(Component::X);
        auto const& rhoVy = rhoV(Component::Y);
        auto const& rhoVz = rhoV(Component::Z);

        auto const& Bx = B(Component::X);
        auto const& By = B(Component::Y);
        auto const& Bz = B(Component::Z);

        auto const vx = rhoVx(index) / rho(index);
        auto const vy = rhoVy(index) / rho(index);
        auto const vz = rhoVz(index) / rho(index);
        auto const bx
            = GridLayout::template project<GridLayout::implT::faceXToCellCenter>(Bx, index);
        auto const by
            = GridLayout::template project<GridLayout::implT::faceYToCellCenter>(By, index);
        auto const bz
            = GridLayout::template project<GridLayout::implT::faceZToCellCenter>(Bz, index);
        P(index) = eosEtotToP(gamma, rho(index), vx, vy, vz, bx, by, bz, Etot(index));

        floorScalarInPlace(P(index), Etot(index), floors.pressure_floor, site, false, floors);
    }


private:
    GridLayout layout_;
};

} // namespace PHARE::core

#endif
