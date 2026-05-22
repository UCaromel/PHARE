#ifndef PHARE_SPAWN_MAXWELLIAN_FROM_MHD_HPP
#define PHARE_SPAWN_MAXWELLIAN_FROM_MHD_HPP

#include "core/data/ions/particle_initializers/maxwellian_particle_initializer.hpp"
#include "core/utilities/span.hpp"
#include "initializer/data_provider.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace PHARE::amr
{

// Build Maxwellian-distributed particles from MHD conservative fields and append to `out`.
//
// CoordToLocal: callable mapping (x, y, z) → coarse local index (Point-like with [0]/[1]/[2]).
// Always invoked with three doubles; unused coordinates are 0.0. Required to range over
// the layout the MaxwellianInitializer samples on.
//
// Field references are evaluated at the coarse local index. B is face-centered
// (Bx Pdd, By dPd, Bz ddP) and averaged to cell centers; rho/rhoV/Etot are cell-centered.
template<typename ParticleArray, typename Layout, typename ScalarField, typename VecFieldArray,
         typename CoordToLocal>
void spawnMaxwellianFromMHD(Layout const& sampleLayout, ScalarField const& rho,
                            VecFieldArray const& rhoV, VecFieldArray const& B,
                            ScalarField const& Etot, CoordToLocal const& coordToCoarseLocal,
                            double const gamma, double const charge, std::uint32_t const nbrPPC,
                            std::optional<std::size_t> const seed, ParticleArray& out)
{
    static constexpr std::size_t dim = Layout::dimension;
    using InputFunction              = PHARE::initializer::InitFunction<dim>;
    using MaxwellianInit             = core::MaxwellianParticleInitializer<ParticleArray, Layout>;

    auto fieldAt = [&](auto const& f, double x, double y, double z) -> double {
        auto lc = coordToCoarseLocal(x, y, z);
        if constexpr (dim == 1)
            return f(lc[0]);
        else if constexpr (dim == 2)
            return f(lc[0], lc[1]);
        else
            return f(lc[0], lc[1], lc[2]);
    };

    auto bCellCentered = [&](std::size_t comp, double x, double y, double z) -> double {
        auto lc = coordToCoarseLocal(x, y, z);
        if constexpr (dim == 1)
        {
            if (comp == 0)
                return 0.5 * (B[0](lc[0]) + B[0](lc[0] + 1));
            return B[comp](lc[0]); // By, Bz are ddd in 1D
        }
        else if constexpr (dim == 2)
        {
            if (comp == 0)
                return 0.5 * (B[0](lc[0], lc[1]) + B[0](lc[0] + 1, lc[1]));
            if (comp == 1)
                return 0.5 * (B[1](lc[0], lc[1]) + B[1](lc[0], lc[1] + 1));
            return B[2](lc[0], lc[1]); // Bz ddd in 2D
        }
        else
        {
            if (comp == 0)
                return 0.5
                       * (B[0](lc[0], lc[1], lc[2]) + B[0](lc[0] + 1, lc[1], lc[2]));
            if (comp == 1)
                return 0.5
                       * (B[1](lc[0], lc[1], lc[2]) + B[1](lc[0], lc[1] + 1, lc[2]));
            return 0.5 * (B[2](lc[0], lc[1], lc[2]) + B[2](lc[0], lc[1], lc[2] + 1));
        }
    };

    auto vthAt = [&](double x, double y, double z) -> double {
        double const rho_k = fieldAt(rho, x, y, z);
        if (rho_k <= 0.0)
            return 0.0;
        double const rhoVx_k = fieldAt(rhoV[0], x, y, z);
        double const rhoVy_k = fieldAt(rhoV[1], x, y, z);
        double const rhoVz_k = fieldAt(rhoV[2], x, y, z);
        double const Vx_k    = rhoVx_k / rho_k;
        double const Vy_k    = rhoVy_k / rho_k;
        double const Vz_k    = rhoVz_k / rho_k;
        double const bx_k    = bCellCentered(0, x, y, z);
        double const by_k    = bCellCentered(1, x, y, z);
        double const bz_k    = bCellCentered(2, x, y, z);
        double const Etot_k  = fieldAt(Etot, x, y, z);
        double const KE_k    = 0.5 * rho_k * (Vx_k * Vx_k + Vy_k * Vy_k + Vz_k * Vz_k);
        double const ME_k    = 0.5 * (bx_k * bx_k + by_k * by_k + bz_k * bz_k);
        double const P_k     = (gamma - 1.0) * (Etot_k - KE_k - ME_k);
        return std::sqrt(std::max(P_k, 0.0) / rho_k);
    };

    InputFunction densityFn;
    std::array<InputFunction, 3> velocityFn;
    std::array<InputFunction, 3> thermalFn;

    if constexpr (dim == 1)
    {
        densityFn = [&](std::vector<double> const& xs) -> std::shared_ptr<core::Span<double>> {
            auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
            for (std::size_t k = 0; k < xs.size(); ++k)
                (*r)[k] = fieldAt(rho, xs[k], 0.0, 0.0);
            return r;
        };
        for (std::size_t comp = 0; comp < 3; ++comp)
            velocityFn[comp]
                = [&, comp](std::vector<double> const& xs) -> std::shared_ptr<core::Span<double>> {
                auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                for (std::size_t k = 0; k < xs.size(); ++k)
                {
                    double const rho_k = fieldAt(rho, xs[k], 0.0, 0.0);
                    (*r)[k]            = (rho_k > 0.0) ? fieldAt(rhoV[comp], xs[k], 0.0, 0.0) / rho_k
                                                       : 0.0;
                }
                return r;
            };
        InputFunction vth
            = [&](std::vector<double> const& xs) -> std::shared_ptr<core::Span<double>> {
            auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
            for (std::size_t k = 0; k < xs.size(); ++k)
                (*r)[k] = vthAt(xs[k], 0.0, 0.0);
            return r;
        };
        thermalFn = {vth, vth, vth};
    }
    else if constexpr (dim == 2)
    {
        densityFn = [&](std::vector<double> const& xs,
                        std::vector<double> const& ys) -> std::shared_ptr<core::Span<double>> {
            auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
            for (std::size_t k = 0; k < xs.size(); ++k)
                (*r)[k] = fieldAt(rho, xs[k], ys[k], 0.0);
            return r;
        };
        for (std::size_t comp = 0; comp < 3; ++comp)
            velocityFn[comp] = [&, comp](std::vector<double> const& xs, std::vector<double> const& ys)
                -> std::shared_ptr<core::Span<double>> {
                auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                for (std::size_t k = 0; k < xs.size(); ++k)
                {
                    double const rho_k = fieldAt(rho, xs[k], ys[k], 0.0);
                    (*r)[k]            = (rho_k > 0.0) ? fieldAt(rhoV[comp], xs[k], ys[k], 0.0) / rho_k
                                                       : 0.0;
                }
                return r;
            };
        InputFunction vth = [&](std::vector<double> const& xs, std::vector<double> const& ys)
            -> std::shared_ptr<core::Span<double>> {
            auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
            for (std::size_t k = 0; k < xs.size(); ++k)
                (*r)[k] = vthAt(xs[k], ys[k], 0.0);
            return r;
        };
        thermalFn = {vth, vth, vth};
    }
    else
    {
        densityFn = [&](std::vector<double> const& xs, std::vector<double> const& ys,
                        std::vector<double> const& zs) -> std::shared_ptr<core::Span<double>> {
            auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
            for (std::size_t k = 0; k < xs.size(); ++k)
                (*r)[k] = fieldAt(rho, xs[k], ys[k], zs[k]);
            return r;
        };
        for (std::size_t comp = 0; comp < 3; ++comp)
            velocityFn[comp] = [&, comp](std::vector<double> const& xs, std::vector<double> const& ys,
                                         std::vector<double> const& zs)
                -> std::shared_ptr<core::Span<double>> {
                auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                for (std::size_t k = 0; k < xs.size(); ++k)
                {
                    double const rho_k = fieldAt(rho, xs[k], ys[k], zs[k]);
                    (*r)[k]            = (rho_k > 0.0)
                                             ? fieldAt(rhoV[comp], xs[k], ys[k], zs[k]) / rho_k
                                             : 0.0;
                }
                return r;
            };
        InputFunction vth = [&](std::vector<double> const& xs, std::vector<double> const& ys,
                                std::vector<double> const& zs)
            -> std::shared_ptr<core::Span<double>> {
            auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
            for (std::size_t k = 0; k < xs.size(); ++k)
                (*r)[k] = vthAt(xs[k], ys[k], zs[k]);
            return r;
        };
        thermalFn = {vth, vth, vth};
    }

    MaxwellianInit init{densityFn, velocityFn, thermalFn, charge, nbrPPC, seed};
    init.loadParticles(out, sampleLayout);
}

} // namespace PHARE::amr

#endif // PHARE_SPAWN_MAXWELLIAN_FROM_MHD_HPP
