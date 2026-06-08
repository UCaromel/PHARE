#ifndef PHARE_MHD_HYBRID_PARTICLE_SPAWN_STRATEGY_HPP
#define PHARE_MHD_HYBRID_PARTICLE_SPAWN_STRATEGY_HPP

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "core/data/ions/particle_initializers/maxwellian_particle_initializer.hpp"
#include "core/data/particles/particle.hpp"
#include "core/utilities/types.hpp"

#include "SAMRAI/xfer/RefinePatchStrategy.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace PHARE::amr
{

// Spawns Maxwellian particles from strategy-owned fine primitive fields (rho, V, P) in
// postprocessRefine. One instance per particle set (Domain/Old/New); each loops over all
// ion populations.
//
// Spawn box invariant: fine_box is clipped to the particle ghost ring
// (grow(patchInterior, particleGhostWidth)) before iterating. This is required because
// fine_box comes from the prim-field schedule (field ghost count = 6) but particles must
// not be placed closer than ghostWidthForParticles cells from the field boundary —
// the deposit stencil (LevelGhostDeposit) would reach before the field allocation start
// and underflow to UINT32_MAX. HybridHybrid avoids this naturally because its
// PatchLevelBorderFillPattern uses the particle ghost count (= 2 for order=2), so
// fine_box is already the 2-cell ring.
template<typename FieldDataT, typename VecFieldDataT, typename ParticlesDataT, typename GridLayoutT>
class MHDHybridParticleSpawnStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

public:
    enum class ParticleBucket { Domain, GhostOld, GhostNew };

    struct PopParams
    {
        int particleDestId;
        double charge;
        std::uint32_t nbrPPC;
        std::optional<std::size_t> seed;
        ParticleBucket bucket = ParticleBucket::Domain;
    };

    // Default-constructible: field IDs and populations set via setters before first fillData().
    MHDHybridParticleSpawnStrategy() = default;

    void setFieldIds(int primRhoId, int primVId, int primPId)
    {
        primRhoId_ = primRhoId;
        primVId_   = primVId;
        primPId_   = primPId;
    }

    void setPopulations(std::vector<PopParams> pops) { populations_ = std::move(pops); }

    bool hasPopulations() const { return !populations_.empty(); }

    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch&, double,
                                        SAMRAI::hier::IntVector const&) override
    {
    }

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    void preprocessRefine(SAMRAI::hier::Patch&, SAMRAI::hier::Patch const&,
                          SAMRAI::hier::Box const&, SAMRAI::hier::IntVector const&) override
    {
    }

    void postprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& /*coarse*/,
                           SAMRAI::hier::Box const& fine_box,
                           SAMRAI::hier::IntVector const& /*ratio*/) override
    {
        std::cout << "[DIAG spawn::postprocessRefine] pops=" << populations_.size()
                  << " fine_box=" << fine_box << std::endl;
        if (populations_.empty())
            return;

        auto& rho    = FieldDataT::getField(fine, primRhoId_);
        auto& Vcomps = VecFieldDataT::getFields(fine, primVId_);
        auto& P      = FieldDataT::getField(fine, primPId_);

        auto const patchLayout = layoutFromPatch<GridLayoutT>(fine);

        auto fieldAt = [](auto const& f, auto const& li) -> double {
            if constexpr (dimension == 1)
                return f(li[0]);
            else if constexpr (dimension == 2)
                return f(li[0], li[1]);
            else
                return f(li[0], li[1], li[2]);
        };

        for (auto const& pop : populations_)
        {
            auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                fine.getPatchData(pop.particleDestId));

            // Clip to particle ghost ring: deposit stencil reaches iCell-shift into the
            // field; particles beyond ghostWidthForParticles cells of the interior cause
            // stencil to underflow past the field allocation start.
            auto const spawnBox
                = fine_box
                  * SAMRAI::hier::Box::grow(fine.getBox(), partData.getGhostCellWidth());
            if (spawnBox.empty())
                continue;

            auto& destParts = [&]() -> decltype(partData.domainParticles)& {
                switch (pop.bucket)
                {
                    case ParticleBucket::Domain:   return partData.domainParticles;
                    case ParticleBucket::GhostOld: return partData.levelGhostParticlesOld;
                    case ParticleBucket::GhostNew: return partData.levelGhostParticlesNew;
                }
                return partData.domainParticles; // unreachable
            }();
            std::cout << "[DIAG spawn::postprocessRefine] bucket=" << static_cast<int>(pop.bucket)
                      << " spawnBox=" << spawnBox << " destParts.size_before_clear=" << destParts.size()
                      << std::endl;
            destParts.clear();

            auto randGen = [&]() -> std::mt19937_64 {
                if (!pop.seed.has_value())
                {
                    std::random_device rd;
                    std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
                    return std::mt19937_64{seq};
                }
                return std::mt19937_64{*pop.seed};
            }();

            core::ParticleDeltaDistribution<double> deltaDistrib;

            bool firstCell = true;
            for (auto const& amrIdx : phare_box_from<dimension>(spawnBox))
            {
                auto const localIdx = patchLayout.AMRToLocal(amrIdx);

                double const rho_k = fieldAt(rho, localIdx);
                if (firstCell)
                {
                    std::cout << "[DIAG spawn::postprocessRefine] first cell amrIdx[0]=" << amrIdx[0]
                              << " rho_k=" << rho_k << std::endl;
                    firstCell = false;
                }
                if (rho_k <= 0.0)
                    continue;

                double const P_k  = fieldAt(P, localIdx);
                double const vth  = std::sqrt(std::max(P_k, 0.0) / rho_k);
                double const Vx_k = fieldAt(Vcomps[0], localIdx);
                double const Vy_k = fieldAt(Vcomps[1], localIdx);
                double const Vz_k = fieldAt(Vcomps[2], localIdx);

                double const cellWeight = rho_k / pop.nbrPPC;
                auto const iCell        = core::for_N_make_array<dimension>(
                    [&](auto d) { return amrIdx[d]; });

                std::array<double, 3> partVelocity;
                for (std::uint32_t ipart = 0; ipart < pop.nbrPPC; ++ipart)
                {
                    core::maxwellianVelocity({Vx_k, Vy_k, Vz_k}, {vth, vth, vth},
                                             randGen, partVelocity);
                    auto const delta = core::for_N_make_array<dimension>(
                        [&](auto) { return deltaDistrib(randGen); });
                    destParts.emplace_back(core::Particle<dimension>{
                        cellWeight, pop.charge, iCell, delta, partVelocity});
                }
            }
            std::cout << "[DIAG spawn::postprocessRefine] destParts.size_after=" << destParts.size()
                      << std::endl;
        }
    }

private:
    int primRhoId_ = -1, primVId_ = -1, primPId_ = -1;
    std::vector<PopParams> populations_;
};

} // namespace PHARE::amr

#endif // PHARE_MHD_HYBRID_PARTICLE_SPAWN_STRATEGY_HPP
