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
#include <functional>
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
    enum class SpawnTargetBucket { Domain, LevelGhostOld, LevelGhostNew };

    struct PopParams
    {
        int particleDestId;
        double charge;
        double mass;
        std::uint32_t nbrPPC;
        std::optional<std::size_t> seed;
    };

    // Default-constructible: field IDs and populations set via setters before first fillData().
    MHDHybridParticleSpawnStrategy() = default;

    // Instance-wide spawn mode: which particle bucket to append to, and whether to clip
    // fine_box to the particle ghost ring (see spawn box invariant above). Nested-frame
    // fragments spawn (Domain, clip=false): interp temp levels have no deposit, so the
    // stencil-underflow rationale for clipping does not apply — and clipping there would
    // starve the copy transactions the spawned content exists to feed.
    void setSpawnMode(SpawnTargetBucket bucket, bool clipToInterior)
    {
        bucket_         = bucket;
        clipToInterior_ = clipToInterior;
    }

    void setFieldIds(int primRhoId, int primVId, int primPId)
    {
        primRhoId_ = primRhoId;
        primVId_   = primVId;
        primPId_   = primPId;
    }

    void setPopulations(std::vector<PopParams> pops)
    {
        populations_ = std::move(pops);
        if (populations_.empty())
            return;
        totalPPC_ = 0;
        for (auto const& pop : populations_)
            totalPPC_ += pop.nbrPPC;
        meanMass_ = 0.0;
        for (auto const& pop : populations_)
            meanMass_ += (static_cast<double>(pop.nbrPPC) / totalPPC_) * pop.mass;
    }

    // Nested single-population fragments must reproduce the across-population weight split
    // of the top-frame instances (cellWeight = rho / (meanMass * totalPPC) with totals over
    // ALL populations, so per-pop deposits sum to rho). setPopulations derives the totals
    // from ITS list — a single-pop list would make that pop carry the full rho. Call this
    // after setPopulations to substitute the across-population totals.
    void setSpawnWeights(std::uint32_t totalPPC, double meanMass)
    {
        totalPPC_ = totalPPC;
        meanMass_ = meanMass;
    }

    // Pe(rho): electron pressure from MHD mass density; unset means Pe = 0.
    void setElectronPressure(std::function<double(double)> pe) { pe_ = std::move(pe); }

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
                = clipToInterior_
                      ? fine_box * SAMRAI::hier::Box::grow(fine.getBox(),
                                                           partData.getGhostCellWidth())
                      : fine_box;
            if (spawnBox.empty())
                continue;

            // Append-only: SAMRAI calls postprocessRefine once per fill box (one per
            // coarse-patch overlap; ghost regions split into per-face boxes), so clearing
            // here would keep only the last box's particles. The messenger clears the
            // buckets once per fill episode, before fillData — same lifecycle as
            // HybridHybrid (ParticlesData transfers append; clears at lifecycle points).
            auto& destParts = [&]() -> decltype(partData.domainParticles)& {
                switch (bucket_)
                {
                    case SpawnTargetBucket::Domain:        return partData.domainParticles;
                    case SpawnTargetBucket::LevelGhostOld: return partData.levelGhostParticlesOld;
                    case SpawnTargetBucket::LevelGhostNew: return partData.levelGhostParticlesNew;
                }
                return partData.domainParticles; // unreachable
            }();

            using ParticleArray_t = std::decay_t<decltype(partData.domainParticles)>;
            auto randGen = core::MaxwellianParticleInitializer<ParticleArray_t,
                                                               GridLayoutT>::getRNG(pop.seed);

            core::ParticleDeltaDistribution<double> deltaDistrib;

            for (auto const& amrIdx : phare_box_from<dimension>(spawnBox))
            {
                auto const localIdx = patchLayout.AMRToLocal(amrIdx);

                double const rho_k = fieldAt(rho, localIdx);
                if (rho_k <= 0.0)
                    continue;

                // Pi = P - Pe: MHD P is total pressure; spawned ions must carry only the
                // ion thermal energy or the sync's explicit Pe/(gamma-1) double-counts it.
                double const P_k  = fieldAt(P, localIdx);
                double const Pe_k = pe_ ? pe_(rho_k) : 0.0;
                double const Pi_k = std::max(P_k - Pe_k, 0.0);
                double const T_k  = Pi_k * meanMass_ / rho_k; // common ion temperature
                double const vth  = std::sqrt(T_k / pop.mass);
                double const Vx_k = fieldAt(Vcomps[0], localIdx);
                double const Vy_k = fieldAt(Vcomps[1], localIdx);
                double const Vz_k = fieldAt(Vcomps[2], localIdx);

                double const cellWeight = rho_k / (meanMass_ * totalPPC_);
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
        }
    }

private:
    int primRhoId_ = -1, primVId_ = -1, primPId_ = -1;
    SpawnTargetBucket bucket_ = SpawnTargetBucket::Domain;
    bool clipToInterior_      = true;
    std::vector<PopParams> populations_;
    std::function<double(double)> pe_;
    std::uint32_t totalPPC_ = 0;
    double meanMass_       = 1.0;
};

} // namespace PHARE::amr

#endif // PHARE_MHD_HYBRID_PARTICLE_SPAWN_STRATEGY_HPP
