#ifndef PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP
#define PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP

#include "core/data/ions/particle_initializers/maxwellian_particle_initializer.hpp"
#include "amr/data/field/field_data.hpp"
#include "initializer/data_provider.hpp"

#include "SAMRAI/xfer/RefinePatchStrategy.h"
#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/IntVector.h"

#include <cstddef>
#include <vector>

namespace PHARE::amr
{

/**
 * @brief Injects Hybrid particles from MHD coarse patch data during initLevel/regrid.
 *
 * Extends SAMRAI::xfer::RefinePatchStrategy. Constructed inside MHDHybridMessengerStrategy
 * and passed to the RefineAlgorithm used for initLevel() and regrid(). SAMRAI calls
 * postprocessRefine() after standard field data filling (B, E, rho, V already filled).
 *
 * For each ion population:
 *   1. Read MHD primitives from coarse patch: rho, V, P
 *   2. Derive thermal velocity: vth[d](x) = sqrt(P_i(x) / rho(x))  (isotropic)
 *   3. Build InitFunction<dim> callables wrapping coarse values linearly interpolated
 *      to fine resolution
 *   4. Construct MaxwellianParticleInitializer and call loadParticles() restricted to fineBox
 */
template<typename MHDModel, typename HybridModel>
class MHDHybridParticleInjectionPatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
    static constexpr std::size_t dimension = HybridModel::dimension;

    using HybridGridLayout = HybridModel::gridlayout_type;
    using MHDGridLayout    = MHDModel::gridlayout_type;
    using MHDFieldT        = MHDModel::field_type;
    using MHDFieldDataT    = FieldData<MHDGridLayout, MHDFieldT>;

    using IonsT         = decltype(std::declval<HybridModel>().state.ions);
    using ParticleArray = IonsT::particle_array_type;

    using MaxwellianInit = core::MaxwellianParticleInitializer<ParticleArray, HybridGridLayout>;
    using InputFunction  = PHARE::initializer::InitFunction<dimension>;

    struct PopInfo
    {
        int particleDataId;
        double charge;
        std::uint32_t nbrParticlesPerCell;
        std::size_t populationIndex;
    };

public:
    MHDHybridParticleInjectionPatchStrategy() = default;

    /**
     * @brief Register MHD coarse patch data IDs for primitives needed by particle injection.
     * Called from MHDHybridMessengerStrategy::registerQuantities().
     */
    void registerMHDPrimIds(int rhoId, int vxId, int vyId, int vzId, int pId)
    {
        mhdRhoId_ = rhoId;
        mhdVxId_  = vxId;
        mhdVyId_  = vyId;
        mhdVzId_  = vzId;
        mhdPId_   = pId;
    }

    /**
     * @brief Register a Hybrid ion population to be initialized from MHD fluid data.
     * Called from MHDHybridMessengerStrategy::registerQuantities() once per population.
     */
    void addPopulation(int particleDataId, double charge, std::uint32_t nbrParticlesPerCell,
                       std::size_t populationIndex)
    {
        populations_.push_back({particleDataId, charge, nbrParticlesPerCell, populationIndex});
    }

    void setLevelNumber(int levelNumber) { levelNumber_ = levelNumber; }

    // -------------------------------------------------------------------------
    // SAMRAI RefinePatchStrategy interface
    // -------------------------------------------------------------------------

    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch& /*patch*/,
                                       double const /*fillTime*/,
                                       SAMRAI::hier::IntVector const& /*ghostWidthToFill*/) override
    {
    }

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector(dim, 0);
    }

    void preprocessRefine(SAMRAI::hier::Patch& /*fine*/, SAMRAI::hier::Patch const& /*coarse*/,
                          SAMRAI::hier::Box const& /*fineBox*/,
                          SAMRAI::hier::IntVector const& /*ratio*/) override
    {
    }

    /**
     * @brief Called by SAMRAI after standard field filling. Populates fine Hybrid particles
     * from MHD coarse primitives (rho, V, P).
     *
     * Uses the pre-filled MHD primitive fields on the coarse patch to build
     * MaxwellianParticleInitializer callables and inject particles restricted to fineBox.
     */
    void postprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                           SAMRAI::hier::Box const& fineBox,
                           SAMRAI::hier::IntVector const& ratio) override
    {
        // TODO: implement particle injection from MHD coarse patch data.
        //
        // 1. Get coarse MHD primitives (already filled by SAMRAI refine schedule):
        //    auto& rho = MHDFieldDataT::getField(coarse, mhdRhoId_);
        //    auto& Vx  = MHDFieldDataT::getField(coarse, mhdVxId_);
        //    ... (Vy, Vz, P)
        //
        // 2. Get fine layout:
        //    auto fineLayout = layoutFromPatch<HybridGridLayout>(fine);
        //
        // 3. Build InitFunction<dim> callables wrapping bilinear interpolation from
        //    coarse rho, V, P to fine spatial positions:
        //    InputFunction density    = [&](auto const&... args) { /* interp rho  */ };
        //    InputFunction bulkVel[3] = { /* interp Vx, Vy, Vz */ };
        //    InputFunction thermalVel[3] = { [&](auto const&... args) {
        //        return std::sqrt(P_i_interp(args...) / rho_interp(args...));
        //    }};
        //
        // 4. For each population:
        //    for (auto const& pop : populations_)
        //    {
        //        std::size_t seed = static_cast<std::size_t>(levelNumber_)
        //                         + pop.populationIndex
        //                         + static_cast<std::size_t>(fine.getGlobalId().getValue());
        //        MaxwellianInit init{density, {bulkVel[0], bulkVel[1], bulkVel[2]},
        //                           {thermalVel[0], thermalVel[1], thermalVel[2]},
        //                           pop.charge, pop.nbrParticlesPerCell, seed};
        //        // get particle array from fine patch via pop.particleDataId
        //        // init.loadParticles(particles, fineLayout);  // restricted to fineBox
        //    }
        (void)fine;
        (void)coarse;
        (void)fineBox;
        (void)ratio;
    }

private:
    int mhdRhoId_ = -1;
    int mhdVxId_  = -1;
    int mhdVyId_  = -1;
    int mhdVzId_  = -1;
    int mhdPId_   = -1;
    int levelNumber_ = -1;

    std::vector<PopInfo> populations_;
};

} // namespace PHARE::amr

#endif // PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP
