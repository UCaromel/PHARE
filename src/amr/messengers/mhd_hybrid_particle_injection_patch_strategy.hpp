#ifndef PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP
#define PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP

#include "core/data/ions/particle_initializers/maxwellian_particle_initializer.hpp"
#include "amr/data/field/field_data.hpp"
#include "amr/data/tensorfield/tensor_field_data.hpp"
#include "amr/data/particles/particles_data.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "core/utilities/span.hpp"
#include "core/utilities/point/point.hpp"
#include "initializer/data_provider.hpp"

#include "SAMRAI/xfer/RefinePatchStrategy.h"
#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/IntVector.h"

#include <cmath>
#include <cstddef>
#include <vector>
#include <memory>

namespace PHARE::amr
{

/**
 * @brief Injects Hybrid particles from MHD coarse patch data during initLevel/regrid.
 *
 * Extends SAMRAI::xfer::RefinePatchStrategy. Constructed inside MHDHybridMessengerStrategy
 * and passed to the RefineAlgorithm used for initLevel() and regrid(). SAMRAI calls
 * postprocessRefine() after standard field data filling (B, E already filled).
 *
 * For each ion population:
 *   1. Read MHD primitives from coarse patch: rho, V (cell-centered), P
 *   2. Derive thermal velocity: vth(x) = sqrt(P(x) / rho(x))  (isotropic, P_total / rho)
 *   3. Build InitFunction<dim> callables with zero-order coarse-cell interpolation
 *   4. Construct MaxwellianParticleInitializer and call loadParticles()
 *
 * Note: rho, V, P are all cell-centered (ddd) in MHD — no face-averaging needed.
 */
template<typename MHDModel, typename HybridModel>
class MHDHybridParticleInjectionPatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
    static constexpr std::size_t dimension = HybridModel::dimension;

    using HybridGridLayout = HybridModel::gridlayout_type;
    using MHDGridLayout    = MHDModel::gridlayout_type;
    using MHDFieldT        = MHDModel::field_type;
    using MHDGridT         = MHDModel::grid_type;
    using MHDFieldDataT    = FieldData<MHDGridLayout, MHDFieldT>;
    using MHDVecFieldDataT = TensorFieldData<1, MHDGridLayout, MHDGridT, core::MHDQuantity>;

    using IonsT          = decltype(std::declval<HybridModel>().state.ions);
    using ParticleArrayT = IonsT::particle_array_type;
    using ParticlesDataT = ParticlesData<ParticleArrayT>;

    using MaxwellianInit = core::MaxwellianParticleInitializer<ParticleArrayT, HybridGridLayout>;
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
     * V is a VecField stored as a single TensorFieldData (one ID, not three).
     * Called from MHDHybridMessengerStrategy::registerQuantities().
     */
    void registerMHDPrimIds(int rhoId, int vId, int pId)
    {
        mhdRhoId_ = rhoId;
        mhdVId_   = vId;
        mhdPId_   = pId;
    }

    /**
     * @brief Register a Hybrid ion population to be initialized from MHD fluid data.
     * charge and nbrParticlesPerCell are filled later via setPopulationPhysics().
     * Called from MHDHybridMessengerStrategy::registerQuantities() once per population.
     */
    void addPopulation(int particleDataId, std::size_t populationIndex)
    {
        populations_.push_back({particleDataId, 0.0, 0u, populationIndex});
    }

    /**
     * @brief Set per-population physics from the Hybrid model.
     * Called from MHDHybridMessengerStrategy::initLevel() before fillData(), where
     * the model is accessible.
     */
    void setPopulationPhysics(std::size_t popIdx, double charge, std::uint32_t nbrPPC)
    {
        populations_[popIdx].charge              = charge;
        populations_[popIdx].nbrParticlesPerCell = nbrPPC;
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
     * from MHD coarse primitives (rho, V, P — all cell-centered).
     *
     * Uses zero-order (nearest coarse cell) interpolation: for a fine cell at physical
     * coordinate x, the containing coarse cell is floor(x/dx_fine)/ratio.
     *
     * TODO: restrict loading to fineBox for regrid correctness (for initLevel, fineBox equals
     * the full patch so this is not an issue).
     */
    void postprocessRefine(SAMRAI::hier::Patch& fine, SAMRAI::hier::Patch const& coarse,
                           SAMRAI::hier::Box const& /*fineBox*/,
                           SAMRAI::hier::IntVector const& /*ratio*/) override
    {
        // Get coarse MHD primitives (allocated on the coarse MHD level, always available)
        auto& rho    = MHDFieldDataT::getField(coarse, mhdRhoId_);
        auto& Vcomps = MHDVecFieldDataT::getFields(coarse, mhdVId_); // std::array<Grid_t, 3>
        auto& P      = MHDFieldDataT::getField(coarse, mhdPId_);

        auto coarseLayout = layoutFromPatch<MHDGridLayout>(coarse);
        auto fineLayout   = layoutFromPatch<HybridGridLayout>(fine);

        // Refinement ratio per direction derived from mesh sizes
        auto const dx_coarse = coarseLayout.meshSize();
        auto const dx_fine   = fineLayout.meshSize();
        int const ratio_x    = static_cast<int>(std::round(dx_coarse[0] / dx_fine[0]));
        [[maybe_unused]] int const ratio_y
            = (dimension >= 2) ? static_cast<int>(std::round(dx_coarse[1] / dx_fine[1])) : 1;
        [[maybe_unused]] int const ratio_z
            = (dimension == 3) ? static_cast<int>(std::round(dx_coarse[2] / dx_fine[2])) : 1;

        // getCoarseLocal: fine cell-center coordinate → coarse local array index (Point<uint32_t>)
        // x = (i_fine_amr + 0.5)*dx_fine  →  static_cast<int>(x/dx_fine) = i_fine_amr
        // j_coarse_amr = i_fine_amr / ratio  →  AMRToLocal gives array index (ghost-aware)
        // All hydro quantities are cell-centered (ddd) so AMRToLocal is correct for all of them.
        auto getCoarseLocal = [&](double x, [[maybe_unused]] double y,
                                  [[maybe_unused]] double z) {
            int const ix = static_cast<int>(x / dx_fine[0]) / ratio_x;
            if constexpr (dimension == 1)
                return coarseLayout.AMRToLocal(core::Point<int, 1>{ix});
            else if constexpr (dimension == 2)
            {
                int const iy = static_cast<int>(y / dx_fine[1]) / ratio_y;
                return coarseLayout.AMRToLocal(core::Point<int, 2>{ix, iy});
            }
            else
            {
                int const iy = static_cast<int>(y / dx_fine[1]) / ratio_y;
                int const iz = static_cast<int>(z / dx_fine[2]) / ratio_z;
                return coarseLayout.AMRToLocal(core::Point<int, 3>{ix, iy, iz});
            }
        };

        // Evaluate a cell-centered coarse field at the coarse cell containing (x,y,z)
        auto fieldAt = [&](auto& field, double x, double y, double z) -> double {
            auto local = getCoarseLocal(x, y, z);
            if constexpr (dimension == 1)
                return field(local[0]);
            else if constexpr (dimension == 2)
                return field(local[0], local[1]);
            else
                return field(local[0], local[1], local[2]);
        };

        for (auto const& pop : populations_)
        {
            // Deterministic per-patch seed: combine level, population, and patch lower corner
            std::size_t const seed = static_cast<std::size_t>(levelNumber_)
                                   + pop.populationIndex
                                   + static_cast<std::size_t>(fine.getBox().lower(0));

            // Build dimension-typed InitFunctions and inject particles.
            // Lambda captures are by reference (&) — all referenced data lives for the
            // duration of this postprocessRefine call.
            if constexpr (dimension == 1)
            {
                auto mk = [&](auto& field) -> InputFunction {
                    return [&, &f = field](std::vector<double> const& xs)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                            (*r)[k] = fieldAt(f, xs[k], 0.0, 0.0);
                        return r;
                    };
                };
                InputFunction vth{[&](std::vector<double> const& xs)
                                      -> std::shared_ptr<core::Span<double>> {
                    auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                    for (std::size_t k = 0; k < xs.size(); ++k)
                    {
                        double const rho_k = fieldAt(rho, xs[k], 0.0, 0.0);
                        double const P_k   = fieldAt(P, xs[k], 0.0, 0.0);
                        // vth = sqrt(P/rho) isotropic; P_total used as proxy (Pe not tracked)
                        (*r)[k] = (rho_k > 0.0) ? std::sqrt(std::max(P_k, 0.0) / rho_k) : 0.0;
                    }
                    return r;
                }};

                MaxwellianInit init{mk(rho),
                                    {mk(Vcomps[0]), mk(Vcomps[1]), mk(Vcomps[2])},
                                    {vth, vth, vth},
                                    pop.charge, pop.nbrParticlesPerCell, seed};
                auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                    fine.getPatchData(pop.particleDataId));
                init.loadParticles(partData.domainParticles, fineLayout);
            }
            else if constexpr (dimension == 2)
            {
                auto mk = [&](auto& field) -> InputFunction {
                    return [&, &f = field](std::vector<double> const& xs,
                                          std::vector<double> const& ys)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                            (*r)[k] = fieldAt(f, xs[k], ys[k], 0.0);
                        return r;
                    };
                };
                InputFunction vth{[&](std::vector<double> const& xs,
                                      std::vector<double> const& ys)
                                      -> std::shared_ptr<core::Span<double>> {
                    auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                    for (std::size_t k = 0; k < xs.size(); ++k)
                    {
                        double const rho_k = fieldAt(rho, xs[k], ys[k], 0.0);
                        double const P_k   = fieldAt(P, xs[k], ys[k], 0.0);
                        (*r)[k] = (rho_k > 0.0) ? std::sqrt(std::max(P_k, 0.0) / rho_k) : 0.0;
                    }
                    return r;
                }};

                MaxwellianInit init{mk(rho),
                                    {mk(Vcomps[0]), mk(Vcomps[1]), mk(Vcomps[2])},
                                    {vth, vth, vth},
                                    pop.charge, pop.nbrParticlesPerCell, seed};
                auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                    fine.getPatchData(pop.particleDataId));
                init.loadParticles(partData.domainParticles, fineLayout);
            }
            else // dimension == 3
            {
                auto mk = [&](auto& field) -> InputFunction {
                    return [&, &f = field](std::vector<double> const& xs,
                                          std::vector<double> const& ys,
                                          std::vector<double> const& zs)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                            (*r)[k] = fieldAt(f, xs[k], ys[k], zs[k]);
                        return r;
                    };
                };
                InputFunction vth{[&](std::vector<double> const& xs,
                                      std::vector<double> const& ys,
                                      std::vector<double> const& zs)
                                      -> std::shared_ptr<core::Span<double>> {
                    auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                    for (std::size_t k = 0; k < xs.size(); ++k)
                    {
                        double const rho_k = fieldAt(rho, xs[k], ys[k], zs[k]);
                        double const P_k   = fieldAt(P, xs[k], ys[k], zs[k]);
                        (*r)[k] = (rho_k > 0.0) ? std::sqrt(std::max(P_k, 0.0) / rho_k) : 0.0;
                    }
                    return r;
                }};

                MaxwellianInit init{mk(rho),
                                    {mk(Vcomps[0]), mk(Vcomps[1]), mk(Vcomps[2])},
                                    {vth, vth, vth},
                                    pop.charge, pop.nbrParticlesPerCell, seed};
                auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                    fine.getPatchData(pop.particleDataId));
                init.loadParticles(partData.domainParticles, fineLayout);
            }
        }
    }

private:
    int mhdRhoId_    = -1;
    int mhdVId_      = -1; // single TensorFieldData ID for V (Vx,Vy,Vz — all cell-centered)
    int mhdPId_      = -1;
    int levelNumber_ = -1;

    std::vector<PopInfo> populations_;
};

} // namespace PHARE::amr

#endif // PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP
