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
 *   1. Read MHD conservatives from coarse patch: rho (ddd), rhoV (ddd), B (Pdd/dPd/ddP), Etot (ddd)
 *   2. Derive V = rhoV/rho (cell-centered), B_cell (face-to-cell average), P via EOS
 *   3. Derive thermal velocity: vth(x) = sqrt(P(x) / rho(x))  (isotropic)
 *   4. Build InitFunction<dim> callables with zero-order coarse-cell interpolation
 *   5. Construct MaxwellianParticleInitializer and call loadParticles()
 *
 * Conservative path: after reflux only rho/rhoV/B/Etot are guaranteed fresh;
 * V and P are derived inline here rather than read from stale primitive fields.
 */
template<typename MHDModel, typename HybridModel>
class MHDHybridParticleInjectionPatchStrategy : public SAMRAI::xfer::RefinePatchStrategy
{
    static constexpr std::size_t dimension = HybridModel::dimension;

    using HybridGridLayout = HybridModel::gridlayout_type;
    using MHDGridLayout    = MHDModel::gridlayout_type;
    using MHDGridT         = MHDModel::grid_type;
    using MHDFieldDataT    = FieldData<MHDGridLayout, MHDGridT>;
    using MHDVecFieldDataT = TensorFieldData<1, MHDGridLayout, MHDGridT, core::PhysicalQuantity>;

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
     * @brief Register MHD coarse patch data IDs for conservative fields needed by particle injection.
     * rhoV and B are VecFields stored as single TensorFieldData IDs (not three).
     * Called from MHDHybridMessengerStrategy::registerInitComms_() and registerGhostComms_().
     */
    void registerMHDConsIds(int rhoId, int rhoVId, int BId, int EtotId, double gamma)
    {
        mhdRhoId_   = rhoId;
        mhdRhoVId_  = rhoVId;
        mhdBId_     = BId;
        mhdEtotId_  = EtotId;
        gamma_      = gamma;
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
        // Get coarse MHD conservatives (always fresh — filled before postprocessRefine fires)
        auto& rho      = MHDFieldDataT::getField(coarse, mhdRhoId_);
        auto& rhoVcomps = MHDVecFieldDataT::getFields(coarse, mhdRhoVId_); // rhoVx/Vy/Vz (ddd)
        auto& Bcomps   = MHDVecFieldDataT::getFields(coarse, mhdBId_);    // Bx(Pdd),By(dPd),Bz(ddP)
        auto& Etot     = MHDFieldDataT::getField(coarse, mhdEtotId_);

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
                auto mkScalar = [&](auto& field) -> InputFunction {
                    return [&, &f = field](std::vector<double> const& xs)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                            (*r)[k] = fieldAt(f, xs[k], 0.0, 0.0);
                        return r;
                    };
                };
                // Velocity: V = rhoV / rho (cell-centered, all ddd)
                auto mkVel = [&](std::size_t comp) -> InputFunction {
                    return [&, comp](std::vector<double> const& xs)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                        {
                            double const rho_k  = fieldAt(rho, xs[k], 0.0, 0.0);
                            double const rhoV_k = fieldAt(rhoVcomps[comp], xs[k], 0.0, 0.0);
                            (*r)[k] = (rho_k > 0.0) ? rhoV_k / rho_k : 0.0;
                        }
                        return r;
                    };
                };
                InputFunction vth{[&](std::vector<double> const& xs)
                                      -> std::shared_ptr<core::Span<double>> {
                    auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                    for (std::size_t k = 0; k < xs.size(); ++k)
                    {
                        double const rho_k   = fieldAt(rho, xs[k], 0.0, 0.0);
                        double const rhoVx_k = fieldAt(rhoVcomps[0], xs[k], 0.0, 0.0);
                        double const rhoVy_k = fieldAt(rhoVcomps[1], xs[k], 0.0, 0.0);
                        double const rhoVz_k = fieldAt(rhoVcomps[2], xs[k], 0.0, 0.0);
                        double const Vx_k    = (rho_k > 0.0) ? rhoVx_k / rho_k : 0.0;
                        double const Vy_k    = (rho_k > 0.0) ? rhoVy_k / rho_k : 0.0;
                        double const Vz_k    = (rho_k > 0.0) ? rhoVz_k / rho_k : 0.0;
                        // Bx (Pdd): face-to-cell average in X; By/Bz are ddd in 1D
                        auto lc              = getCoarseLocal(xs[k], 0.0, 0.0);
                        double const bx_k    = 0.5 * (Bcomps[0](lc[0]) + Bcomps[0](lc[0] + 1));
                        double const by_k    = Bcomps[1](lc[0]);
                        double const bz_k    = Bcomps[2](lc[0]);
                        double const Etot_k  = fieldAt(Etot, xs[k], 0.0, 0.0);
                        double const KE_k    = 0.5 * rho_k * (Vx_k * Vx_k + Vy_k * Vy_k + Vz_k * Vz_k);
                        double const ME_k    = 0.5 * (bx_k * bx_k + by_k * by_k + bz_k * bz_k);
                        double const P_k     = (gamma_ - 1.0) * (Etot_k - KE_k - ME_k);
                        (*r)[k] = (rho_k > 0.0) ? std::sqrt(std::max(P_k, 0.0) / rho_k) : 0.0;
                    }
                    return r;
                }};

                MaxwellianInit init{mkScalar(rho),
                                    {mkVel(0), mkVel(1), mkVel(2)},
                                    {vth, vth, vth},
                                    pop.charge, pop.nbrParticlesPerCell, seed};
                auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                    fine.getPatchData(pop.particleDataId));
                init.loadParticles(partData.domainParticles, fineLayout);
            }
            else if constexpr (dimension == 2)
            {
                auto mkScalar = [&](auto& field) -> InputFunction {
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
                auto mkVel = [&](std::size_t comp) -> InputFunction {
                    return [&, comp](std::vector<double> const& xs,
                                    std::vector<double> const& ys)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                        {
                            double const rho_k  = fieldAt(rho, xs[k], ys[k], 0.0);
                            double const rhoV_k = fieldAt(rhoVcomps[comp], xs[k], ys[k], 0.0);
                            (*r)[k] = (rho_k > 0.0) ? rhoV_k / rho_k : 0.0;
                        }
                        return r;
                    };
                };
                InputFunction vth{[&](std::vector<double> const& xs,
                                      std::vector<double> const& ys)
                                      -> std::shared_ptr<core::Span<double>> {
                    auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                    for (std::size_t k = 0; k < xs.size(); ++k)
                    {
                        double const rho_k   = fieldAt(rho, xs[k], ys[k], 0.0);
                        double const rhoVx_k = fieldAt(rhoVcomps[0], xs[k], ys[k], 0.0);
                        double const rhoVy_k = fieldAt(rhoVcomps[1], xs[k], ys[k], 0.0);
                        double const rhoVz_k = fieldAt(rhoVcomps[2], xs[k], ys[k], 0.0);
                        double const Vx_k    = (rho_k > 0.0) ? rhoVx_k / rho_k : 0.0;
                        double const Vy_k    = (rho_k > 0.0) ? rhoVy_k / rho_k : 0.0;
                        double const Vz_k    = (rho_k > 0.0) ? rhoVz_k / rho_k : 0.0;
                        // Bx (Pdd): avg in X; By (dPd): avg in Y; Bz (ddP): ddd in 2D
                        auto lc              = getCoarseLocal(xs[k], ys[k], 0.0);
                        double const bx_k    = 0.5 * (Bcomps[0](lc[0], lc[1]) + Bcomps[0](lc[0] + 1, lc[1]));
                        double const by_k    = 0.5 * (Bcomps[1](lc[0], lc[1]) + Bcomps[1](lc[0], lc[1] + 1));
                        double const bz_k    = Bcomps[2](lc[0], lc[1]);
                        double const Etot_k  = fieldAt(Etot, xs[k], ys[k], 0.0);
                        double const KE_k    = 0.5 * rho_k * (Vx_k * Vx_k + Vy_k * Vy_k + Vz_k * Vz_k);
                        double const ME_k    = 0.5 * (bx_k * bx_k + by_k * by_k + bz_k * bz_k);
                        double const P_k     = (gamma_ - 1.0) * (Etot_k - KE_k - ME_k);
                        (*r)[k] = (rho_k > 0.0) ? std::sqrt(std::max(P_k, 0.0) / rho_k) : 0.0;
                    }
                    return r;
                }};

                MaxwellianInit init{mkScalar(rho),
                                    {mkVel(0), mkVel(1), mkVel(2)},
                                    {vth, vth, vth},
                                    pop.charge, pop.nbrParticlesPerCell, seed};
                auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                    fine.getPatchData(pop.particleDataId));
                init.loadParticles(partData.domainParticles, fineLayout);
            }
            else // dimension == 3
            {
                auto mkScalar = [&](auto& field) -> InputFunction {
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
                auto mkVel = [&](std::size_t comp) -> InputFunction {
                    return [&, comp](std::vector<double> const& xs,
                                    std::vector<double> const& ys,
                                    std::vector<double> const& zs)
                        -> std::shared_ptr<core::Span<double>>
                    {
                        auto r = std::make_shared<core::VectorSpan<double>>(xs.size(), 0.0);
                        for (std::size_t k = 0; k < xs.size(); ++k)
                        {
                            double const rho_k  = fieldAt(rho, xs[k], ys[k], zs[k]);
                            double const rhoV_k = fieldAt(rhoVcomps[comp], xs[k], ys[k], zs[k]);
                            (*r)[k] = (rho_k > 0.0) ? rhoV_k / rho_k : 0.0;
                        }
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
                        double const rho_k   = fieldAt(rho, xs[k], ys[k], zs[k]);
                        double const rhoVx_k = fieldAt(rhoVcomps[0], xs[k], ys[k], zs[k]);
                        double const rhoVy_k = fieldAt(rhoVcomps[1], xs[k], ys[k], zs[k]);
                        double const rhoVz_k = fieldAt(rhoVcomps[2], xs[k], ys[k], zs[k]);
                        double const Vx_k    = (rho_k > 0.0) ? rhoVx_k / rho_k : 0.0;
                        double const Vy_k    = (rho_k > 0.0) ? rhoVy_k / rho_k : 0.0;
                        double const Vz_k    = (rho_k > 0.0) ? rhoVz_k / rho_k : 0.0;
                        // Bx (Pdd): avg in X; By (dPd): avg in Y; Bz (ddP): avg in Z
                        auto lc              = getCoarseLocal(xs[k], ys[k], zs[k]);
                        double const bx_k    = 0.5 * (Bcomps[0](lc[0], lc[1], lc[2]) + Bcomps[0](lc[0] + 1, lc[1], lc[2]));
                        double const by_k    = 0.5 * (Bcomps[1](lc[0], lc[1], lc[2]) + Bcomps[1](lc[0], lc[1] + 1, lc[2]));
                        double const bz_k    = 0.5 * (Bcomps[2](lc[0], lc[1], lc[2]) + Bcomps[2](lc[0], lc[1], lc[2] + 1));
                        double const Etot_k  = fieldAt(Etot, xs[k], ys[k], zs[k]);
                        double const KE_k    = 0.5 * rho_k * (Vx_k * Vx_k + Vy_k * Vy_k + Vz_k * Vz_k);
                        double const ME_k    = 0.5 * (bx_k * bx_k + by_k * by_k + bz_k * bz_k);
                        double const P_k     = (gamma_ - 1.0) * (Etot_k - KE_k - ME_k);
                        (*r)[k] = (rho_k > 0.0) ? std::sqrt(std::max(P_k, 0.0) / rho_k) : 0.0;
                    }
                    return r;
                }};

                MaxwellianInit init{mkScalar(rho),
                                    {mkVel(0), mkVel(1), mkVel(2)},
                                    {vth, vth, vth},
                                    pop.charge, pop.nbrParticlesPerCell, seed};
                auto& partData = *std::dynamic_pointer_cast<ParticlesDataT>(
                    fine.getPatchData(pop.particleDataId));
                init.loadParticles(partData.domainParticles, fineLayout);
            }
        }
    }

private:
    int mhdRhoId_   = -1;
    int mhdRhoVId_  = -1; // TensorFieldData ID for rhoV (rhoVx/Vy/Vz — all ddd)
    int mhdBId_     = -1; // TensorFieldData ID for B (Pdd/dPd/ddP — face-centered)
    int mhdEtotId_  = -1; // scalar total energy (ddd)
    double gamma_   = 5.0 / 3.0;
    int levelNumber_ = -1;

    std::vector<PopInfo> populations_;
};

} // namespace PHARE::amr

#endif // PHARE_MHD_HYBRID_PARTICLE_INJECTION_PATCH_STRATEGY_HPP
