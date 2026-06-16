#ifndef PHARE_MOMENTUM_TENSOR_BORDER_FILLER_HPP
#define PHARE_MOMENTUM_TENSOR_BORDER_FILLER_HPP

#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "core/physical_quantities.hpp"
#include "core/utilities/types.hpp"
#include "core/numerics/interpolator/interpolator.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/messengers/field_operate_transaction.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"

#include <SAMRAI/xfer/RefineAlgorithm.h>
#include <SAMRAI/xfer/RefineSchedule.h>
#include <SAMRAI/hier/PatchLevel.h>

#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace PHARE::amr
{
// Produces a border-completed ion momentum tensor on a single level for the coupled
// MHD-Hybrid momentum/energy flux. The per-population tensors are deposited from
// particles, each is additively border-summed across patch/sibling-patch seams with the
// same PlusEquals seam-sum the fluid diagnostic uses (diagnostic_model_view.hpp 188-259),
// then aggregated into ions.momentumTensor().
//
// Solver-owned (the deposit + consume both live in SolverPPC::accumulateFluxSum and the
// single-level border-sum schedule needs only the in-hand PatchLevel). Schedules are
// cached per level number and cleared by the solver's onRegrid() — a RefineSchedule bakes
// in patch geometry that regrid changes.
template<typename ResourcesManager, typename GridLayout, typename Ions, std::size_t dimension,
         std::size_t interp_order>
class MomentumTensorBorderFiller
{
    using TensorFieldT      = typename Ions::tensorfield_type;
    using TensorFieldData_t = typename ResourcesManager::template UserTensorField_t<2>::patch_data_type;
    using value_type        = typename TensorFieldT::value_type;
    using PlusEqualsOp      = core::PlusEquals<value_type>;
    static constexpr auto N = core::detail::tensor_field_dim_from_rank<2>(); // 6

    // One per ion population: a RefineAlgorithm (registered once, staging<-pop.M) plus a
    // per-level-number cache of single-level border-sum schedules.
    struct MTAlgo
    {
        SAMRAI::xfer::RefineSchedule&
        getOrCreateSchedule(std::shared_ptr<SAMRAI::hier::PatchLevel> const& level)
        {
            int const ilvl = level->getLevelNumber();
            if (not schedules.count(ilvl))
                schedules.try_emplace(
                    ilvl, algo->createSchedule(
                              level, 0,
                              std::make_shared<
                                  FieldBorderOpTransactionFactory<TensorFieldData_t, PlusEqualsOp>>()));
            return *schedules[ilvl];
        }

        std::unique_ptr<SAMRAI::xfer::RefineAlgorithm> algo
            = std::make_unique<SAMRAI::xfer::RefineAlgorithm>();
        std::map<int, std::shared_ptr<SAMRAI::xfer::RefineSchedule>> schedules; // keyed by level#
    };

public:
    // Register one border-sum algorithm per population: (idDst=staging, idSrc=pop.M,
    // scratch=idDst). Mirrors diag declareMomentumTensorAlgos (220-240). Schedules are
    // created lazily (the hierarchy has no levels yet at registration).
    void declareAlgos(Ions& ions, ResourcesManager& rm)
    {
        algos_.clear();
        auto const dst_name = staging_.name();
        for (auto& pop : ions)
        {
            auto& mtAlgo          = algos_.emplace_back();
            auto const src_name   = pop.momentumTensor().name();
            auto&& [idDst, idSrc] = rm.getIDsList(dst_name, src_name);
            mtAlgo.algo->registerRefine(
                idDst, idSrc, idDst, nullptr,
                std::make_shared<TensorFieldGhostInterpOverlapFillPattern<GridLayout, /*rank_=*/2>>());
        }
    }

    // Full "complete momentum tensor on one level" sequence:
    //   (1) per-patch deposit per pop (zero + interpolate domain + levelGhostOld),
    //   (2) per-pop copy-in -> fillData(fillTime) -> copy-out (border seam-sum),
    //   (3) per-patch ions.computeFullMomentumTensor() aggregate.
    // fillData is an MPI-collective per-level op: it runs once per population between the
    // deposit pass and the aggregate, never inside a per-patch loop.
    void fillCompleteMomentumTensor(Ions& ions,
                                    std::shared_ptr<SAMRAI::hier::PatchLevel> const& level,
                                    ResourcesManager& rm, double fillTime)
    {
        auto& lvl = *level;

        // (1) deposit partial per-population tensors from particles
        for (auto patch : rm.enumerate(lvl, ions))
        {
            auto const& layout = layoutFromPatch<GridLayout>(*patch);
            for (auto& pop : ions)
            {
                auto& popMT = pop.momentumTensor();
                popMT.zero();
                interpolator_(pop.domainParticles(), popMT, layout, pop.mass());
                interpolator_(pop.levelGhostParticlesOld(), popMT, layout, pop.mass());
            }
        }

        // (2) per-population border completion through the staging buffer
        for (std::size_t popidx = 0; popidx < algos_.size(); ++popidx)
        {
            for (auto patch : rm.enumerate(lvl, ions, staging_))
                for (std::uint8_t c = 0; c < N; ++c)
                    std::memcpy(staging_[c].data(), ions[popidx].momentumTensor()[c].data(),
                                ions[popidx].momentumTensor()[c].size() * sizeof(value_type));

            algos_[popidx].getOrCreateSchedule(level).fillData(fillTime);

            for (auto patch : rm.enumerate(lvl, ions, staging_))
                for (std::uint8_t c = 0; c < N; ++c)
                    std::memcpy(ions[popidx].momentumTensor()[c].data(), staging_[c].data(),
                                ions[popidx].momentumTensor()[c].size() * sizeof(value_type));
        }

        // (3) aggregate the now border-complete per-pop tensors into ions.momentumTensor()
        for (auto patch : rm.enumerate(lvl, ions))
            ions.computeFullMomentumTensor();
    }

    void onRegrid()
    {
        for (auto& a : algos_)
            a.schedules.clear();
    }

    // for solver registerResources (non-const) / allocate (const)
    TensorFieldT& stagingField() { return staging_; }
    TensorFieldT const& stagingField() const { return staging_; }

private:
    std::vector<MTAlgo> algos_;
    core::MomentumTensorInterpolator<dimension, interp_order> interpolator_;
    TensorFieldT staging_{"PHARE_fluxMomTensorSum", core::PhysicalQuantity::Tensor::M};
};

} // namespace PHARE::amr

#endif
