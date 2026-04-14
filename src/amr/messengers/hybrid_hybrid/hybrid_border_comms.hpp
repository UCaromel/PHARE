#ifndef PHARE_HYBRID_BORDER_COMMS_HPP
#define PHARE_HYBRID_BORDER_COMMS_HPP

#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/refiner_pool.hpp"
#include "core/utilities/types.hpp"

#include "SAMRAI/hier/PatchHierarchy.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace PHARE::amr
{

template<typename HybridModel>
struct HybridBorderComms
{
    using GridLayoutT = typename HybridModel::gridlayout_type;
    using rm_t        = typename HybridModel::resources_manager_type;

    using VecFieldGhostSumRefinerPool = RefinerPool<rm_t, RefinerType::PatchVecFieldBorderSum>;
    using FieldGhostSumRefinerPool    = RefinerPool<rm_t, RefinerType::PatchFieldBorderSum>;
    using FieldGhostMaxRefinerPool    = RefinerPool<rm_t, RefinerType::PatchFieldBorderMax>;
    using VecFieldGhostMaxRefinerPool = RefinerPool<rm_t, RefinerType::PatchVecFieldBorderMax>;

    std::shared_ptr<rm_t> resourcesManager_;
    std::vector<VecFieldGhostSumRefinerPool> popFluxBorderSumRefiners_;
    std::vector<FieldGhostSumRefinerPool> popDensityBorderSumRefiners_;
    std::vector<FieldGhostMaxRefinerPool> ionDensityBorderMaxRefiners_;
    std::vector<VecFieldGhostMaxRefinerPool> ionFluxBorderMaxRefiners_;

    explicit HybridBorderComms(std::shared_ptr<rm_t> rm)
        : resourcesManager_{std::move(rm)}
    {
    }

    void registerInfo(HybridMessengerInfo const& info,
                      std::string const& sumVecName,
                      std::string const& sumFieldName)
    {
        for (auto const& vecfield : info.ghostFlux)
            popFluxBorderSumRefiners_.emplace_back(resourcesManager_)
                .addStaticRefiner(
                    sumVecName, vecfield, nullptr, sumVecName,
                    std::make_shared<
                        TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());

        for (auto const& field : info.sumBorderFields)
            popDensityBorderSumRefiners_.emplace_back(resourcesManager_)
                .addStaticRefiner(sumFieldName, field, nullptr, sumFieldName,
                                  std::make_shared<FieldGhostInterpOverlapFillPattern<GridLayoutT>>());

        assert(info.maxBorderFields.size() == 2); // mass & charge densities
        for (auto const& field : info.maxBorderFields)
            ionDensityBorderMaxRefiners_.emplace_back(resourcesManager_)
                .addStaticRefiner(field, field, nullptr, field,
                                  std::make_shared<FieldGhostInterpOverlapFillPattern<GridLayoutT>>());

        assert(info.maxBorderVecFields.size() == 1);
        for (auto const& vecfield : info.maxBorderVecFields)
            ionFluxBorderMaxRefiners_.emplace_back(resourcesManager_)
                .addStaticRefiner(
                    vecfield, vecfield, nullptr, vecfield,
                    std::make_shared<
                        TensorFieldGhostInterpOverlapFillPattern<GridLayoutT, /*rank_=*/1>>());
    }

    void registerLevel(int levelNumber,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy)
    {
        auto const level = hierarchy->getPatchLevel(levelNumber);

        for (auto& refiner : popFluxBorderSumRefiners_)
            refiner.registerLevel(hierarchy, level);

        for (auto& refiner : popDensityBorderSumRefiners_)
            refiner.registerLevel(hierarchy, level);

        for (auto& refiner : ionFluxBorderMaxRefiners_)
            refiner.registerLevel(hierarchy, level);

        for (auto& refiner : ionDensityBorderMaxRefiners_)
            refiner.registerLevel(hierarchy, level);
    }

    template<typename IonsT, typename LevelT, typename VecFieldT>
    void fillFluxBorders(IonsT& ions, LevelT& level, VecFieldT& sumVec, double fillTime)
    {
        auto constexpr N = core::detail::tensor_field_dim_from_rank<1>();
        using value_type = typename VecFieldT::field_type::value_type;

        for (std::size_t i = 0; i < ions.size(); ++i)
        {
            for (auto patch : resourcesManager_->enumerate(level, ions, sumVec))
                for (std::uint8_t c = 0; c < N; ++c)
                    std::memcpy(sumVec[c].data(), ions[i].flux()[c].data(),
                                ions[i].flux()[c].size() * sizeof(value_type));

            popFluxBorderSumRefiners_[i].fill(level.getLevelNumber(), fillTime);

            for (auto patch : resourcesManager_->enumerate(level, ions, sumVec))
                for (std::uint8_t c = 0; c < N; ++c)
                    std::memcpy(ions[i].flux()[c].data(), sumVec[c].data(),
                                ions[i].flux()[c].size() * sizeof(value_type));
        }
    }

    template<typename IonsT, typename LevelT, typename FieldT>
    void fillDensityBorders(IonsT& ions, LevelT& level, FieldT& sumField, double fillTime)
    {
        using value_type = typename FieldT::value_type;

        assert(popDensityBorderSumRefiners_.size() % ions.size() == 0);
        std::size_t const fieldsPerPop = popDensityBorderSumRefiners_.size() / ions.size();

        for (std::size_t i = 0; i < ions.size(); ++i)
        {
            for (auto patch : resourcesManager_->enumerate(level, ions, sumField))
                std::memcpy(sumField.data(), ions[i].particleDensity().data(),
                            ions[i].particleDensity().size() * sizeof(value_type));

            popDensityBorderSumRefiners_[i * fieldsPerPop].fill(level.getLevelNumber(), fillTime);

            for (auto patch : resourcesManager_->enumerate(level, ions, sumField))
                std::memcpy(ions[i].particleDensity().data(), sumField.data(),
                            ions[i].particleDensity().size() * sizeof(value_type));

            for (auto patch : resourcesManager_->enumerate(level, ions, sumField))
                std::memcpy(sumField.data(), ions[i].chargeDensity().data(),
                            ions[i].chargeDensity().size() * sizeof(value_type));

            popDensityBorderSumRefiners_[i * fieldsPerPop + 1].fill(level.getLevelNumber(),
                                                                     fillTime);

            for (auto patch : resourcesManager_->enumerate(level, ions, sumField))
                std::memcpy(ions[i].chargeDensity().data(), sumField.data(),
                            ions[i].chargeDensity().size() * sizeof(value_type));
        }
    }

    template<typename LevelT>
    void fillIonBorders(LevelT& level, double fillTime)
    {
        assert(ionFluxBorderMaxRefiners_.size() == 1);
        assert(ionDensityBorderMaxRefiners_.size() == 2);

        for (auto& refiner : ionFluxBorderMaxRefiners_)
            refiner.fill(level.getLevelNumber(), fillTime);

        for (auto& refiner : ionDensityBorderMaxRefiners_)
            refiner.fill(level.getLevelNumber(), fillTime);
    }
};

} // namespace PHARE::amr

#endif
