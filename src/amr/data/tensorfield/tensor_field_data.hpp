#ifndef PHARE_SRC_AMR_TENSORFIELD_TENSORFIELD_DATA_HPP
#define PHARE_SRC_AMR_TENSORFIELD_TENSORFIELD_DATA_HPP

#include "amr/data/field/field_geometry.hpp"
#include "amr/data/tensorfield/tensor_field_overlap.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "core/logger.hpp"
#include "core/data/field/field_box.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/tensorfield/tensorfield.hpp"


#include "amr/data/field/field_overlap.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/data/tensorfield/tensor_field_geometry.hpp"

#include <SAMRAI/hier/PatchData.h>
#include <SAMRAI/tbox/MemoryUtilities.h>

#include <amr/data/field/field_data.hpp>
#include <core/utilities/types.hpp>
#include <optional>
#include <type_traits>


namespace PHARE::amr
{
// We use another class here so that we can specialize specifics function: copy , pack , unpack
// on the dimension and we don't want to loose non specialized function related to SAMRAI
// interface
template<typename GridLayoutT, std::size_t dim, typename Grid_t, typename PhysicalQuantity>
class TensorFieldDataInternals
{
};

/**
 * @brief TensorFieldData is the specialization of SAMRAI::hier::PatchData to Field objects
 */
template<std::size_t rank, typename GridLayoutT, typename Grid_t, typename PhysicalQuantity>
class TensorFieldData : public SAMRAI::hier::PatchData
{
    using This        = TensorFieldData<rank, GridLayoutT, Grid_t, PhysicalQuantity>;
    using Super       = SAMRAI::hier::PatchData;
    using FieldData_t = FieldData<GridLayoutT, Grid_t, PhysicalQuantity>;

    static constexpr auto NO_ROTATE = SAMRAI::hier::Transformation::NO_ROTATE;

    using tensor_t             = typename PhysicalQuantity::template TensorType<rank>;
    using TensorFieldOverlap_t = TensorFieldOverlap<rank>;

    template<typename ComponentNames, typename GridLayout>
    auto static make_field_data_array(ComponentNames const& compNames, GridLayout const& layout,
                                      tensor_t qty, SAMRAI::hier::Box const& domain,
                                      SAMRAI::hier::IntVector const& ghost)
    {
        auto qts = PhysicalQuantity::componentsQuantities(qty);
        return core::for_N<N, core::for_N_R_mode::make_array>([&](auto i) {
            return std::make_shared<FieldData_t>(domain, ghost, compNames[i], layout, qts[i]);
        });
    }

    auto make_grids()
    {
        return core::for_N<N, core::for_N_R_mode::make_array>(
            [&](auto c) { return components_[c]->field; });
    }

    using value_type = typename Grid_t::value_type;
    using SetEqualOp = core::Equals<value_type>;

public:
    static constexpr std::size_t dimension    = GridLayoutT::dimension;
    static constexpr std::size_t interp_order = GridLayoutT::interp_order;
    static constexpr auto N                   = core::detail::tensor_field_dim_from_rank<rank>();

    using Geometry        = TensorFieldGeometry<rank, GridLayoutT, PhysicalQuantity>;
    using gridlayout_type = GridLayoutT;

    /*** \brief Construct a TensorFieldData from information associated to a patch
     *
     * It will create FieldData instances for each tensor component
     */
    TensorFieldData(SAMRAI::hier::Box const& domain, SAMRAI::hier::IntVector const& ghost,
                    std::string name, GridLayoutT const& layout, tensor_t qty)
        : SAMRAI::hier::PatchData(domain, ghost)
        , components_(make_field_data_array(core::detail::tensor_field_names<rank>(name), layout,
                                            qty, domain, ghost))
        , grids_{make_grids()}
    {
    }

    TensorFieldData()                                  = delete;
    TensorFieldData(TensorFieldData const&)            = delete;
    TensorFieldData(TensorFieldData&&)                 = default;
    TensorFieldData& operator=(TensorFieldData const&) = delete;

    void getFromRestart(std::shared_ptr<SAMRAI::tbox::Database> const& restart_db) override
    {
        Super::getFromRestart(restart_db);

        for (std::uint16_t c = 0; c < N; ++c)
        {
            components_[c]->getFromRestart(restart_db);
        }
    }

    void putToRestart(std::shared_ptr<SAMRAI::tbox::Database> const& restart_db) const override
    {
        Super::putToRestart(restart_db);

        for (std::uint16_t c = 0; c < N; ++c)
        {
            components_[c]->putToRestart(restart_db);
        }
    }

    /*** \brief Copy information from another TensorFieldData where data overlap
     */
    void copy(const SAMRAI::hier::PatchData& source) final
    {
        PHARE_LOG_SCOPE(3, "TensorFieldData::copy");

        TBOX_ASSERT_OBJDIM_EQUALITY2(*this, source);

        // throws on failure
        auto& fieldSource = dynamic_cast<TensorFieldData const&>(source);

        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->copy(*fieldSource.components_[c]);
        }
    }

    /*** \brief This form should not be called since we cannot derive from TensorFieldData
     */
    void copy2([[maybe_unused]] SAMRAI::hier::PatchData& destination) const final
    {
        throw std::runtime_error("Error cannot cast the PatchData to TensorFieldData");
    }

    /*** \brief Copy data from the source into the destination using the designated overlap
     */
    void copy(const SAMRAI::hier::PatchData& source, const SAMRAI::hier::BoxOverlap& overlap) final
    {
        PHARE_LOG_SCOPE(3, "TensorFieldData::copy");

        // casts throw on failure
        auto& fieldSource  = dynamic_cast<TensorFieldData const&>(source);
        auto& fieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        copy_(fieldSource, fieldOverlap);
    }

    void copy2([[maybe_unused]] SAMRAI::hier::PatchData& destination,
               [[maybe_unused]] const SAMRAI::hier::BoxOverlap& overlap) const final
    {
        throw std::runtime_error("Error cannot cast the PatchData to TensorFieldData");
    }

    bool canEstimateStreamSizeFromBox() const final
    {
        return components_[0]->canEstimateStreamSizeFromBox();
    }

    std::size_t getDataStreamSize(const SAMRAI::hier::BoxOverlap& overlap) const final
    {
        auto& tFieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        std::size_t totalSize = 0;
        for (std::size_t c = 0; c < N; ++c)
        {
            totalSize += components_[c]->getDataStreamSize(*tFieldOverlap[c]);
        }
        return totalSize;
    }

    void packStream(SAMRAI::tbox::MessageStream& stream,
                    const SAMRAI::hier::BoxOverlap& overlap) const final
    {
        PHARE_LOG_SCOPE(3, "packStream");

        auto& tFieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->packStream(stream, *tFieldOverlap[c]);
        }
    }

    void unpackStream(SAMRAI::tbox::MessageStream& stream,
                      const SAMRAI::hier::BoxOverlap& overlap) final
    {
        PHARE_LOG_SCOPE(3, "unpackStream");

        auto& tFieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->unpackStream(stream, *tFieldOverlap[c]);
        }
    }

    template<typename Operator = SetEqualOp>
    void unpackStream(SAMRAI::tbox::MessageStream& stream, const SAMRAI::hier::BoxOverlap& overlap,
                      auto& dst_grids)
    {
        PHARE_LOG_SCOPE(3, "unpackStream");

        auto& tFieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->template unpackStream<Operator>(stream, *tFieldOverlap[c],
                                                            dst_grids[c]);
        }
    }

    auto* getPointer() { return &grids_; }

    static GridLayoutT const& getLayout(SAMRAI::hier::Patch const& patch, int id)
    {
        auto const& patchData = std::dynamic_pointer_cast<This>(patch.getPatchData(id));
        if (!patchData)
            throw std::runtime_error("cannot cast to TensorFieldData");
        return patchData->gridLayout;
    }

    static auto getFields(SAMRAI::hier::Patch const& patch, int const id)
    {
        auto const& patchData = std::dynamic_pointer_cast<This>(patch.getPatchData(id));
        if (!patchData)
            throw std::runtime_error("cannot cast to TensorFieldData");

        std::array<std::reference_wrapper<Grid_t>, N> fields;
        for (std::size_t c = 0; c < N; ++c)
        {
            fields[c] = std::ref(patchData->components_[c]->field);
        }
        return fields;
    }

    void sum(SAMRAI::hier::PatchData const& src, SAMRAI::hier::BoxOverlap const& overlap)
    {
        auto& fieldSource   = dynamic_cast<TensorFieldData const&>(src);
        auto& tFieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->sum(*fieldSource.components_[c], *tFieldOverlap[c]);
        }
    }

    void unpackStreamAndSum(SAMRAI::tbox::MessageStream& stream,
                            SAMRAI::hier::BoxOverlap const& overlap)
    {
        auto& tFieldOverlap = dynamic_cast<TensorFieldOverlap_t const&>(overlap);

        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->unpackStreamAndSum(stream, *tFieldOverlap[c]);
        }
    }

    GridLayoutT gridLayout;
    std::array<std::shared_ptr<FieldData_t>, N> components_;
    std::array<Grid_t, N> grids_;

private:
    tensor_t quantity_; ///! PhysicalQuantity used for this field data

    void copy_(TensorFieldData const& source, TensorFieldOverlap_t const& overlaps)
    {
        for (std::size_t c = 0; c < N; ++c)
        {
            components_[c]->copy(*source.components_[c], *overlaps[c]);
        }
    }
};

} // namespace PHARE::amr


#endif
