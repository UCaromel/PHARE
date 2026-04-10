#ifndef PHARE_FIELD_DATA_COARSEN_HPP
#define PHARE_FIELD_DATA_COARSEN_HPP

#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "amr/data/field/field_data.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "amr/data/field/field_geometry.hpp"
#include "amr/data/tensorfield/tensor_field_data.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>
#include <SAMRAI/hier/CoarsenOperator.h>


namespace PHARE::amr
{


template<typename Dst>
void coarsen_field(Dst& destinationField, auto& sourceField, auto& intersectionBox, auto& coarsener)
{
    for (auto const bix : phare_box_from<Dst::dimension>(intersectionBox))
        coarsener(sourceField, destinationField, bix);
}


} // namespace PHARE::amr


namespace PHARE
{
namespace amr
{

    //
    template<typename GridLayoutT, typename FieldT, typename FieldCoarsenerPolicy,
             typename PhysicalQuantity = decltype(std::declval<FieldT>().physicalQuantity())>
    /**
     * @brief The FieldCoarsenOperator class
     */
    class FieldCoarsenOperator : public SAMRAI::hier::CoarsenOperator
    {
    public:
        static constexpr std::size_t dimension = GridLayoutT::dimension;
        using FieldDataT                       = FieldData<GridLayoutT, FieldT>;

        FieldCoarsenOperator()
            : SAMRAI::hier::CoarsenOperator("FieldDataCoarsenOperator")
        {
        }

        FieldCoarsenOperator(FieldCoarsenOperator const&)            = delete;
        FieldCoarsenOperator(FieldCoarsenOperator&&)                 = delete;
        FieldCoarsenOperator& operator=(FieldCoarsenOperator const&) = delete;
        FieldCoarsenOperator&& operator=(FieldCoarsenOperator&&)     = delete;


        virtual ~FieldCoarsenOperator() = default;




        /** @brief return the priority of the operator
         *  this return 0, meaning that this operator
         * have the most priority
         */
        int getOperatorPriority() const override { return 0; }




        /** @brief Return the stencil width associated with the coarsening operator.
         *
         *  The SAMRAI transfer routines guarantee that the source patch will contain
         * sufficient ghostCell data surrounding the interior to satisfy the stencil
         * width requirements for each coarsening operator.
         *
         */
        SAMRAI::hier::IntVector getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
        {
            return SAMRAI::hier::IntVector{dim, 2};
        }




        /** @brief given a coarseBox, coarse data from the fine patch on the intersection of
         * this box and the box of the destination (the box of the coarse patch).
         *
         * This method will extract fieldData from the two patches, and then
         * get the Field and GridLayout encapsulated into the fieldData.
         * With the help of FieldGeometry, transform the coarseBox to the correct index.
         * After that we can now create FieldCoarsen with the indexAndWeight implementation
         * selected. Finally loop over the indexes in the box, and apply the coarsening defined
         * in FieldCoarsen operator
         *
         */
        void coarsen(SAMRAI::hier::Patch& destinationPatch, SAMRAI::hier::Patch const& sourcePatch,
                     int const destinationId, int const sourceId,
                     SAMRAI::hier::Box const& coarseBox,
                     SAMRAI::hier::IntVector const& ratio) const override
        {
            auto& destinationField   = FieldDataT::getField(destinationPatch, destinationId);
            auto const& sourceField  = FieldDataT::getField(sourcePatch, sourceId);
            auto const& sourceLayout = FieldDataT::getLayout(sourcePatch, sourceId);
            auto const& destLayout   = FieldDataT::getLayout(destinationPatch, destinationId);
            using FieldGeometryT     = FieldGeometry<GridLayoutT, PhysicalQuantity>;

            // we assume that quantity are the same
            // note that an assertion will be raised
            // in coarseIt operator
            auto const& qty = destinationField.physicalQuantity();

            // We get different boxes : destination , source, restrictBoxes
            // and transform them in the correct indexing.
            auto destPData = destinationPatch.getPatchData(destinationId);
            auto srcPData  = sourcePatch.getPatchData(sourceId);
            auto destGBox  = FieldGeometryT::toFieldBox(destPData->getGhostBox(), qty, destLayout);
            auto srcGBox   = FieldGeometryT::toFieldBox(srcPData->getGhostBox(), qty, sourceLayout);
            auto coarseLayout          = FieldGeometryT::layoutFromBox(coarseBox, destLayout);
            auto coarseFieldBox        = FieldGeometryT::toFieldBox(coarseBox, qty, coarseLayout);
            auto const intersectionBox = destGBox * coarseFieldBox;
            // We can now create the coarsening operator
            FieldCoarsenerPolicy coarsener{destLayout.centering(qty), srcGBox, destGBox, ratio};

            coarsen_field(destinationField, sourceField, intersectionBox, coarsener);
        }
    };
} // namespace amr
} // namespace PHARE


namespace PHARE::amr
{


template<std::size_t rank, typename GridLayoutT, typename FieldT, typename FieldCoarsenerPolicy,
         typename PhysicalQuantity>
class TensorFieldCoarsenOperator : public SAMRAI::hier::CoarsenOperator
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    using TensorFieldDataT = TensorFieldData<rank, GridLayoutT, FieldT, PhysicalQuantity>;
    using FieldDataT       = FieldData<GridLayoutT, FieldT>;

    static constexpr std::size_t N = TensorFieldDataT::N;

    TensorFieldCoarsenOperator()
        : SAMRAI::hier::CoarsenOperator("FieldDataCoarsenOperator")
    {
    }

    TensorFieldCoarsenOperator(TensorFieldCoarsenOperator const&)            = delete;
    TensorFieldCoarsenOperator(TensorFieldCoarsenOperator&&)                 = delete;
    TensorFieldCoarsenOperator& operator=(TensorFieldCoarsenOperator const&) = delete;
    TensorFieldCoarsenOperator&& operator=(TensorFieldCoarsenOperator&&)     = delete;


    virtual ~TensorFieldCoarsenOperator() = default;




    /** @brief return the priority of the operator
     *  this return 0, meaning that this operator have the most priority
     */
    int getOperatorPriority() const override { return 0; }




    /** @brief Return the stencil width associated with the coarsening operator.
     *
     *  The SAMRAI transfer routines guarantee that the source patch will contain
     * sufficient ghostCell data surrounding the interior to satisfy the stencil
     * width requirements for each coarsening operator.
     *
     * In our case, we allow a RF up to 10, so having 5 ghost width is sufficient
     */
    SAMRAI::hier::IntVector getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector{dim, 2};
    }




    /** @brief given a coarseBox, coarse data from the fine patch on the intersection of
     * this box and the box of the destination (the box of the coarse patch).
     *
     * This method will extract fieldData from the two patches, and then
     * get the Field and GridLayout encapsulated into the fieldData.
     * With the help of FieldGeometry, transform the coarseBox to the correct index.
     * After that we can now create FieldCoarsen with the indexAndWeight implementation
     * selected. Finnaly loop over the indexes in the box, and apply the coarsening defined
     * in FieldCoarsen operator
     *
     */
    void coarsen(SAMRAI::hier::Patch& destinationPatch, SAMRAI::hier::Patch const& sourcePatch,
                 int const destinationId, int const sourceId, SAMRAI::hier::Box const& coarseBox,
                 SAMRAI::hier::IntVector const& ratio) const override
    {
        auto& destinationFields  = TensorFieldDataT::getFields(destinationPatch, destinationId);
        auto const& sourceFields = TensorFieldDataT::getFields(sourcePatch, sourceId);
        auto const& sourceLayout = TensorFieldDataT::getLayout(sourcePatch, sourceId);
        auto const& destLayout   = TensorFieldDataT::getLayout(destinationPatch, destinationId);


        // we assume that quantity are the same
        // note that an assertion will be raised in coarseIt operator

        for (std::uint16_t c = 0; c < N; ++c)
        {
            auto const& qty      = destinationFields[c].physicalQuantity();
            using FieldGeometryT = FieldGeometry<GridLayoutT, std::decay_t<decltype(qty)>>;


            // We get different boxes : destination , source, restrictBoxes
            // and transform them in the correct indexing.
            auto const& destPData = destinationPatch.getPatchData(destinationId);
            auto const& srcPData  = sourcePatch.getPatchData(sourceId);
            auto const& destGBox
                = FieldGeometryT::toFieldBox(destPData->getGhostBox(), qty, destLayout);
            auto const& srcGBox
                = FieldGeometryT::toFieldBox(srcPData->getGhostBox(), qty, sourceLayout);
            auto const& coarseLayout   = FieldGeometryT::layoutFromBox(coarseBox, destLayout);
            auto const& coarseFieldBox = FieldGeometryT::toFieldBox(coarseBox, qty, coarseLayout);
            auto const intersectionBox = destGBox * coarseFieldBox;
            // We can now create the coarsening operator
            FieldCoarsenerPolicy coarsener{destLayout.centering(qty), srcGBox, destGBox, ratio};

            coarsen_field(destinationFields[c], sourceFields[c], intersectionBox, coarsener);
        }
    }
};

template<typename GridLayoutT, typename FieldT, typename FieldCoarsenerPolicy,
         typename PhysicalQuantity>
using VecFieldCoarsenOperator = TensorFieldCoarsenOperator</*rank=*/1, GridLayoutT, FieldT,
                                                           FieldCoarsenerPolicy, PhysicalQuantity>;


// Cross-type variant of TensorFieldCoarsenOperator for coarsening between different field type
// systems (e.g. Hybrid flux sums → MHD flux receivers). Source and destination share the same
// spatial face centering but use different Grid_t/PhysicalQuantity enums.
template<std::size_t rank,
         typename SrcGridLayoutT, typename SrcGrid_t, typename SrcPhysQty,
         typename DstGridLayoutT, typename DstGrid_t, typename DstPhysQty,
         typename FieldCoarsenerPolicy>
class CrossTypeTensorFieldCoarsenOperator : public SAMRAI::hier::CoarsenOperator
{
public:
    static constexpr std::size_t dimension = DstGridLayoutT::dimension;
    using SrcTFDataT = TensorFieldData<rank, SrcGridLayoutT, SrcGrid_t, SrcPhysQty>;
    using DstTFDataT = TensorFieldData<rank, DstGridLayoutT, DstGrid_t, DstPhysQty>;
    static constexpr std::size_t N = DstTFDataT::N;

    CrossTypeTensorFieldCoarsenOperator()
        : SAMRAI::hier::CoarsenOperator("CrossTypeTensorFieldCoarsenOperator")
    {
    }

    CrossTypeTensorFieldCoarsenOperator(CrossTypeTensorFieldCoarsenOperator const&)            = delete;
    CrossTypeTensorFieldCoarsenOperator(CrossTypeTensorFieldCoarsenOperator&&)                 = delete;
    CrossTypeTensorFieldCoarsenOperator& operator=(CrossTypeTensorFieldCoarsenOperator const&) = delete;
    CrossTypeTensorFieldCoarsenOperator& operator=(CrossTypeTensorFieldCoarsenOperator&&)      = delete;

    virtual ~CrossTypeTensorFieldCoarsenOperator() = default;

    int getOperatorPriority() const override { return 0; }

    SAMRAI::hier::IntVector getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector{dim, 2};
    }

    void coarsen(SAMRAI::hier::Patch& destinationPatch, SAMRAI::hier::Patch const& sourcePatch,
                 int const destinationId, int const sourceId,
                 SAMRAI::hier::Box const& coarseBox,
                 SAMRAI::hier::IntVector const& ratio) const override
    {
        auto& destinationFields  = DstTFDataT::getFields(destinationPatch, destinationId);
        auto const& sourceFields = SrcTFDataT::getFields(sourcePatch, sourceId);
        auto const& srcLayout    = SrcTFDataT::getLayout(sourcePatch, sourceId);
        auto const& dstLayout    = DstTFDataT::getLayout(destinationPatch, destinationId);

        for (std::uint16_t c = 0; c < N; ++c)
        {
            auto const& dstQty = destinationFields[c].physicalQuantity();
            auto const& srcQty = sourceFields[c].physicalQuantity();
            using DstGeomT     = FieldGeometry<DstGridLayoutT, std::decay_t<decltype(dstQty)>>;
            using SrcGeomT     = FieldGeometry<SrcGridLayoutT, std::decay_t<decltype(srcQty)>>;

            auto const& destPData   = destinationPatch.getPatchData(destinationId);
            auto const& srcPData    = sourcePatch.getPatchData(sourceId);
            auto const  destGBox    = DstGeomT::toFieldBox(destPData->getGhostBox(), dstQty, dstLayout);
            auto const  srcGBox     = SrcGeomT::toFieldBox(srcPData->getGhostBox(), srcQty, srcLayout);
            auto const  coarseLayout    = DstGeomT::layoutFromBox(coarseBox, dstLayout);
            auto const  coarseFieldBox  = DstGeomT::toFieldBox(coarseBox, dstQty, coarseLayout);
            auto const  intersectionBox = destGBox * coarseFieldBox;

            FieldCoarsenerPolicy coarsener{dstLayout.centering(dstQty), srcGBox, destGBox, ratio};
            coarsen_field(destinationFields[c], sourceFields[c], intersectionBox, coarsener);
        }
    }
};

template<typename SrcGridLayoutT, typename SrcGrid_t, typename SrcPhysQty,
         typename DstGridLayoutT, typename DstGrid_t, typename DstPhysQty,
         typename FieldCoarsenerPolicy>
using CrossTypeVecFieldCoarsenOperator
    = CrossTypeTensorFieldCoarsenOperator</*rank=*/1, SrcGridLayoutT, SrcGrid_t, SrcPhysQty,
                                          DstGridLayoutT, DstGrid_t, DstPhysQty,
                                          FieldCoarsenerPolicy>;


// Cross-type scalar coarsen operator: separate Src/Dst GridLayoutT and FieldT types.
// Used to coarsen scalar flux Fields between model boundaries (e.g. HybridQuantity→MHDQuantity).
template<typename SrcGridLayoutT, typename SrcFieldT,
         typename DstGridLayoutT, typename DstFieldT,
         typename FieldCoarsenerPolicy>
class CrossTypeScalarFieldCoarsenOperator : public SAMRAI::hier::CoarsenOperator
{
public:
    static constexpr std::size_t dimension = DstGridLayoutT::dimension;
    using SrcFieldDataT = FieldData<SrcGridLayoutT, SrcFieldT>;
    using DstFieldDataT = FieldData<DstGridLayoutT, DstFieldT>;

    CrossTypeScalarFieldCoarsenOperator()
        : SAMRAI::hier::CoarsenOperator("CrossTypeScalarFieldCoarsenOperator")
    {
    }

    CrossTypeScalarFieldCoarsenOperator(CrossTypeScalarFieldCoarsenOperator const&)            = delete;
    CrossTypeScalarFieldCoarsenOperator(CrossTypeScalarFieldCoarsenOperator&&)                 = delete;
    CrossTypeScalarFieldCoarsenOperator& operator=(CrossTypeScalarFieldCoarsenOperator const&) = delete;
    CrossTypeScalarFieldCoarsenOperator& operator=(CrossTypeScalarFieldCoarsenOperator&&)      = delete;

    virtual ~CrossTypeScalarFieldCoarsenOperator() = default;

    int getOperatorPriority() const override { return 0; }

    SAMRAI::hier::IntVector getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector{dim, 2};
    }

    void coarsen(SAMRAI::hier::Patch& destinationPatch, SAMRAI::hier::Patch const& sourcePatch,
                 int const destinationId, int const sourceId,
                 SAMRAI::hier::Box const& coarseBox,
                 SAMRAI::hier::IntVector const& ratio) const override
    {
        auto& destinationField  = DstFieldDataT::getField(destinationPatch, destinationId);
        auto const& sourceField = SrcFieldDataT::getField(sourcePatch, sourceId);
        auto const& srcLayout   = SrcFieldDataT::getLayout(sourcePatch, sourceId);
        auto const& dstLayout   = DstFieldDataT::getLayout(destinationPatch, destinationId);

        auto const& dstQty = destinationField.physicalQuantity();
        auto const& srcQty = sourceField.physicalQuantity();
        using DstQty       = std::decay_t<decltype(dstQty)>;
        using SrcQty       = std::decay_t<decltype(srcQty)>;
        using DstGeomT     = FieldGeometry<DstGridLayoutT, DstQty>;
        using SrcGeomT     = FieldGeometry<SrcGridLayoutT, SrcQty>;

        auto const destPData = destinationPatch.getPatchData(destinationId);
        auto const srcPData  = sourcePatch.getPatchData(sourceId);
        auto const destGBox  = DstGeomT::toFieldBox(destPData->getGhostBox(), dstQty, dstLayout);
        auto const srcGBox   = SrcGeomT::toFieldBox(srcPData->getGhostBox(), srcQty, srcLayout);

        auto const coarseLayout    = DstGeomT::layoutFromBox(coarseBox, dstLayout);
        auto const coarseFieldBox  = DstGeomT::toFieldBox(coarseBox, dstQty, coarseLayout);
        auto const intersectionBox = destGBox * coarseFieldBox;

        FieldCoarsenerPolicy coarsener{dstLayout.centering(dstQty), srcGBox, destGBox, ratio};
        coarsen_field(destinationField, sourceField, intersectionBox, coarsener);
    }
};


// Variant of FieldCoarsenOperator where source and destination fields have different FieldT types
// (and therefore different PhysicalQuantity enum types). Used for cross-model coarsening,
// e.g. Hybrid flux sums → MHD flux registers, where both sides share the same face centering
// but differ in their quantity enum.
template<typename GridLayoutT, typename SrcFieldT, typename DstFieldT, typename FieldCoarsenerPolicy>
class CrossTypeFieldCoarsenOperator : public SAMRAI::hier::CoarsenOperator
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    using SrcFieldDataT = FieldData<GridLayoutT, SrcFieldT>;
    using DstFieldDataT = FieldData<GridLayoutT, DstFieldT>;

    CrossTypeFieldCoarsenOperator()
        : SAMRAI::hier::CoarsenOperator("CrossTypeFieldCoarsenOperator")
    {
    }

    CrossTypeFieldCoarsenOperator(CrossTypeFieldCoarsenOperator const&)            = delete;
    CrossTypeFieldCoarsenOperator(CrossTypeFieldCoarsenOperator&&)                 = delete;
    CrossTypeFieldCoarsenOperator& operator=(CrossTypeFieldCoarsenOperator const&) = delete;
    CrossTypeFieldCoarsenOperator&& operator=(CrossTypeFieldCoarsenOperator&&)     = delete;

    virtual ~CrossTypeFieldCoarsenOperator() = default;

    int getOperatorPriority() const override { return 0; }

    SAMRAI::hier::IntVector getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector{dim, 2};
    }

    void coarsen(SAMRAI::hier::Patch& destinationPatch, SAMRAI::hier::Patch const& sourcePatch,
                 int const destinationId, int const sourceId,
                 SAMRAI::hier::Box const& coarseBox,
                 SAMRAI::hier::IntVector const& ratio) const override
    {
        auto& destinationField   = DstFieldDataT::getField(destinationPatch, destinationId);
        auto const& sourceField  = SrcFieldDataT::getField(sourcePatch, sourceId);
        auto const& sourceLayout = SrcFieldDataT::getLayout(sourcePatch, sourceId);
        auto const& destLayout   = DstFieldDataT::getLayout(destinationPatch, destinationId);

        auto const& dstQty = destinationField.physicalQuantity();
        auto const& srcQty = sourceField.physicalQuantity();
        using DstQty = std::decay_t<decltype(dstQty)>;
        using SrcQty = std::decay_t<decltype(srcQty)>;
        using DstGeomT = FieldGeometry<GridLayoutT, DstQty>;
        using SrcGeomT = FieldGeometry<GridLayoutT, SrcQty>;

        auto destPData = destinationPatch.getPatchData(destinationId);
        auto srcPData  = sourcePatch.getPatchData(sourceId);
        auto destGBox  = DstGeomT::toFieldBox(destPData->getGhostBox(), dstQty, destLayout);
        auto srcGBox   = SrcGeomT::toFieldBox(srcPData->getGhostBox(), srcQty, sourceLayout);

        auto coarseLayout   = DstGeomT::layoutFromBox(coarseBox, destLayout);
        auto coarseFieldBox = DstGeomT::toFieldBox(coarseBox, dstQty, coarseLayout);
        auto const intersectionBox = destGBox * coarseFieldBox;

        // Use destination centering to drive the coarsening policy (same face centering as source).
        FieldCoarsenerPolicy coarsener{destLayout.centering(dstQty), srcGBox, destGBox, ratio};
        coarsen_field(destinationField, sourceField, intersectionBox, coarsener);
    }
};

} // namespace PHARE::amr


#endif
