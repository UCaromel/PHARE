#ifndef PHARE_MHD_HYBRID_REFLUX_COMMS_HPP
#define PHARE_MHD_HYBRID_REFLUX_COMMS_HPP

#include "amr/data/field/coarsening/electric_field_coarsener.hpp"
#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/mhd_flux_coarsener.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/messengers/hybrid_messenger_info.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "core/physical_quantities.hpp"

#include "SAMRAI/hier/CoarsenOperator.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/RefineOperator.h"

#include <memory>
#include <stdexcept>

namespace PHARE::amr
{

// MHDHybridRefluxComms owns the four RefluxChannels (E, HydroX, HydroY, HydroZ) and the
// coarsening operators exclusive to the MHD-Hybrid reflux path.
// Source IDs come from hybRm (Hybrid flux sums); destination IDs come from mhdRm (MHD timeFluxes).
// Shared same-type operators (EfieldRefineOp, mhdFluxRefineOp, mhdVecFluxRefineOp) are passed
// as parameters, matching the pattern of MHDMHDRefluxComms.
template<typename MHDModel, typename HybridModel>
struct MHDHybridRefluxComms
{
    static constexpr auto dimension              = MHDModel::dimension;
    static constexpr std::size_t rootLevelNumber = 0;

    using GridLayoutT = typename MHDModel::gridlayout_type;
    using Grid_t      = typename MHDModel::grid_type;
    using FieldT      = typename MHDModel::field_type;

    using MHDResourcesManagerT    = typename MHDModel::resources_manager_type;
    using HybridResourcesManagerT = typename HybridModel::resources_manager_type;

    using RefOp_ptr     = std::shared_ptr<SAMRAI::hier::RefineOperator>;
    using CoarsenOp_ptr = std::shared_ptr<SAMRAI::hier::CoarsenOperator>;
    using TFfillPattern = TensorFieldFillPattern<MHDModel::dimension>;

    // Coarsen ops: Hybrid flux sums → MHD timeFluxes
    using ScalarFluxCoarsenOp = FieldCoarsenOperator<GridLayoutT, FieldT, MHDFluxCoarsener<dimension>>;
    using VecFluxCoarsenOp   = TensorFieldCoarsenOperator<1, GridLayoutT, Grid_t,
                                                           MHDFluxCoarsener<dimension>,
                                                           core::PhysicalQuantity>;
    using ElectricCoarsenOp  = TensorFieldCoarsenOperator<1, GridLayoutT, Grid_t,
                                                           ElectricFieldCoarsener<dimension>,
                                                           core::PhysicalQuantity>;

    // 4 channels: E + HydroX + HydroY + HydroZ
    RefluxChannel refluxE_{dimension};
    RefluxChannel refluxHydroX_{dimension};
    RefluxChannel refluxHydroY_{dimension};
    RefluxChannel refluxHydroZ_{dimension};

    // Owned coarsen ops (exclusive to this reflux path)
    CoarsenOp_ptr hybridScalarFluxCoarsenOp_{std::make_shared<ScalarFluxCoarsenOp>()};
    CoarsenOp_ptr hybridVecFluxCoarsenOp_{std::make_shared<VecFluxCoarsenOp>()};
    CoarsenOp_ptr hybridElectricCoarsenOp_{std::make_shared<ElectricCoarsenOp>()};

    // called from MHDHybridMessengerStrategy::registerQuantities
    // mhdRm provides destination IDs (MHD timeFluxes); hybRm provides source IDs (Hybrid flux sums)
    // Shared same-type MHD refine ops for ghost fills after coarsening are passed as params.
    void registerQuantities(MHDMessengerInfo const& dstInfo,
                            HybridMessengerInfo const& srcInfo,
                            MHDResourcesManagerT& mhdRm,
                            HybridResourcesManagerT& hybRm,
                            RefOp_ptr mhdERefineOp,
                            RefOp_ptr mhdFluxRefineOp,
                            RefOp_ptr mhdVecFluxRefineOp,
                            std::shared_ptr<TFfillPattern> fillPattern)
    {
        // X-direction hydro fluxes
        auto rho_fx_dst  = mhdRm.getID(dstInfo.reflux.rho_fx);
        auto rhoV_fx_dst = mhdRm.getID(dstInfo.reflux.rhoV_fx);
        auto Etot_fx_dst = mhdRm.getID(dstInfo.reflux.Etot_fx);
        if (!rho_fx_dst or !rhoV_fx_dst or !Etot_fx_dst)
            throw std::runtime_error(
                "MHDHybridRefluxComms: missing MHD reflux IDs for x fluxes");
        auto rho_fx_src  = hybRm.getID(srcInfo.fluxSumRho_fx);
        auto rhoV_fx_src = hybRm.getID(srcInfo.fluxSumRhoV_fx);
        auto Etot_fx_src = hybRm.getID(srcInfo.fluxSumEtot_fx);
        if (!rho_fx_src or !rhoV_fx_src or !Etot_fx_src)
            throw std::runtime_error(
                "MHDHybridRefluxComms: missing Hybrid flux sum IDs for x fluxes");
        registerHydroChannel(refluxHydroX_,
                             HydroChannelSpec{*rho_fx_dst, *rhoV_fx_dst, *Etot_fx_dst,
                                              *rho_fx_src, *rhoV_fx_src, *Etot_fx_src,
                                              hybridScalarFluxCoarsenOp_, hybridVecFluxCoarsenOp_,
                                              mhdFluxRefineOp, mhdVecFluxRefineOp, fillPattern});

        if constexpr (dimension >= 2)
        {
            auto rho_fy_dst  = mhdRm.getID(dstInfo.reflux.rho_fy);
            auto rhoV_fy_dst = mhdRm.getID(dstInfo.reflux.rhoV_fy);
            auto Etot_fy_dst = mhdRm.getID(dstInfo.reflux.Etot_fy);
            if (!rho_fy_dst or !rhoV_fy_dst or !Etot_fy_dst)
                throw std::runtime_error(
                    "MHDHybridRefluxComms: missing MHD reflux IDs for y fluxes");
            auto rho_fy_src  = hybRm.getID(srcInfo.fluxSumRho_fy);
            auto rhoV_fy_src = hybRm.getID(srcInfo.fluxSumRhoV_fy);
            auto Etot_fy_src = hybRm.getID(srcInfo.fluxSumEtot_fy);
            if (!rho_fy_src or !rhoV_fy_src or !Etot_fy_src)
                throw std::runtime_error(
                    "MHDHybridRefluxComms: missing Hybrid flux sum IDs for y fluxes");
            registerHydroChannel(refluxHydroY_,
                                 HydroChannelSpec{*rho_fy_dst, *rhoV_fy_dst, *Etot_fy_dst,
                                                  *rho_fy_src, *rhoV_fy_src, *Etot_fy_src,
                                                  hybridScalarFluxCoarsenOp_, hybridVecFluxCoarsenOp_,
                                                  mhdFluxRefineOp, mhdVecFluxRefineOp, fillPattern});

            if constexpr (dimension == 3)
            {
                auto rho_fz_dst  = mhdRm.getID(dstInfo.reflux.rho_fz);
                auto rhoV_fz_dst = mhdRm.getID(dstInfo.reflux.rhoV_fz);
                auto Etot_fz_dst = mhdRm.getID(dstInfo.reflux.Etot_fz);
                if (!rho_fz_dst or !rhoV_fz_dst or !Etot_fz_dst)
                    throw std::runtime_error(
                        "MHDHybridRefluxComms: missing MHD reflux IDs for z fluxes");
                auto rho_fz_src  = hybRm.getID(srcInfo.fluxSumRho_fz);
                auto rhoV_fz_src = hybRm.getID(srcInfo.fluxSumRhoV_fz);
                auto Etot_fz_src = hybRm.getID(srcInfo.fluxSumEtot_fz);
                if (!rho_fz_src or !rhoV_fz_src or !Etot_fz_src)
                    throw std::runtime_error(
                        "MHDHybridRefluxComms: missing Hybrid flux sum IDs for z fluxes");
                registerHydroChannel(
                    refluxHydroZ_,
                    HydroChannelSpec{*rho_fz_dst, *rhoV_fz_dst, *Etot_fz_dst,
                                     *rho_fz_src, *rhoV_fz_src, *Etot_fz_src,
                                     hybridScalarFluxCoarsenOp_, hybridVecFluxCoarsenOp_,
                                     mhdFluxRefineOp, mhdVecFluxRefineOp, fillPattern});
            }
        }

        // Electric field reflux (VecField coarsen: Hybrid E → MHD E)
        auto e_dst = mhdRm.getID(dstInfo.refluxElectric);
        auto e_src = hybRm.getID(srcInfo.fluxSumElectric);
        if (!e_dst or !e_src)
            throw std::runtime_error(
                "MHDHybridRefluxComms: missing electric reflux IDs");
        refluxE_.coarsenAlgo.registerCoarsen(*e_dst, *e_src, hybridElectricCoarsenOp_);
        refluxE_.refineAlgo.registerRefine(*e_dst, *e_dst, *e_dst, mhdERefineOp, fillPattern);
    }

    // called from MHDHybridMessengerStrategy::registerLevel
    void registerLevel(int levelNumber,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy)
    {
        auto const level = hierarchy->getPatchLevel(levelNumber);
        for (auto* ch : {&refluxE_, &refluxHydroX_, &refluxHydroY_, &refluxHydroZ_})
            ch->registerLevel(hierarchy, level, levelNumber, rootLevelNumber);
    }

    // called from MHDHybridMessengerStrategy::reflux
    void reflux(int fineLevelNumber, int coarserLevelNumber, double syncTime)
    {
        for (auto* ch : {&refluxE_, &refluxHydroX_, &refluxHydroY_, &refluxHydroZ_})
            ch->reflux(fineLevelNumber, coarserLevelNumber, syncTime);
    }
};

} // namespace PHARE::amr
#endif
