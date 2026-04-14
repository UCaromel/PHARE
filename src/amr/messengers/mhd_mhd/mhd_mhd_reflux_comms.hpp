#ifndef PHARE_MHD_MHD_REFLUX_COMMS_HPP
#define PHARE_MHD_MHD_REFLUX_COMMS_HPP

#include "amr/data/field/coarsening/field_coarsen_operator.hpp"
#include "amr/data/field/coarsening/mhd_flux_coarsener.hpp"
#include "amr/data/field/field_variable_fill_pattern.hpp"
#include "amr/messengers/messenger_utils.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "core/mhd/mhd_quantities.hpp"

#include "SAMRAI/hier/CoarsenOperator.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/RefineOperator.h"

#include <memory>
#include <stdexcept>

namespace PHARE::amr
{

// MHDMHDRefluxComms owns the four RefluxChannels (E, HydroX, HydroY, HydroZ) and the two
// coarsening operators that are exclusive to reflux. Shared operators (EfieldRefineOp,
// electricFieldCoarseningOp, mhdFluxRefineOp, mhdVecFluxRefineOp) are passed as parameters.
template<typename MHDModel>
struct MHDMHDRefluxComms
{
    using GridLayoutT   = typename MHDModel::gridlayout_type;
    using GridT         = typename MHDModel::grid_type;
    using rm_t          = typename MHDModel::resources_manager_type;
    using RefOp_ptr     = std::shared_ptr<SAMRAI::hier::RefineOperator>;
    using CoarsenOp_ptr = std::shared_ptr<SAMRAI::hier::CoarsenOperator>;
    using TFfillPattern = TensorFieldFillPattern<MHDModel::dimension>;

    static constexpr auto dimension       = MHDModel::dimension;
    static constexpr std::size_t rootLevelNumber = 0;

    template<typename Policy>
    using FieldCoarseningOp = FieldCoarsenOperator<GridLayoutT, GridT, Policy>;
    template<typename Policy>
    using VecFieldCoarsenOp
        = VecFieldCoarsenOperator<GridLayoutT, GridT, Policy, core::MHDQuantity>;

    using MHDFluxCoarsenOp    = FieldCoarseningOp<MHDFluxCoarsener<dimension>>;
    using MHDVecFluxCoarsenOp = VecFieldCoarsenOp<MHDFluxCoarsener<dimension>>;

    // Four channels: one per reflux direction + E-field
    RefluxChannel refluxE_{dimension};
    RefluxChannel refluxHydroX_{dimension};
    RefluxChannel refluxHydroY_{dimension};
    RefluxChannel refluxHydroZ_{dimension};

    // Operators exclusive to reflux (not used elsewhere in the messenger)
    CoarsenOp_ptr mhdFluxCoarseningOp_{std::make_shared<MHDFluxCoarsenOp>()};
    CoarsenOp_ptr mhdVecFluxCoarseningOp_{std::make_shared<MHDVecFluxCoarsenOp>()};

    // called from MHDMessenger::registerQuantities
    // shared operators passed as params; exclusive ops (mhdFluxCoarseningOp_,
    // mhdVecFluxCoarseningOp_) are owned by this struct
    void registerQuantities(MHDMessengerInfo const& info,
                            rm_t& rm,
                            RefOp_ptr EfieldRefineOp,
                            CoarsenOp_ptr electricFieldCoarseningOp,
                            RefOp_ptr mhdFluxRefineOp,
                            RefOp_ptr mhdVecFluxRefineOp,
                            std::shared_ptr<TFfillPattern> nonOverwriteInteriorTFfillPattern)
    {
        auto rho_fx_reflux_id  = rm.getID(info.reflux.rho_fx);
        auto rhoV_fx_reflux_id = rm.getID(info.reflux.rhoV_fx);
        auto Etot_fx_reflux_id = rm.getID(info.reflux.Etot_fx);
        if (!rho_fx_reflux_id or !rhoV_fx_reflux_id or !Etot_fx_reflux_id)
            throw std::runtime_error(
                "MHDMHDRefluxComms: missing reflux variable IDs for fluxes in x direction");
        auto rho_fx_fluxsum_id  = rm.getID(info.fluxSum.rho_fx);
        auto rhoV_fx_fluxsum_id = rm.getID(info.fluxSum.rhoV_fx);
        auto Etot_fx_fluxsum_id = rm.getID(info.fluxSum.Etot_fx);
        if (!rho_fx_fluxsum_id or !rhoV_fx_fluxsum_id or !Etot_fx_fluxsum_id)
            throw std::runtime_error(
                "MHDMHDRefluxComms: missing flux sum variable IDs for fluxes in x direction");
        registerHydroChannel_(refluxHydroX_,
                              *rho_fx_reflux_id, *rhoV_fx_reflux_id, *Etot_fx_reflux_id,
                              *rho_fx_fluxsum_id, *rhoV_fx_fluxsum_id, *Etot_fx_fluxsum_id,
                              mhdFluxRefineOp, mhdVecFluxRefineOp, nonOverwriteInteriorTFfillPattern);

        if constexpr (dimension >= 2)
        {
            auto rho_fy_reflux_id  = rm.getID(info.reflux.rho_fy);
            auto rhoV_fy_reflux_id = rm.getID(info.reflux.rhoV_fy);
            auto Etot_fy_reflux_id = rm.getID(info.reflux.Etot_fy);
            if (!rho_fy_reflux_id or !rhoV_fy_reflux_id or !Etot_fy_reflux_id)
                throw std::runtime_error(
                    "MHDMHDRefluxComms: missing reflux variable IDs for fluxes in y direction");
            auto rho_fy_fluxsum_id  = rm.getID(info.fluxSum.rho_fy);
            auto rhoV_fy_fluxsum_id = rm.getID(info.fluxSum.rhoV_fy);
            auto Etot_fy_fluxsum_id = rm.getID(info.fluxSum.Etot_fy);
            if (!rho_fy_fluxsum_id or !rhoV_fy_fluxsum_id or !Etot_fy_fluxsum_id)
                throw std::runtime_error(
                    "MHDMHDRefluxComms: missing flux sum variable IDs for fluxes in y direction");
            registerHydroChannel_(refluxHydroY_,
                                  *rho_fy_reflux_id, *rhoV_fy_reflux_id, *Etot_fy_reflux_id,
                                  *rho_fy_fluxsum_id, *rhoV_fy_fluxsum_id, *Etot_fy_fluxsum_id,
                                  mhdFluxRefineOp, mhdVecFluxRefineOp, nonOverwriteInteriorTFfillPattern);

            if constexpr (dimension == 3)
            {
                auto rho_fz_reflux_id  = rm.getID(info.reflux.rho_fz);
                auto rhoV_fz_reflux_id = rm.getID(info.reflux.rhoV_fz);
                auto Etot_fz_reflux_id = rm.getID(info.reflux.Etot_fz);
                if (!rho_fz_reflux_id or !rhoV_fz_reflux_id or !Etot_fz_reflux_id)
                    throw std::runtime_error("MHDMHDRefluxComms: missing reflux variable IDs for "
                                             "fluxes in z direction");
                auto rho_fz_fluxsum_id  = rm.getID(info.fluxSum.rho_fz);
                auto rhoV_fz_fluxsum_id = rm.getID(info.fluxSum.rhoV_fz);
                auto Etot_fz_fluxsum_id = rm.getID(info.fluxSum.Etot_fz);
                if (!rho_fz_fluxsum_id or !rhoV_fz_fluxsum_id or !Etot_fz_fluxsum_id)
                    throw std::runtime_error("MHDMHDRefluxComms: missing flux sum variable IDs for "
                                             "fluxes in z direction");
                registerHydroChannel_(refluxHydroZ_,
                                      *rho_fz_reflux_id, *rhoV_fz_reflux_id, *Etot_fz_reflux_id,
                                      *rho_fz_fluxsum_id, *rhoV_fz_fluxsum_id, *Etot_fz_fluxsum_id,
                                      mhdFluxRefineOp, mhdVecFluxRefineOp, nonOverwriteInteriorTFfillPattern);
            }
        }

        auto e_reflux_id  = rm.getID(info.refluxElectric);
        auto e_fluxsum_id = rm.getID(info.fluxSumElectric);

        if (!e_reflux_id or !e_fluxsum_id)
            throw std::runtime_error(
                "MHDMHDRefluxComms: missing electric refluxing field variable IDs");

        refluxE_.coarsenAlgo.registerCoarsen(*e_reflux_id, *e_fluxsum_id,
                                             electricFieldCoarseningOp);
        refluxE_.refineAlgo.registerRefine(*e_reflux_id, *e_reflux_id, *e_reflux_id,
                                           EfieldRefineOp, nonOverwriteInteriorTFfillPattern);
    }

    // called from MHDMessenger::registerLevel
    void registerLevel(int levelNumber,
                       std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy)
    {
        auto const level = hierarchy->getPatchLevel(levelNumber);
        for (auto* ch : {&refluxE_, &refluxHydroX_, &refluxHydroY_, &refluxHydroZ_})
            ch->registerLevel(hierarchy, level, levelNumber, rootLevelNumber);
    }

    // called from MHDMessenger::reflux
    void reflux(int fineLevelNumber, int coarserLevelNumber, double syncTime)
    {
        for (auto* ch : {&refluxE_, &refluxHydroX_, &refluxHydroY_, &refluxHydroZ_})
            ch->reflux(fineLevelNumber, coarserLevelNumber, syncTime);
    }

private:
    void registerHydroChannel_(RefluxChannel& channel,
                               int rho_reflux_id,  int rhoV_reflux_id,  int Etot_reflux_id,
                               int rho_fluxsum_id, int rhoV_fluxsum_id, int Etot_fluxsum_id,
                               RefOp_ptr mhdFluxRefineOp,
                               RefOp_ptr mhdVecFluxRefineOp,
                               std::shared_ptr<TFfillPattern> fillPattern)
    {
        channel.coarsenAlgo.registerCoarsen(rho_reflux_id,  rho_fluxsum_id,  mhdFluxCoarseningOp_);
        channel.coarsenAlgo.registerCoarsen(rhoV_reflux_id, rhoV_fluxsum_id, mhdVecFluxCoarseningOp_);
        channel.coarsenAlgo.registerCoarsen(Etot_reflux_id, Etot_fluxsum_id, mhdFluxCoarseningOp_);
        channel.refineAlgo.registerRefine(rho_reflux_id,  rho_reflux_id,  rho_reflux_id,  mhdFluxRefineOp,    fillPattern);
        channel.refineAlgo.registerRefine(rhoV_reflux_id, rhoV_reflux_id, rhoV_reflux_id, mhdVecFluxRefineOp, fillPattern);
        channel.refineAlgo.registerRefine(Etot_reflux_id, Etot_reflux_id, Etot_reflux_id, mhdFluxRefineOp,    fillPattern);
    }
};

} // namespace PHARE::amr
#endif
