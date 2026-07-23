#ifndef PHARE_MHD_MESSENGER_INFO_HPP
#define PHARE_MHD_MESSENGER_INFO_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/models/mhd_state_increment.hpp"
#include "messenger_info.hpp"



namespace PHARE
{
namespace amr
{
    class MHDMessengerInfo : public IMessengerInfo
    {
    public:
        std::string modelDensity;
        std::string modelVelocity;
        std::string modelMagnetic;
        std::string modelPressure;

        std::string modelMomentum;
        std::string modelTotalEnergy;

        std::string modelElectric;

        std::vector<std::string> initDensity;
        std::vector<std::string> initMomentum;
        std::vector<std::string> initMagnetic;
        std::vector<std::string> initTotalEnergy;

        std::vector<std::string> ghostDensity;
        std::vector<std::string> ghostVelocity;
        std::vector<std::string> ghostMagnetic; // not actually to fill ghost cells but rather for
                                                // amr operations, see hybrid
        std::vector<std::string> ghostPressure;
        std::vector<std::string> ghostMomentum;
        std::vector<std::string> ghostTotalEnergy;
        // std::vector<std::string> ghostMagneticFluxesX;
        // std::vector<std::string> ghostMagneticFluxesY;
        // std::vector<std::string> ghostMagneticFluxesZ;
        std::vector<std::string> ghostElectric;

        // no point-value entries: point-value quantities are local derived quantities
        // (computed on shrunk ghost boxes from average ghosts), never communicated.

        core::AllFluxesNames reflux;
        core::AllFluxesNames fluxSum;
        std::string refluxElectric;
        std::string fluxSumElectric;

        // MC2011 temporal reconstruction (persisted back-solve inputs): populated only
        // when the selected integrator exposes exposeStageStates() (SSPRK4_5 today).
        // Default-constructed (empty field names) for every other integrator -- the
        // messenger treats an empty stageState1.rho as "not provided by this
        // integrator". Conserved quads only (rho/rhoV/Etot/B), hence the increment
        // name-bundle type even though the integrator persists full MHD states.
        core::MHDStateIncrementNames stageState1;
        core::MHDStateIncrementNames stageState2;
        core::MHDStateIncrementNames stageState3;
        core::MHDStateIncrementNames stageState4;
        core::MHDStateIncrementNames unp1;

        virtual ~MHDMessengerInfo() = default;
    };

} // namespace amr


} // namespace PHARE
#endif
