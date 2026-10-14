#ifndef PHARE_AMR_MESSENGERS_MHD_MESSENGER_FWD_HPP
#define PHARE_AMR_MESSENGERS_MHD_MESSENGER_FWD_HPP

namespace PHARE::amr
{
template<typename MHDModel>
class Linear2TemporalTransfer;

template<typename MHDModel>
class MC2011TemporalTransfer;

// Second-order linear temporal interpolation is the default coarse-fine transfer.
template<typename MHDModel, typename TemporalTransfer = Linear2TemporalTransfer<MHDModel>>
class MHDMessenger;
} // namespace PHARE::amr

#endif // PHARE_AMR_MESSENGERS_MHD_MESSENGER_FWD_HPP
