#ifndef PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_HPP
#define PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_HPP

#include "core/data/vecfield/vecfield_component.hpp"

#include "diagnostic/detail/h5typewriter.hpp"

#include <stdexcept>

namespace PHARE::diagnostic::h5
{
/*
 * Possible outputs
 *
 * /t#/pl#/p#/electromag_(B, E)/(x,y,z)
 */
template<typename H5Writer>
class ElectromagDiagnosticWriter : public H5TypeWriter<H5Writer>
{
public:
    using Super = H5TypeWriter<H5Writer>;
    using Super::checkCreateFileFor_;
    using Super::fileData_;
    using Super::h5Writer_;
    using Super::initDataSets_;
    using Super::writeAttributes_;
    using Super::writeGhostsAttr_;
    using Attributes = Super::Attributes;
    using GridLayout = H5Writer::GridLayout;
    using FloatType  = H5Writer::FloatType;

    ElectromagDiagnosticWriter(H5Writer& h5Writer)
        : Super{h5Writer}
    {
    }

    void write(DiagnosticProperties&) override;
    void compute(DiagnosticProperties& diagnostic) override;

    void createFiles(DiagnosticProperties& diagnostic) override;

    void getDataSetInfo(DiagnosticProperties& diagnostic, std::size_t iLevel,
                        std::string const& patchID, Attributes& patchAttributes) override;

    void initDataSets(DiagnosticProperties& diagnostic,
                      std::unordered_map<std::size_t, std::vector<std::string>> const& patchIDs,
                      Attributes& patchAttributes, std::size_t maxLevel) override;

    void writeAttributes(
        DiagnosticProperties&, Attributes&,
        std::unordered_map<std::size_t, std::vector<std::pair<std::string, Attributes>>>&,
        std::size_t maxLevel) override;

private:
    auto isActiveDiag(DiagnosticProperties const& diagnostic, std::string const& tree,
                      std::string var)
    {
        return diagnostic.quantity == tree + var;
    };
};


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::compute(DiagnosticProperties& diagnostic)
{
    if (!isActiveDiag(diagnostic, "/", "EM_B"))
        return;

    auto& modelView = this->h5Writer_.modelView();
    if constexpr (!(requires { modelView.getB1(); modelView.getB0(); }))
        return;
    else
    {
        auto minLvl     = this->h5Writer_.minLevel;
        auto maxLvl     = this->h5Writer_.maxLevel;

        auto& B         = modelView.getB();
        auto const& B1  = modelView.getB1();
        auto const& B0  = modelView.getB0();

        auto reconstructB = [&](GridLayout& layout, std::string patchID, std::size_t iLevel) {
            auto& Bx = B.getComponent(core::Component::X);
            auto& By = B.getComponent(core::Component::Y);
            auto& Bz = B.getComponent(core::Component::Z);
            auto const& B1x = B1.getComponent(core::Component::X);
            auto const& B1y = B1.getComponent(core::Component::Y);
            auto const& B1z = B1.getComponent(core::Component::Z);
            auto const& B0x = B0.getComponent(core::Component::X);
            auto const& B0y = B0.getComponent(core::Component::Y);
            auto const& B0z = B0.getComponent(core::Component::Z);

            auto const rebuildComponent = [&](auto& dst, auto const& perturbed,
                                             auto const& background) {
                layout.evalOnGhostBox(dst, [&](auto&... args) mutable {
                    dst(args...) = perturbed(args...) + background(args...);
                });
            };

            rebuildComponent(Bx, B1x, B0x);
            rebuildComponent(By, B1y, B0y);
            rebuildComponent(Bz, B1z, B0z);
        };

        modelView.visitHierarchy(reconstructB, minLvl, maxLvl);
    }
}

template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::createFiles(DiagnosticProperties& diagnostic)
{
    std::string tree = "/";
    checkCreateFileFor_(diagnostic, fileData_, tree, "EM_B", "EM_E");
}


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::getDataSetInfo(DiagnosticProperties& diagnostic,
                                                          std::size_t iLevel,
                                                          std::string const& patchID,
                                                          Attributes& patchAttributes)
{
    auto& h5Writer         = this->h5Writer_;
    std::string lvlPatchID = std::to_string(iLevel) + "_" + patchID;

    auto const infoVF = [&](auto& vecF, std::string name, auto& attr) {
        for (auto& [id, type] : core::Components::componentMap())
        {
            // highfive doesn't accept uint32 which ndarray.shape() is
            auto const& array_shape = vecF.getComponent(type).shape();
            attr[name][id]          = std::vector<std::size_t>(array_shape.data(),
                                                               array_shape.data() + array_shape.size());
            auto ghosts = GridLayout::nDNbrGhosts(vecF.getComponent(type).physicalQuantity());
            for (std::uint8_t i = 1; i < GridLayout::dimension; ++i)
                if (ghosts[i] != ghosts[i - 1])
                    throw std::runtime_error("ghosts per direction must be constant");
            attr[name][id + "_ghosts"] = static_cast<std::size_t>(ghosts[0]);
        }
    };

    if (isActiveDiag(diagnostic, "/", "EM_B"))
    {
        auto& B = h5Writer.modelView().getB();
        infoVF(B, "EM_B", patchAttributes[lvlPatchID]);
    }
    if (isActiveDiag(diagnostic, "/", "EM_E"))
    {
        auto& E = h5Writer.modelView().getE();
        infoVF(E, "EM_E", patchAttributes[lvlPatchID]);
    }
}


template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::initDataSets(
    DiagnosticProperties& diagnostic,
    std::unordered_map<std::size_t, std::vector<std::string>> const& patchIDs,
    Attributes& patchAttributes, std::size_t maxLevel)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = *fileData_.at(diagnostic.quantity);

    auto const initVF = [&](auto& path, auto& attr, std::string key, auto null) {
        for (auto& [id, type] : core::Components::componentMap())
        {
            auto vFPath = path + "/" + key + "_" + id;
            h5Writer.template createDataSet<FloatType>(
                h5file, vFPath,
                null ? std::vector<std::size_t>(GridLayout::dimension, 0)
                     : attr[key][id].template to<std::vector<std::size_t>>());

            this->writeGhostsAttr_(h5file, vFPath,
                                   null ? 0 : attr[key][id + "_ghosts"].template to<std::size_t>(),
                                   null);
        }
    };

    auto const initPatch = [&](auto& level, auto& attr, std::string patchID = "") {
        bool null = patchID.empty();
        std::string path{h5Writer.getPatchPathAddTimestamp(level, patchID)};
        std::string tree = "/";

        if (isActiveDiag(diagnostic, tree, "EM_B"))
        {
            initVF(path, attr, "EM_B", null);
        }
        if (isActiveDiag(diagnostic, tree, "EM_E"))
        {
            initVF(path, attr, "EM_E", null);
        }
    };

    initDataSets_(patchIDs, patchAttributes, maxLevel, initPatch);
}



template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::write(DiagnosticProperties& diagnostic)
{
    auto& h5Writer = this->h5Writer_;
    auto& h5file   = *fileData_.at(diagnostic.quantity);

    std::string tree = "/";
    std::string path = h5Writer.patchPath() + "/";

    if (isActiveDiag(diagnostic, tree, "EM_B"))
    {
        auto& B = h5Writer.modelView().getB();
        h5Writer.writeTensorFieldAsDataset(h5file, path + "EM_B", B);
    }
    if (isActiveDiag(diagnostic, tree, "EM_E"))
    {
        auto& E = h5Writer.modelView().getE();
        h5Writer.writeTensorFieldAsDataset(h5file, path + "EM_E", E);
    }
}



template<typename H5Writer>
void ElectromagDiagnosticWriter<H5Writer>::writeAttributes(
    DiagnosticProperties& diagnostic, Attributes& fileAttributes,
    std::unordered_map<std::size_t, std::vector<std::pair<std::string, Attributes>>>&
        patchAttributes,
    std::size_t maxLevel)
{
    writeAttributes_(diagnostic, Super::h5FileForQuantity(diagnostic), fileAttributes,
                     patchAttributes, maxLevel);
}


} // namespace PHARE::diagnostic::h5

#endif /* PHARE_DIAGNOSTIC_DETAIL_TYPES_ELECTROMAG_H */
