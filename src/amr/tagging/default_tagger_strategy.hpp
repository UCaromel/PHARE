#ifndef DEFAULT_TAGGER_STRATEGY_H
#define DEFAULT_TAGGER_STRATEGY_H

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/types.hpp"

#include "initializer/data_provider.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "tagger_strategy.hpp"

namespace PHARE::amr
{
namespace tagger_detail
{
    inline std::vector<std::string> readFieldList(initializer::PHAREDict const& dict)
    {
        std::vector<std::string> names;
        if (!dict.contains("nbr_fields"))
            return names;
        auto const n = dict["nbr_fields"].template to<std::size_t>();
        names.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            names.push_back(dict["field" + std::to_string(i)].template to<std::string>());
        return names;
    }

    inline std::string nameTail(std::string const& s)
    {
        auto const p = s.rfind('_');
        return p == std::string::npos ? s : s.substr(p + 1);
    }

    // "EM_B_x" -> "Bx" ; otherwise empty
    inline std::string compactComponentName(std::string const& s)
    {
        auto const p = s.rfind('_');
        if (p == std::string::npos || p == 0)
            return {};
        auto const dir = s.substr(p + 1);
        if (dir.size() != 1 || (dir[0] != 'x' && dir[0] != 'y' && dir[0] != 'z'))
            return {};
        return nameTail(s.substr(0, p)) + dir;
    }

    // Recursively walks the model's resource tree (tuples returned by
    // getCompileTimeResourcesViewList) and pushes every scalar Field whose name matches one of
    // the requested strings. A TensorField (e.g. VecField) matching by name expands to all its
    // components. A scalar Field matches by full name, last token, or compact "Bx" form.
    template<typename FieldPtr, typename Node>
    void collectFields(std::vector<FieldPtr>& out, std::vector<std::string> const& want, Node& node)
    {
        auto const matches = [&](std::string const& full, bool acceptCompact) {
            for (auto const& w : want)
            {
                if (w == full || w == nameTail(full))
                    return true;
                if (acceptCompact && w == compactComponentName(full))
                    return true;
            }
            return false;
        };

        if constexpr (requires {
                          node.components();
                          node.name();
                      })
        {
            // TensorField (rank >= 1): match by vector name, else recurse into components.
            if (matches(node.name(), /*acceptCompact=*/false))
            {
                std::apply([&](auto&... c) { (out.push_back(&c), ...); }, node.components());
            }
            else
            {
                std::apply([&](auto&... c) { (collectFields<FieldPtr>(out, want, c), ...); },
                           node.components());
            }
        }
        else if constexpr (requires {
                               node.name();
                               node.physicalQuantity();
                           })
        {
            // Scalar Field.
            if (matches(node.name(), /*acceptCompact=*/true))
                out.push_back(&node);
        }
        else if constexpr (requires { node.getCompileTimeResourcesViewList(); })
        {
            std::apply([&](auto&... c) { (collectFields<FieldPtr>(out, want, c), ...); },
                       node.getCompileTimeResourcesViewList());
        }
        // else: unrelated node (e.g. ion population vector entries) — ignored.
    }

    // Per-cell criterion: max over directions of |F(p+2e_d)-F(p)| / (1 + |F(p+e_d)-F(p)|).
    template<std::size_t dim, typename Field>
    double cellCriterion(Field const& F, std::array<std::uint32_t, dim> const& p)
    {
        double crit = 0;
        core::for_N<dim>([&](auto idx) {
            constexpr auto d = idx();
            auto q1          = p;
            auto q2          = p;
            q1[d] += 1;
            q2[d] += 2;
            auto const delta_2 = std::abs(F(q2) - F(p));
            auto const delta_1 = std::abs(F(q1) - F(p));
            crit               = std::max(crit, delta_2 / (1 + delta_1));
        });
        return crit;
    }

} // namespace tagger_detail


template<typename Model>
class DefaultTaggerStrategy : public TaggerStrategy<Model>
{
    using gridlayout_type           = Model::gridlayout_type;
    static auto constexpr dimension = Model::dimension;


public:
    DefaultTaggerStrategy(initializer::PHAREDict const& dict)
        : threshold_{cppdict::get_value(dict, "threshold", 0.1)}
        , innerBoundaryNoRefinementHalo_{
              dict.contains("inner_boundary_no_refinement_halo")
                  ? std::optional{dict["inner_boundary_no_refinement_halo"].template to<double>()}
                  : std::nullopt}
        , physicalBoundaryNoRefinementHalo_{
              dict.contains("physical_boundary_no_refinement_halo")
                  ? std::optional{
                        dict["physical_boundary_no_refinement_halo"].template to<double>()}
                  : std::nullopt}
        , fieldNames_{tagger_detail::readFieldList(dict)}
    {
        if (physicalBoundaryNoRefinementHalo_)
        {
            char const* const axes[3] = {"x", "y", "z"};
            for (std::size_t d = 0; d < dimension; ++d)
            {
                domainLower_[d]
                    = dict[std::string{"domain_lower_"} + axes[d]].template to<double>();
                domainUpper_[d]
                    = dict[std::string{"domain_upper_"} + axes[d]].template to<double>();
                periodic_[d] = dict[std::string{"bdry_periodic_"} + axes[d]].template to<bool>();
            }
        }
    }

    void tag(Model& model, gridlayout_type const& layout, int* tags) const override;

private:
    void legacyTag1D_(Model& model, gridlayout_type const& layout, int* tags) const;
    void genericTag_(Model& model, gridlayout_type const& layout, int* tags) const;
    void applyInnerBoundaryMask_(Model& model, gridlayout_type const& layout, int* tags) const;
    void applyPhysicalBoundaryMask_(gridlayout_type const& layout, int* tags) const;

    // Default: the model's magnetic VecField (B1 for MHD predictor/corrector, else B). Returned
    // as a vector of Field pointers to feed the generic kernel.
    template<typename Model_>
    static auto defaultBComponents_(Model_& model);

    double threshold_ = 0.1;
    std::optional<double> innerBoundaryNoRefinementHalo_;
    std::optional<double> physicalBoundaryNoRefinementHalo_;
    std::array<double, dimension> domainLower_{};
    std::array<double, dimension> domainUpper_{};
    std::array<bool, dimension> periodic_{};
    std::vector<std::string> fieldNames_; // empty => use default (B components)
};


template<typename Model>
template<typename Model_>
auto DefaultTaggerStrategy<Model>::defaultBComponents_(Model_& model)
{
    using Comp = PHARE::core::Component;
    auto& B    = [&]() -> auto& {
        if constexpr (requires { model.get_B1(); })
            return model.get_B1();
        else
            return model.get_B();
    }();
    using FieldPtr = decltype(&B.getComponent(Comp::X));
    return std::vector<FieldPtr>{&B.getComponent(Comp::X), &B.getComponent(Comp::Y),
                                 &B.getComponent(Comp::Z)};
}


template<typename Model>
void DefaultTaggerStrategy<Model>::tag(Model& model, gridlayout_type const& layout, int* tags) const
{
    bool const isDefault = fieldNames_.empty();

    // Preserve legacy 1D criterion (5-point smoothed L2 over By,Bz) only when the user did not
    // override the field list. The generic max-over-(field,direction) path matches the legacy
    // 2D/3D path exactly, so no extra branch is needed there.
    if constexpr (dimension == 1)
    {
        if (isDefault)
        {
            legacyTag1D_(model, layout, tags);
            applyInnerBoundaryMask_(model, layout, tags);
            applyPhysicalBoundaryMask_(layout, tags);
            return;
        }
    }

    genericTag_(model, layout, tags);
    applyInnerBoundaryMask_(model, layout, tags);
    applyPhysicalBoundaryMask_(layout, tags);
}


template<typename Model>
void DefaultTaggerStrategy<Model>::applyPhysicalBoundaryMask_(gridlayout_type const& layout,
                                                               int* tags) const
{
    if (!physicalBoundaryNoRefinementHalo_)
        return;

    bool constexpr c_ordering = false;
    auto tagsv        = core::NdArrayView<dimension, int, c_ordering>(tags, layout.nbrCells());
    auto const amrBox = layout.AMRBox();
    auto const tagBox = boxFromNbrCells(layout.nbrCells());

    for (auto const [amrPoint, tagPoint] : core::boxes_iterator{amrBox, tagBox})
    {
        auto const coords = layout.cellCenteredCoordinates(amrPoint);
        bool nearBdry     = false;
        core::for_N<dimension>([&](auto idx) {
            constexpr auto d = idx();
            if (!periodic_[d])
            {
                double const dlow = coords[d] - domainLower_[d];
                double const dhi  = domainUpper_[d] - coords[d];
                if (std::min(dlow, dhi) <= *physicalBoundaryNoRefinementHalo_)
                    nearBdry = true;
            }
        });
        if (nearBdry)
            tagsv(tagPoint.toArray()) = 0;
    }
}


template<typename Model>
void DefaultTaggerStrategy<Model>::legacyTag1D_(Model& model, gridlayout_type const& layout,
                                                int* tags) const
{
    auto& B = [&]() -> auto& {
        if constexpr (requires { model.get_B1(); })
            return model.get_B1();
        else
            return model.get_B();
    }();
    auto& By = B.getComponent(PHARE::core::Component::Y);
    auto& Bz = B.getComponent(PHARE::core::Component::Z);

    auto const start_x
        = layout.physicalStartIndex(PHARE::core::QtyCentering::dual, PHARE::core::Direction::X);
    auto const end_x = layout.nbrCells()[0] - 1;

    bool constexpr c_ordering = false;
    auto tagsv = core::NdArrayView<dimension, int, c_ordering>(tags, layout.nbrCells());

    // At interp order 1 (nbrGhosts==2) the 5-point stencil would reach past the last ghost
    // node if we tagged the last patch cell, so we skip it. Higher interp orders are safe.
    auto constexpr doLastCell = gridlayout_type::nbrGhosts() > 2;
    std::size_t oneOrZero     = doLastCell ? 1 : 0;

    for (auto iCell = 0u, ix = start_x; iCell < end_x + oneOrZero; ++ix, ++iCell)
    {
        auto const Byavg     = 0.2 * (By(ix - 2) + By(ix - 1) + By(ix) + By(ix + 1) + By(ix + 2));
        auto const Bzavg     = 0.2 * (Bz(ix - 2) + Bz(ix - 1) + Bz(ix) + Bz(ix + 1) + Bz(ix + 2));
        auto const Byavgp1   = 0.2 * (By(ix - 1) + By(ix) + By(ix + 1) + By(ix + 2) + By(ix + 3));
        auto const Bzavgp1   = 0.2 * (Bz(ix - 1) + Bz(ix) + Bz(ix + 1) + Bz(ix + 2) + Bz(ix + 3));
        auto const criter_by = std::abs(Byavgp1 - Byavg) / (1 + std::abs(Byavg));
        auto const criter_bz = std::abs(Bzavgp1 - Bzavg) / (1 + std::abs(Bzavg));
        auto const criter    = std::sqrt(criter_by * criter_by + criter_bz * criter_bz);

        tagsv(iCell) = (criter > threshold_) ? 1 : 0;
    }
}


template<typename Model>
void DefaultTaggerStrategy<Model>::genericTag_(Model& model, gridlayout_type const& layout,
                                               int* tags) const
{
    // Resolve the field list. Default (empty list) selects the model's magnetic VecField
    // components, matching the legacy behavior of the 2D/3D tagger.
    auto fields = [&]() {
        if (fieldNames_.empty())
            return defaultBComponents_(model);

        using FieldPtr = typename decltype(defaultBComponents_(model))::value_type;
        std::vector<FieldPtr> out;
        if constexpr (requires { model.state; })
            tagger_detail::collectFields<FieldPtr>(out, fieldNames_, model.state);
        else
            tagger_detail::collectFields<FieldPtr>(out, fieldNames_, model);

        if (out.empty())
            throw std::runtime_error("DefaultTaggerStrategy: none of the requested tag_fields "
                                     "matched any model field");
        return out;
    }();

    // We loop on tag-buffer indices (no ghosts) and offset into fields by the dual-centered
    // start index in every direction. The original implementation reads all components at the
    // dual cell-index regardless of their actual centering; we preserve that behavior.
    std::array<std::uint32_t, dimension> start;
    core::for_N<dimension>([&](auto idx) {
        constexpr auto d = idx();
        start[d]         = layout.physicalStartIndex(PHARE::core::QtyCentering::dual,
                                                     static_cast<PHARE::core::Direction>(d));
    });

    auto const ncells         = layout.nbrCells();
    bool constexpr c_ordering = false;
    auto tagsv                = core::NdArrayView<dimension, int, c_ordering>(tags, ncells);

    auto const tagBox = boxFromNbrCells(ncells);
    for (auto const& tagPoint : tagBox)
    {
        std::array<std::uint32_t, dimension> fp;
        core::for_N<dimension>([&](auto idx) {
            constexpr auto d = idx();
            fp[d]            = start[d] + static_cast<std::uint32_t>(tagPoint[d]);
        });

        double crit = 0;
        for (auto const* F : fields)
            crit = std::max(crit, tagger_detail::cellCriterion<dimension>(*F, fp));

        tagsv(tagPoint.toArray()) = (crit > threshold_) ? 1 : 0;
    }
}


template<typename Model>
void DefaultTaggerStrategy<Model>::applyInnerBoundaryMask_(Model& model,
                                                           gridlayout_type const& layout,
                                                           int* tags) const
{
    if constexpr (requires { model.innerBoundaryManager; })
    {
        if (model.hasInnerBoundary() && innerBoundaryNoRefinementHalo_)
        {
            bool constexpr c_ordering = false;
            auto tagsv = core::NdArrayView<dimension, int, c_ordering>(tags, layout.nbrCells());

            auto const& geom  = model.innerBoundaryManager->getGeometry();
            auto const amrBox = layout.AMRBox();
            auto const tagBox = boxFromNbrCells(layout.nbrCells());

            for (auto const [amrPoint, tagPoint] : core::boxes_iterator{amrBox, tagBox})
            {
                auto const amrCoords = layout.cellCenteredCoordinates(amrPoint);
                if (geom.signedDistance(amrCoords) <= *innerBoundaryNoRefinementHalo_)
                    tagsv(tagPoint.toArray()) = 0;
            }
        }
    }
}

} // namespace PHARE::amr

#endif // DEFAULT_TAGGER_STRATEGY_H
