#ifndef PHARE_CORE_NUMERICS_GODUNOV_FLUXES_SHARED_SMOOTHNESS_HPP
#define PHARE_CORE_NUMERICS_GODUNOV_FLUXES_SHARED_SMOOTHNESS_HPP

#include "core/def.hpp"
#include "core/utilities/index/index.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace PHARE::core
{
/// Cell-centred shared smoothness indicators, one set of N per direction. Godunov fills them
/// before its face reconstructions and the UCT reuses them for its edge reconstructions, so that
/// both pick the same stencils. Allocated only when enabled.
template<typename VecField, std::size_t N>
class SharedSmoothnessState
{
    using Field                     = VecField::field_type;
    constexpr static auto dimension = VecField::dimension;

public:
    using Betas = std::array<double, N>;

    struct Indicators
    {
        NO_DISCARD auto getCompileTimeResourcesViewList() { return views_(*this); }
        NO_DISCARD auto getCompileTimeResourcesViewList() const { return views_(*this); }

        std::array<Field, N> x = make_("ssi_x", std::make_index_sequence<N>{});
        std::array<Field, N> y = make_("ssi_y", std::make_index_sequence<N>{});
        std::array<Field, N> z = make_("ssi_z", std::make_index_sequence<N>{});

    private:
        template<std::size_t... Is>
        static auto make_(std::string const& prefix, std::index_sequence<Is...>)
        {
            return std::array<Field, N>{
                Field{prefix + "_" + std::to_string(Is), MHDQuantity::Scalar::rho}...};
        }

        static auto views_(auto& self)
        {
            auto refs = [](auto& arr) {
                return std::apply([](auto&... fs) { return std::forward_as_tuple(fs...); }, arr);
            };
            if constexpr (dimension == 1)
                return refs(self.x);
            else if constexpr (dimension == 2)
                return std::tuple_cat(refs(self.x), refs(self.y));
            else
                return std::tuple_cat(refs(self.x), refs(self.y), refs(self.z));
        }
    };

    SharedSmoothnessState() = default;
    explicit SharedSmoothnessState(bool const enabled)
    {
        if (enabled)
            indicators_.emplace_back();
    }

    NO_DISCARD bool enabled() const { return !indicators_.empty(); }

    NO_DISCARD std::vector<Indicators>& getRunTimeResourcesViewList() { return indicators_; }
    NO_DISCARD std::vector<Indicators> const& getRunTimeResourcesViewList() const
    {
        return indicators_;
    }

    template<auto direction>
    void set(MeshIndex<dimension> const& idx, Betas const& betas)
    {
        auto& fields = fields_<direction>(indicators_[0]);
        for (std::size_t m = 0; m < N; ++m)
            fields[m](idx) = betas[m];
    }

    template<auto direction>
    NO_DISCARD auto get(MeshIndex<dimension> const& idx) const
    {
        auto const& fields = fields_<direction>(indicators_[0]);
        Betas betas;
        for (std::size_t m = 0; m < N; ++m)
            betas[m] = fields[m](idx);
        return betas;
    }

private:
    template<auto direction>
    static auto& fields_(auto& indicators)
    {
        if constexpr (direction == Direction::X)
            return indicators.x;
        else if constexpr (direction == Direction::Y)
            return indicators.y;
        else
            return indicators.z;
    }

    // holds at most one element: the resources manager handles optional resources this way
    std::vector<Indicators> indicators_;
};

} // namespace PHARE::core

#endif
