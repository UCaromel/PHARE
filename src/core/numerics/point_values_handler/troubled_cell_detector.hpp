#ifndef PHARE_CORE_NUMERICS_TROUBLED_CELL_DETECTOR_HPP
#define PHARE_CORE_NUMERICS_TROUBLED_CELL_DETECTOR_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/differential_operators.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/types.hpp"

#include "initializer/data_provider.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace PHARE::core
{
enum class TroubledCellDetectorType { None, Jameson, DerivativeRatio };

/**
 * @brief What a troubled cell limits.
 *
 *  - All: the conversion of rho, rhoV, Etot, and of B on the faces of the cell.
 *  - Hydro: rho, rhoV and the hydrodynamic energy e_h = Etot - magnetic energy, which is what is
 *    converted in place of Etot, in every cell. B is never limited, so the pressure is never
 *    taken from an averaged B next to one that is not. e_h is built from the point values of B
 *    and converted in turn: the point-value Etot is known two cells less far into the ghosts.
 */
enum class LimitedVariables { All, Hydro };


/**
 * @brief Limiter of the fourth-order conversion between cell averages and point values.
 *
 * In a troubled cell the point value is the cell average: the Laplacian correction of the
 * conversion overshoots across a discontinuity and can make the point-value pressure negative.
 * With fluxAveragingSkip, the faces of a troubled cell also keep their point-value flux as face
 * average (Berta et al. 2024, arXiv:2310.11831, section 3.4).
 */
struct ConversionLimiterInfo
{
    TroubledCellDetectorType detector = TroubledCellDetectorType::None;
    double threshold                  = 0.;
    bool fluxAveragingSkip            = true;
    LimitedVariables variables        = LimitedVariables::All;

    // ghost cells the Hydro form reads beyond what the conversion of Etot itself does
    static constexpr std::uint32_t hydroEnergyReach = 2;

    bool limitsHydroOnly() const { return enabled() && variables == LimitedVariables::Hydro; }
    bool enabled() const { return detector != TroubledCellDetectorType::None; }
    bool skipsFluxAveraging() const { return enabled() && fluxAveragingSkip; }

    // default thresholds: Jameson's is the one of the pressure switch it comes from; the
    // derivative ratio has no published value, 0.3 flags as many cells as Jameson on Orszag-Tang
    static double defaultThreshold(TroubledCellDetectorType const detector)
    {
        return detector == TroubledCellDetectorType::DerivativeRatio ? 0.3 : 0.05;
    }

    static TroubledCellDetectorType detectorFromName(std::string const& name)
    {
        if (name == "none")
            return TroubledCellDetectorType::None;
        if (name == "jameson")
            return TroubledCellDetectorType::Jameson;
        if (name == "derivative_ratio")
            return TroubledCellDetectorType::DerivativeRatio;
        throw std::runtime_error("ConversionLimiterInfo: unknown troubled-cell detector " + name);
    }

    static LimitedVariables variablesFromName(std::string const& name)
    {
        if (name == "all")
            return LimitedVariables::All;
        if (name == "hydro")
            return LimitedVariables::Hydro;
        throw std::runtime_error("ConversionLimiterInfo: unknown limited variables " + name);
    }

    // dict is the solver's algo dict; without a "conversion_limiter" entry nothing is limited
    static ConversionLimiterInfo FROM(initializer::PHAREDict const& dict)
    {
        if (!dict.contains("conversion_limiter"))
            return {};

        auto const& limiter = dict["conversion_limiter"];
        auto const detector
            = detectorFromName(cppdict::get_value(limiter, "detector", std::string{"none"}));
        return {detector, cppdict::get_value(limiter, "threshold", defaultThreshold(detector)),
                cppdict::get_value(limiter, "flux_averaging_skip", true),
                variablesFromName(cppdict::get_value(limiter, "variables", std::string{"all"}))};
    }
};


/**
 * @brief Second difference of u over three cells, relative to their sum (the pressure switch of
 * Jameson, Schmidt & Turkel 1981). O(h²) where u is smooth and does not vanish, O(1) on a jump.
 */
inline double jamesonSensor(double const uPrev, double const u, double const uNext)
{
    constexpr double eps = 1.e-12;
    return std::abs(uNext - 2. * u + uPrev)
           / (std::abs(uNext) + 2. * std::abs(u) + std::abs(uPrev) + eps);
}


/**
 * @brief Derivative ratio of Berta et al. (2024), eqs 36-37, on the five cells u[0..4] centred on
 * u[2]: the third (resp. fourth) undivided difference against the first (resp. second) and a
 * reference magnitude, whichever is larger.
 */
inline double derivativeRatioSensor(std::array<double, 5> const& u, double const reference)
{
    constexpr double eps = 1.e-12;

    auto const d1  = 0.5 * (u[3] - u[1]);
    auto const d2  = u[3] - 2. * u[2] + u[1];
    auto const d3  = 0.5 * (u[4] - 2. * u[3] + 2. * u[1] - u[0]);
    auto const d4  = u[4] - 4. * u[3] + 6. * u[2] - 4. * u[1] + u[0];
    auto const ref = std::abs(reference);

    auto const odd  = std::abs(d3) / (ref + std::abs(d1) + std::abs(d3) + eps);
    auto const even = std::abs(d4) / (ref + std::abs(d2) + std::abs(d4) + eps);
    return std::max(odd, even);
}


/**
 * @brief Flags the troubled cells of a patch from the primitives of its cell averages.
 *
 * A cell is troubled when its sensor exceeds the threshold, or when one of its face neighbours
 * is. Only the cells of the patch are flagged: their sensors read the averages up to
 * sensorReach + 1 ghost cells away, and the flags of the ghost cells come from the patches that
 * own them (MHDMessenger::fillTroubledCellGhosts).
 *
 *  - Jameson: rho and P, largest ratio over the directions. The magnetic pressure is left out on
 *    purpose: it vanishes in smooth flows, where a relative second difference is not small.
 *  - DerivativeRatio: rho, P and V, norm of the ratios over the directions. V is compared to the
 *    isothermal sound speed. B is left out for the same reason as above.
 */
template<typename GridLayout>
class TroubledCellDetector
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    TroubledCellDetector(GridLayout const& layout, ConversionLimiterInfo const& info)
        : layout_{layout}
        , info_{info}
    {
    }

    // cells a sensor reads on each side of the cell it tests
    static constexpr std::uint32_t sensorReach(TroubledCellDetectorType const detector)
    {
        return detector == TroubledCellDetectorType::DerivativeRatio ? 2 : 1;
    }

    /**
     * @param rho, V, P primitives of the cell averages
     * @param sensed scratch: 1 where the sensor exceeds the threshold, on the patch cells and one
     * ghost layer
     * @param troubled 1 in the troubled cells of the patch, 0 in the others; ghosts untouched
     */
    template<typename Field, typename VecField>
    void operator()(Field const& rho, VecField const& V, Field const& P, Field& sensed,
                    Field& troubled) const
    {
        switch (info_.detector)
        {
            case TroubledCellDetectorType::Jameson:
                sense_(sensed, [&](auto const& index) { return jameson_(rho, P, index); });
                break;
            case TroubledCellDetectorType::DerivativeRatio:
                sense_(sensed,
                       [&](auto const& index) { return derivativeRatio_(rho, V, P, index); });
                break;
            case TroubledCellDetectorType::None:
                throw std::logic_error("TroubledCellDetector: no detector selected");
        }

        layout_.evalOnBox(troubled, [&](auto&... args) mutable {
            MeshIndex<dimension> const index{args...};
            auto flag = sensed(index);
            for_N<dimension>([&](auto d) {
                constexpr auto dir = decltype(d)::value;
                flag               = std::max({flag, detail::shifted<dir>(sensed, index, -1),
                                               detail::shifted<dir>(sensed, index, +1)});
            });
            troubled(index) = flag;
        });
    }

private:
    void sense_(auto& sensed, auto const& sensor) const
    {
        Point<std::uint32_t, dimension> oneLayer;
        for (std::size_t i = 0; i < dimension; ++i)
            oneLayer[i] = 1;

        layout_.evalOnBiggerBox(sensed, oneLayer, [&](auto&... args) mutable {
            MeshIndex<dimension> const index{args...};
            sensed(index) = sensor(index) > info_.threshold ? 1. : 0.;
        });
    }

    static double jameson_(auto const& rho, auto const& P, MeshIndex<dimension> const& index)
    {
        double eta = 0.;
        for_N<dimension>([&](auto d) {
            constexpr auto dir = decltype(d)::value;
            auto const along   = [&](auto const& u) {
                return jamesonSensor(detail::shifted<dir>(u, index, -1), u(index),
                                       detail::shifted<dir>(u, index, +1));
            };
            eta = std::max({eta, along(rho), along(P)});
        });
        return eta;
    }

    static double derivativeRatio_(auto const& rho, auto const& V, auto const& P,
                                   MeshIndex<dimension> const& index)
    {
        auto const norm = [&](auto const& u, double const reference) {
            double sum = 0.;
            for_N<dimension>([&](auto d) {
                constexpr auto dir = decltype(d)::value;
                auto const ratio   = derivativeRatioSensor(
                    {detail::shifted<dir>(u, index, -2), detail::shifted<dir>(u, index, -1),
                       u(index), detail::shifted<dir>(u, index, +1),
                       detail::shifted<dir>(u, index, +2)},
                    reference);
                sum += ratio * ratio;
            });
            return std::sqrt(sum);
        };

        auto const soundSpeed = std::sqrt(std::abs(P(index) / rho(index)));
        return std::max({norm(rho, rho(index)), norm(P, P(index)),
                         norm(V(Component::X), soundSpeed), norm(V(Component::Y), soundSpeed),
                         norm(V(Component::Z), soundSpeed)});
    }

    GridLayout layout_;
    ConversionLimiterInfo info_;
};


/**
 * @brief Whether the face `index` of direction `direction` touches a troubled cell.
 *
 * The outermost faces of the ghost box have one cell only. In a direction the mesh does not have,
 * the quantity sits at the cell centre and the cell is its own.
 */
template<auto direction, typename GridLayout, typename Field>
bool faceIsTroubled(GridLayout const& layout, Field const& troubled,
                    MeshIndex<GridLayout::dimension> const& index)
{
    constexpr auto dir = static_cast<std::size_t>(direction);

    if constexpr (dir >= GridLayout::dimension)
        return troubled(index) > 0.;
    else
    {
        auto const lastCell = layout.ghostEndIndex(QtyCentering::dual, direction);
        bool const lower    = index[dir] > 0 && detail::shifted<dir>(troubled, index, -1) > 0.;
        bool const upper    = index[dir] <= lastCell && troubled(index) > 0.;
        return lower || upper;
    }
}

} // namespace PHARE::core

#endif
