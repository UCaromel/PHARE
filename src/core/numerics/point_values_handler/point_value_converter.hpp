#ifndef PHARE_CORE_NUMERICS_POINT_VALUE_CONVERTER_HPP
#define PHARE_CORE_NUMERICS_POINT_VALUE_CONVERTER_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/utilities/index/index.hpp"
#include "core/utilities/point/point.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace PHARE::core
{
enum class PointValueConversionMode { ToPointValue, ToAverage };

inline constexpr std::uint32_t point_value_conversion_shrink = 1;

// EXPERIMENT PHARE_O4_PV_LIMIT=1: Jameson troubled-cell mask (port of exp/high-order-limit-pointvalue
// 752fb9321 build_troubled_mask_): in troubled cells the average->point conversion is skipped.
struct PVLimitStats
{
    long flagged = 0, total = 0;
};
inline PVLimitStats pvLimitStats;

// per-patch mask of the current stage, keyed by the patch AMR lower corner (2D), for the
// reconstruction fallback (PHARE_O4_RECON_FALLBACK) in Godunov
struct PVMask
{
    int nx = 0, ny = 0;
    std::vector<double> m;
};
inline std::map<std::pair<int, int>, PVMask> pvMaskRegistry;
inline bool pvLimitOn()
{
    static bool const v = [] {
        auto const* e = std::getenv("PHARE_O4_PV_LIMIT");
        return e && std::atoi(e) != 0;
    }();
    return v;
}
// EXPERIMENT PHARE_O4_EH_CONV=1 (T1, 2D): convert the hydro energy instead of Etot,
// e_h = Etot - (1 + D/24) m, m = 1/2 |K B| from the unlimited face point values (m_NN with
// PHARE_O4_P_NN); pv Etot := (1 - D/24) e_h + m so that p = (g-1)[(1 - D/24) e_h - KE_pv].
// PHARE_O4_PV_LIMIT_HYDRO=1 (T2, needs T1): in Jameson-flagged cells rho, rhoV := averages and
// pv Etot := e_h + m (p = (g-1)[e_h - KE(avg)]); face B and m never limited.
inline bool envOn_(char const* n)
{
    auto const* e = std::getenv(n);
    return e && std::atoi(e) != 0;
}
inline bool ehConvOn()
{
    static bool const v = envOn_("PHARE_O4_EH_CONV");
    return v;
}
inline bool pvLimitHydroOn()
{
    static bool const v = envOn_("PHARE_O4_PV_LIMIT_HYDRO");
    return v;
}
inline bool ehPressureNN()
{
    static bool const v = envOn_("PHARE_O4_P_NN");
    return v;
}
// EXPERIMENT PHARE_O4_PV_DET: troubled-cell detector. 0 Jameson (default), 2 Loehner (both on the
// averages rho, p, p_B), 3 relaxed discrete maximum principle with the MOOD u2 smooth-extremum pass,
// 4 physical admissibility. 3 and 4 test the converted point values (rho, p) and need T1+T2.
inline int pvDetector()
{
    static int const v = [] {
        auto const* e = std::getenv("PHARE_O4_PV_DET");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
inline double pvLimitThreshold()
{
    static double const v = [] {
        auto const* e = std::getenv("PHARE_O4_PV_LIMIT_THR");
        // Jameson: the branch's jameson_threshold default; Loehner: 0.7 (offline replica)
        return e ? std::atof(e) : (pvDetector() == 2 ? 0.7 : 0.05);
    }();
    return v;
}

// EXPERIMENT PHARE_O4_FACEP_DIAG: first non-positive reconstructed face state of the stage
struct O4FaceProbe
{
    bool on  = std::getenv("PHARE_O4_FACEP_DIAG") != nullptr;
    long n   = 0;
    double P = 0, rho = 0, x = 0, y = 0;
    int dir  = -1;
};
inline O4FaceProbe o4FaceProbe;

template<typename GridLayout>
class PointValueConverter
{
    constexpr static auto dimension = GridLayout::dimension;

public:
    explicit PointValueConverter(GridLayout const& layout)
        : layout_{layout}
    {
    }

    void toPointValues(auto& pv, auto const& state, double const gamma = 0.)
    {
        static constexpr auto toPointValue = PointValueConversionMode::ToPointValue;
        static constexpr auto shrink       = point_value_conversion_shrink;

        bool const ehc  = dimension == 2 && ehConvOn() && gamma > 0.;
        bool const limH = ehc && pvLimitHydroOn();
        bool const post = limH && pvDetector() >= 3; // mask from the converted values
        if constexpr (dimension == 2)
            if ((pvLimitOn() || limH) && gamma > 0. && !post)
                buildMask_(state, gamma);
        bool const limit  = dimension == 2 && pvLimitOn() && gamma > 0. && !post;
        bool const limitC = (limit || limH) && !post; // cell rho, rhoV limited
        auto const nan    = std::numeric_limits<double>::quiet_NaN();

        auto convert_cell = [&](auto const& src, auto& dst) {
            Point<std::uint32_t, dimension> amount;
            for (std::size_t i = 0; i < dimension; ++i)
                amount[i] = shrink;

            layout_.evalOnShrinkedGhostBox(src, amount, [&](auto&... args) mutable {
                if constexpr (dimension == 2)
                    if (limitC)
                    {
                        double const m = maskAt_(args...);
                        if (!(m == m))
                            dst(args...) = nan; // mask not computable here: reach guard
                        else if (m > 0)
                            dst(args...) = src(args...);
                        else
                            dst(args...) = getCellCentered<toPointValue>(src, {args...});
                        return;
                    }
                dst(args...) = getCellCentered<toPointValue>(src, {args...});
            });
        };

        convert_cell(state.rho, pv.rho);
        if (!ehc)
            convert_cell(state.Etot, pv.Etot);
        for (int i = 0; i < 3; ++i)
            convert_cell(state.rhoV(static_cast<core::Component>(i)),
                         pv.rhoV(static_cast<core::Component>(i)));

        auto convert_face = [&]<auto dir_tag>(auto const& src, auto& dst) {
            Point<std::uint32_t, dimension> amount;
            for (std::size_t i = 0; i < dimension; ++i)
                amount[i] = (i == static_cast<std::size_t>(dir_tag)) ? 0u : shrink;

            layout_.evalOnShrinkedGhostBox(src, amount, [&](auto&... args) mutable {
                if constexpr (dimension == 2)
                    if (limit)
                    {
                        double const m = faceMask_<dir_tag>(args...);
                        if (!(m == m))
                            dst(args...) = nan;
                        else if (m > 0)
                            dst(args...) = src(args...);
                        else
                            dst(args...) = getFaceCentered<dir_tag, toPointValue>(src, {args...});
                        return;
                    }
                dst(args...) = getFaceCentered<dir_tag, toPointValue>(src, {args...});
            });
        };

        convert_face.template operator()<Direction::X>(state.B(core::Component::X),
                                                       pv.B(core::Component::X));
        convert_face.template operator()<Direction::Y>(state.B(core::Component::Y),
                                                       pv.B(core::Component::Y));
        convert_face.template operator()<Direction::Z>(state.B(core::Component::Z),
                                                       pv.B(core::Component::Z));

        if constexpr (dimension == 2)
            if (ehc)
            {
                convertHydroEnergy_(pv, state, limitC);
                if (post)
                    limitPost_(pv, state, gamma);
            }
    }

    void fluxesToAverages(auto& pv, auto& fluxes, auto& E)
    {
        static constexpr auto toAverage = PointValueConversionMode::ToAverage;

        pv.tmpE.copyData(E);
        pv.tmpFluxes.copyData(fluxes);

        auto convert_hydro_fluxes
            = [&]<auto dir_tag>(auto& src_rho_f, auto& src_rhov_f, auto& src_etot_f,
                                auto& dst_rho_f, auto& dst_rhov_f, auto& dst_etot_f) {
                  auto convert = [&](auto const& src, auto& dst) {
                      layout_.evalOnBox(dst, [&](auto&... args) mutable {
                          dst(args...) = getFaceCentered<dir_tag, toAverage>(src, {args...});
                      });
                  };

                  convert(src_rho_f, dst_rho_f);
                  convert(src_etot_f, dst_etot_f);
                  for (int i = 0; i < 3; ++i)
                      convert(src_rhov_f(static_cast<core::Component>(i)),
                              dst_rhov_f(static_cast<core::Component>(i)));
              };

        convert_hydro_fluxes.template operator()<Direction::X>(
            pv.tmpFluxes.rho_fx, pv.tmpFluxes.rhoV_fx, pv.tmpFluxes.Etot_fx, fluxes.rho_fx,
            fluxes.rhoV_fx, fluxes.Etot_fx);
        if constexpr (dimension >= 2)
        {
            convert_hydro_fluxes.template operator()<Direction::Y>(
                pv.tmpFluxes.rho_fy, pv.tmpFluxes.rhoV_fy, pv.tmpFluxes.Etot_fy, fluxes.rho_fy,
                fluxes.rhoV_fy, fluxes.Etot_fy);

            if constexpr (dimension == 3)
                convert_hydro_fluxes.template operator()<Direction::Z>(
                    pv.tmpFluxes.rho_fz, pv.tmpFluxes.rhoV_fz, pv.tmpFluxes.Etot_fz, fluxes.rho_fz,
                    fluxes.rhoV_fz, fluxes.Etot_fz);
        }

        auto convert_edge = [&]<auto dir_tag>(auto const& src, auto& dst) {
            layout_.evalOnBox(dst, [&](auto&... args) mutable {
                dst(args...) = getEdgeCentered<dir_tag, toAverage>(src, {args...});
            });
        };

        convert_edge.template operator()<Direction::X>(pv.tmpE(core::Component::X),
                                                       E(core::Component::X));
        convert_edge.template operator()<Direction::Y>(pv.tmpE(core::Component::Y),
                                                       E(core::Component::Y));
        convert_edge.template operator()<Direction::Z>(pv.tmpE(core::Component::Z),
                                                       E(core::Component::Z));
    }

private:
    // cell-centred mask over the ghost box (index = local index); NaN where not computable
    std::vector<double> mask_;
    std::vector<double> m_, eh_; // T1 magnetic energy and hydro energy (ghost box, NaN outside reach)
    int mnx_ = 0, mny_ = 0;

    double maskAt_(auto i, auto j) const
    {
        int const a = static_cast<int>(i), b = static_cast<int>(j);
        if (a < 0 || b < 0 || a >= mnx_ || b >= mny_)
            return std::numeric_limits<double>::quiet_NaN();
        return mask_[static_cast<std::size_t>(a * mny_ + b)];
    }

    // face troubled = either adjacent cell troubled (branch's face_troubled); 2D Bz: its own cell
    template<auto direction>
    double faceMask_(auto i, auto j) const
    {
        int const a = static_cast<int>(i), b = static_cast<int>(j);
        if constexpr (direction == Direction::X)
            return std::max(maskAt_(a - 1, b), maskAt_(a, b)) + 0 * (maskAt_(a - 1, b) + maskAt_(a, b));
        else if constexpr (direction == Direction::Y)
            return std::max(maskAt_(a, b - 1), maskAt_(a, b)) + 0 * (maskAt_(a, b - 1) + maskAt_(a, b));
        else
            return maskAt_(a, b);
    }

    // T1/T2 (see ehConvOn). Reach: m needs the face point values one face beyond the cell (4-point
    // projection), so m is valid on the ghost box shrunk by 1, e_h by 2, the converted pv Etot by 3
    // (flagged cells: 2); NaN outside (reach guard). Two layers more than the plain conversion.
    void convertHydroEnergy_(auto& pv, auto const& state, bool const limited)
    {
        auto const& E  = state.Etot;
        auto const& Bx = pv.B(Component::X);
        auto const& By = pv.B(Component::Y);
        auto const& Bz = pv.B(Component::Z);
        int const nx   = static_cast<int>(layout_.ghostEndIndex(E, Direction::X)) + 1;
        int const ny   = static_cast<int>(layout_.ghostEndIndex(E, Direction::Y)) + 1;
        auto const nan = std::numeric_limits<double>::quiet_NaN();
        auto const N   = static_cast<std::size_t>(nx * ny);
        auto& m  = m_;
        auto& eh = eh_;
        m.assign(N, nan);
        eh.assign(N, nan);
        auto at       = [&](int i, int j) { return static_cast<std::size_t>(i * ny + j); };
        bool const nn = ehPressureNN();
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
            {
                MeshIndex<dimension> const idx{static_cast<std::uint32_t>(i),
                                               static_cast<std::uint32_t>(j)};
                if (nn)
                {
                    constexpr double w[4] = {-1. / 16, 9. / 16, 9. / 16, -1. / 16};
                    double bx2 = 0, by2 = 0;
                    for (int q = 0; q < 4; ++q)
                    {
                        double const fx = Bx(i - 1 + q, j), fy = By(i, j - 1 + q);
                        bx2 += w[q] * fx * fx;
                        by2 += w[q] * fy * fy;
                    }
                    m[at(i, j)] = 0.5 * (bx2 + by2 + Bz(i, j) * Bz(i, j));
                }
                else
                {
                    double const bx = GridLayout::template project<
                        GridLayout::implT::template faceXToCellCenter<4>>(Bx, idx);
                    double const by = GridLayout::template project<
                        GridLayout::implT::template faceYToCellCenter<4>>(By, idx);
                    double const bz = GridLayout::template project<
                        GridLayout::implT::template faceZToCellCenter<4>>(Bz, idx);
                    m[at(i, j)] = 0.5 * (bx * bx + by * by + bz * bz);
                }
            }
        auto lap = [&](std::vector<double> const& u, int i, int j) {
            return u[at(i + 1, j)] + u[at(i - 1, j)] + u[at(i, j + 1)] + u[at(i, j - 1)]
                   - 4.0 * u[at(i, j)];
        };
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                eh[at(i, j)] = E(i, j) - m[at(i, j)] - lap(m, i, j) / 24.;
        auto& out = pv.Etot;
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
            {
                double const mk = limited ? maskAt_(i, j) : 0.;
                if (!(mk == mk))
                    out(i, j) = nan;
                else if (mk > 0)
                    out(i, j) = eh[at(i, j)] + m[at(i, j)];
                else
                    out(i, j) = eh[at(i, j)] - lap(eh, i, j) / 24. + m[at(i, j)];
            }
    }

    // detectors 3 (relaxed DMP + u2) and 4 (admissibility) on the T1 converted values, then the T2
    // hydro fallback in flagged cells. Candidate: rho_pv, p_pv = (g-1)[pv Etot - KE_pv - m]; bounds:
    // rho avg and the fallback pressure (g-1)[e_h - KE(avg)] on the 3x3 neighbours.
    // Reach: p_pv ghost box shrunk by 3, fallback 2, bounds 3, u2 curvatures 4, grown mask 5.
    void limitPost_(auto& pv, auto const& state, double const gamma)
    {
        auto const& rho = state.rho;
        mnx_            = static_cast<int>(layout_.ghostEndIndex(rho, Direction::X)) + 1;
        mny_            = static_cast<int>(layout_.ghostEndIndex(rho, Direction::Y)) + 1;
        auto const N    = static_cast<std::size_t>(mnx_ * mny_);
        auto const nan  = std::numeric_limits<double>::quiet_NaN();
        auto at         = [&](int i, int j) { return static_cast<std::size_t>(i * mny_ + j); };
        std::vector<double> pfb(N, nan), ppv(N, nan), raw(N, nan);
        auto ke = [](auto const& r, auto const& mv, int i, int j) {
            double k = 0;
            for (int c = 0; c < 3; ++c)
            {
                double const q = mv(static_cast<Component>(c))(i, j);
                k += q * q;
            }
            return 0.5 * k / r(i, j);
        };
        for (int i = 2; i < mnx_ - 2; ++i)
            for (int j = 2; j < mny_ - 2; ++j)
                pfb[at(i, j)] = (gamma - 1.0) * (eh_[at(i, j)] - ke(rho, state.rhoV, i, j));
        for (int i = 3; i < mnx_ - 3; ++i)
            for (int j = 3; j < mny_ - 3; ++j)
                ppv[at(i, j)]
                    = (gamma - 1.0) * (pv.Etot(i, j) - ke(pv.rho, pv.rhoV, i, j) - m_[at(i, j)]);

        bool const dmp = pvDetector() == 3;
        auto avg = [&](int v, int i, int j) { return v == 0 ? rho(i, j) : pfb[at(i, j)]; };
        auto dmpBad = [&](int v, double const cand, int i, int j) {
            double lo = avg(v, i, j), hi = lo;
            for (int a = -1; a <= 1; ++a)
                for (int b = -1; b <= 1; ++b)
                {
                    lo = std::min(lo, avg(v, i + a, j + b));
                    hi = std::max(hi, avg(v, i + a, j + b));
                }
            double const d = std::max(1.e-3 * (hi - lo), 1.e-8 * std::max(std::abs(hi), std::abs(lo)));
            if (cand >= lo - d && cand <= hi + d)
                return false;
            // u2: a smooth extremum (curvatures of one sign and comparable size) is not troubled
            for (int ax = 0; ax < 2; ++ax)
            {
                double mn = 1e300, mx = -1e300;
                for (int k = -1; k <= 1; ++k)
                {
                    int const ii = i + (ax == 0 ? k : 0), jj = j + (ax == 1 ? k : 0);
                    double const X = ax == 0 ? avg(v, ii + 1, jj) - 2 * avg(v, ii, jj) + avg(v, ii - 1, jj)
                                             : avg(v, ii, jj + 1) - 2 * avg(v, ii, jj) + avg(v, ii, jj - 1);
                    mn = std::min(mn, X);
                    mx = std::max(mx, X);
                }
                double const amax = std::max(std::abs(mn), std::abs(mx));
                double const amin = std::min(std::abs(mn), std::abs(mx));
                bool const ok     = (mn * mx > 0 && amin >= 0.5 * amax) || amax <= d;
                if (!ok)
                    return true;
            }
            return false;
        };
        int const r0 = dmp ? 4 : 3;
        for (int i = r0; i < mnx_ - r0; ++i)
            for (int j = r0; j < mny_ - r0; ++j)
            {
                double const rp = pv.rho(i, j), pp = ppv[at(i, j)];
                bool bad        = !(rp > 0) || !(pp > 0);
                if (dmp && !bad)
                    bad = dmpBad(0, rp, i, j) || dmpBad(1, pp, i, j);
                raw[at(i, j)] = bad ? 1.0 : 0.0;
            }
        mask_.assign(N, nan);
        for (int i = r0 + 1; i < mnx_ - r0 - 1; ++i)
            for (int j = r0 + 1; j < mny_ - r0 - 1; ++j)
                mask_[at(i, j)] = std::max({raw[at(i, j)], raw[at(i - 1, j)], raw[at(i + 1, j)],
                                            raw[at(i, j - 1)], raw[at(i, j + 1)]});
        {
            auto const lo = layout_.AMRBox().lower;
            auto& e       = pvMaskRegistry[{int(lo[0]), int(lo[1])}];
            e.nx          = mnx_;
            e.ny          = mny_;
            e.m           = mask_;
        }
        for (int i = 1; i < mnx_ - 1; ++i)
            for (int j = 1; j < mny_ - 1; ++j)
            {
                double const mk = mask_[at(i, j)];
                if (!(mk == mk))
                {
                    pv.rho(i, j) = nan; // reach guard
                    pv.Etot(i, j) = nan;
                }
                else if (mk > 0)
                {
                    pv.rho(i, j) = rho(i, j);
                    for (int c = 0; c < 3; ++c)
                        pv.rhoV(static_cast<Component>(c))(i, j)
                            = state.rhoV(static_cast<Component>(c))(i, j);
                    pv.Etot(i, j) = eh_[at(i, j)] + m_[at(i, j)];
                }
            }
        for (auto i = layout_.physicalStartIndex(rho, Direction::X);
             i <= layout_.physicalEndIndex(rho, Direction::X); ++i)
            for (auto j = layout_.physicalStartIndex(rho, Direction::Y);
                 j <= layout_.physicalEndIndex(rho, Direction::Y); ++j)
            {
                ++pvLimitStats.total;
                if (mask_[at(int(i), int(j))] > 0)
                    ++pvLimitStats.flagged;
            }
    }

    void buildMask_(auto const& state, double const gamma)
    {
        auto const& rho  = state.rho;
        auto const& Etot = state.Etot;
        auto const& mx   = state.rhoV(Component::X);
        auto const& my   = state.rhoV(Component::Y);
        auto const& mz   = state.rhoV(Component::Z);
        auto const& Bx   = state.B(Component::X);
        auto const& By   = state.B(Component::Y);
        auto const& Bz   = state.B(Component::Z);
        mnx_ = static_cast<int>(layout_.ghostEndIndex(rho, Direction::X)) + 1;
        mny_ = static_cast<int>(layout_.ghostEndIndex(rho, Direction::Y)) + 1;
        auto const N = static_cast<std::size_t>(mnx_ * mny_);
        auto const nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> r(N), p(N), pb(N), raw(N, nan);
        auto at = [&](int i, int j) { return static_cast<std::size_t>(i * mny_ + j); };
        for (int i = 0; i < mnx_; ++i)
            for (int j = 0; j < mny_; ++j)
            {
                // averages; B cell centre = 2-point face average (the branch's faceXToCellCenter())
                double const bx = 0.5 * (Bx(i, j) + Bx(i + 1, j));
                double const by = 0.5 * (By(i, j) + By(i, j + 1));
                double const bz = Bz(i, j);
                double const rr = rho(i, j);
                double const ke = 0.5 * (mx(i, j) * mx(i, j) + my(i, j) * my(i, j) + mz(i, j) * mz(i, j)) / rr;
                double const me = 0.5 * (bx * bx + by * by + bz * bz);
                r[at(i, j)]  = rr;
                pb[at(i, j)] = me;
                p[at(i, j)]  = (gamma - 1.0) * (Etot(i, j) - ke - me);
            }
        double const thr = pvLimitThreshold();
        constexpr double eps = 1.e-12;
        bool const lohner    = pvDetector() == 2;
        auto sens = [&](std::vector<double> const& u, int i, int j) {
            double eta = 0;
            for (int ax = 0; ax < 2; ++ax)
            {
                double const um = ax == 0 ? u[at(i - 1, j)] : u[at(i, j - 1)];
                double const u0 = u[at(i, j)];
                double const up = ax == 0 ? u[at(i + 1, j)] : u[at(i, j + 1)];
                double const num = std::abs(up - 2.0 * u0 + um);
                double const sum = std::abs(up) + 2.0 * std::abs(u0) + std::abs(um);
                if (lohner) // Loehner: first differences in the denominator, 0.01 noise filter
                    eta = std::max(eta, num / (std::abs(up - u0) + std::abs(u0 - um) + 0.01 * sum
                                               + 1.e-300));
                else
                    eta = std::max(eta, num / (sum + eps));
            }
            return eta;
        };
        for (int i = 1; i < mnx_ - 1; ++i)
            for (int j = 1; j < mny_ - 1; ++j)
            {
                double const eta = std::max({sens(r, i, j), sens(p, i, j), sens(pb, i, j)});
                raw[at(i, j)]    = eta > thr ? 1.0 : 0.0;
            }
        mask_.assign(N, nan);
        for (int i = 2; i < mnx_ - 2; ++i)
            for (int j = 2; j < mny_ - 2; ++j)
                mask_[at(i, j)] = std::max({raw[at(i, j)], raw[at(i - 1, j)], raw[at(i + 1, j)],
                                            raw[at(i, j - 1)], raw[at(i, j + 1)]});
        {
            auto const lo = layout_.AMRBox().lower;
            auto& e       = pvMaskRegistry[{int(lo[0]), int(lo[1])}];
            e.nx          = mnx_;
            e.ny          = mny_;
            e.m           = mask_;
        }
        // flagged fraction on the physical cells
        for (auto i = layout_.physicalStartIndex(rho, Direction::X);
             i <= layout_.physicalEndIndex(rho, Direction::X); ++i)
            for (auto j = layout_.physicalStartIndex(rho, Direction::Y);
                 j <= layout_.physicalEndIndex(rho, Direction::Y); ++j)
            {
                ++pvLimitStats.total;
                if (mask_[at(int(i), int(j))] > 0)
                    ++pvLimitStats.flagged;
            }
    }

    template<PointValueConversionMode mode, typename Field>
    auto getCellCentered(Field const& f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl
            = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
        return f(index) + layout_.lapl(f, index) * wlapl;
    }

    template<auto direction, PointValueConversionMode mode, typename Field>
    auto getFaceCentered(Field const& f, MeshIndex<dimension> index) const
    {
        static constexpr auto wlapl
            = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
        return f(index) + layout_.template transverseLapl<direction>(f, index) * wlapl;
    }

    template<auto direction, PointValueConversionMode mode, typename Field>
    auto getEdgeCentered(Field const& f, MeshIndex<dimension> index) const
    {
        constexpr bool has_direction = (dimension == 3)
                                       || (dimension == 2 && direction != Direction::Z)
                                       || (dimension == 1 && direction == Direction::X);
        if constexpr (!has_direction)
            return f(index);
        else
        {
            static constexpr auto wlapl
                = (mode == PointValueConversionMode::ToPointValue) ? -1. / 24. : 1. / 24.;
            return f(index) + layout_.template directionalLapl<direction>(f, index) * wlapl;
        }
    }

    GridLayout layout_;
};
} // namespace PHARE::core

#endif
