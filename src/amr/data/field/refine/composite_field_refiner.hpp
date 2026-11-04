#ifndef PHARE_COMPOSITE_FIELD_REFINER_HPP
#define PHARE_COMPOSITE_FIELD_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/types.hpp"
#include "core/models/quantities/mhd_quantities.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "field_refiner_kernel.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>


namespace PHARE::amr
{

//! Tag: the unlimited composite fill (compile-time stencils).
struct NoLimiter
{
};

/**
 * @brief Tag: the WENO-limited composite fill.
 *
 * Each dual ±¼ row blends its two one-sided slopes with Jiang-Shu weights (WENO-AO, linear
 * weights ½,½, q = 2) instead of using their central mean, after Balsara, Samantaray &
 * Subramanian (2023), doi:10.1007/s42967-021-00182-x. Limiting acts on the 1-D rows, which keep
 * summing to 1 with a σ-antisymmetric correction, so the children still mean back to the coarse
 * value: conservation and the coarse face means that ∇·B relies on hold limited or not.
 */
struct WenoLimiter
{
};

// EXPERIMENT HACK (exp/refine-ssi-weno, never for a PR): env-switched slope rule of the limited
// dual row, and shared-smoothness-indicator (SSI) WENO for the five MHD hydro conserved variables.
//   PHARE_REFINE_SLOPE = weno (default) | minmod | vanleer   (per-variable rows)
//   PHARE_REFINE_SSI   = set => hydro rows use ONE weight set per (coarse anchor, direction) from
//                        the average over rho,rhoVx,rhoVy,rhoVz,Etot of IS/(eps + sum u^2 over the
//                        3-cell stencil), as feature/shared-smoothness-indicators does. Needs the
//                        messenger's two-pass fill (mode Gather, then Apply); Off = per-variable.
namespace ssi_hack
{
    enum class Mode { Off, Gather, Apply };
    inline Mode mode = Mode::Off;
    using Key        = std::array<int, 4>; // anchor (padded to 3) + direction
    using Slots      = std::array<std::array<double, 2>, 5>;
    inline std::map<Key, Slots> cache;
    inline long hits = 0, misses = 0;

    inline int slope()
    {
        static int const s = [] {
            char const* e = std::getenv("PHARE_REFINE_SLOPE");
            std::string const v{e ? e : "weno"};
            return v == "minmod"   ? 1
                   : v == "vanleer" ? 2
                   : v == "weno3z"  ? 3
                   : v == "minmax"  ? 4
                                    : 0;
        }();
        return s;
    }
    // DIAG (exp/refine-diag): fill context (set by the MHD messenger), ablation knobs, counters.
    //   PHARE_REFINE_LIMIT_GHOSTS=0      : ghost / time-interpolated fills (fillCtx 0) unlimited
    //   PHARE_REFINE_LIMIT_SCOPE=state   : only rho, rhoV, Etot, B limited (fluxes, E, J unlimited)
    //   PHARE_REFINE_TRACE=1             : per (fillCtx, quantity) row counters on stderr at exit
    //   PHARE_WENO3Z_EPS                 : epsilon of the weno3z rule (default 1e-40)
    inline int fillCtx = 0; // 0 = ghost / other fills, 1 = initLevel, 2 = regrid
    inline int curQ    = 0;
    inline bool envIs_(char const* name, char const* val)
    {
        char const* e = std::getenv(name);
        return e and std::string{e} == val;
    }
    inline bool limitGhosts()
    {
        static bool const b = not envIs_("PHARE_REFINE_LIMIT_GHOSTS", "0");
        return b;
    }
    inline bool stateOnly()
    {
        static bool const b = envIs_("PHARE_REFINE_LIMIT_SCOPE", "state");
        return b;
    }
    inline bool trace()
    {
        static bool const b = std::getenv("PHARE_REFINE_TRACE") != nullptr;
        return b;
    }
    inline double weno3zEps()
    {
        static double const e = [] {
            char const* s = std::getenv("PHARE_WENO3Z_EPS");
            return s ? std::atof(s) : 1e-40;
        }();
        return e;
    }
    struct Count
    {
        long rows = 0, devW = 0, devS = 0, flat = 0, ext = 0;
        double sumRelDev = 0.;
    };
    inline std::array<std::array<Count, 64>, 3> counts{};
    struct TraceDump
    {
        ~TraceDump()
        {
            if (!trace())
                return;
            for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t q = 0; q < 64; ++q)
                    if (auto const& k = counts[c][q]; k.rows > 0)
                        std::cerr << "LIMTRACE ctx=" << c << " q=" << q << " rows=" << k.rows
                                  << " devW=" << k.devW << " devS=" << k.devS << " flat=" << k.flat
                                  << " ext=" << k.ext << " meanRelDev=" << k.sumRelDev / k.rows
                                  << std::endl;
        }
    };
    inline TraceDump traceDump;
    // count one limited row: (a, b) the one-sided slope weights, central = a = b = 1/2
    inline void countRow(double const a, double const b, double const left, double const right)
    {
        if (!trace())
            return;
        auto& k = counts[fillCtx][curQ];
        ++k.rows;
        double const scale = std::abs(left) + std::abs(right);
        if (scale == 0.)
        {
            ++k.flat;
            return;
        }
        if (left * right <= 0.)
            ++k.ext;
        if (std::abs(a - 0.5) > 1e-3 or std::abs(b - 0.5) > 1e-3)
            ++k.devW;
        double const rel = std::abs(a * left + b * right - 0.5 * (left + right)) / scale;
        k.sumRelDev += rel;
        if (rel > 1e-3)
            ++k.devS;
    }
    template<typename Q>
    int qIndexOf(Q const q)
    {
        if constexpr (std::is_enum_v<Q>)
            return static_cast<int>(q) & 63;
        else
            return 63;
    }
    template<typename Q>
    bool isState(Q const q)
    {
        if constexpr (std::is_same_v<Q, core::MHDQuantity::Scalar>)
        {
            using S = core::MHDQuantity::Scalar;
            switch (q)
            {
                case S::rho:
                case S::rhoVx:
                case S::rhoVy:
                case S::rhoVz:
                case S::Etot:
                case S::Bx:
                case S::By:
                case S::Bz: return true;
                default: return false;
            }
        }
        return true;
    }

    inline bool ssiOn()
    {
        static bool const b = std::getenv("PHARE_REFINE_SSI") != nullptr;
        return b;
    }
    template<typename Q>
    int slotOf(Q const q)
    {
        if constexpr (std::is_same_v<Q, core::MHDQuantity::Scalar>)
        {
            using S = core::MHDQuantity::Scalar;
            switch (q)
            {
                case S::rho: return 0;
                case S::rhoVx: return 1;
                case S::rhoVy: return 2;
                case S::rhoVz: return 3;
                case S::Etot: return 4;
                default: return -1;
            }
        }
        return -1;
    }
    template<typename P>
    Key keyOf(P const& anchor, std::size_t d)
    {
        Key k{0, 0, 0, static_cast<int>(d)};
        for (std::size_t i = 0; i < anchor.size(); ++i)
            k[i] = anchor[i];
        return k;
    }
    inline Slots& slotsAt(Key const& k)
    {
        auto it = cache.find(k);
        if (it == cache.end())
        {
            Slots nan;
            for (auto& v : nan)
                v = {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN()};
            it = cache.emplace(k, nan).first;
        }
        return it->second;
    }
} // namespace ssi_hack

namespace detail
{
    /**
     * @brief The composite refinement fill: the order-`order` stencil tables, and the loop that
     * applies them to every fine index of a region.
     *
     * For ratio 2, each fine index maps to a coarse anchor I = floor(f/2) (toCoarseIndex) and a
     * parity p = f − 2I ∈ {0,1} per direction. Per direction the 1-D weight row is chosen by
     * centering+parity:
     *   - primal, p=0 : exact copy (coincident node)            → {offset 0, 1.0}
     *   - primal, p=1 : half-point midpoint (Primitive A.2)     → directionalInterp<dir,
     *                                                             PrimalToDual>
     *   - dual,   p   : ±¼ child ladder (Primitive B)           → directionalProlongation<dir,
     *                                                             σ=2p−1>
     * (both at <order>). The multi-D stencil is the outer product of the rows (consteval
     * tensorProduct), which sizes each multi-D stencil exactly from its rows' own lengths (copy=1,
     * half-point=2, dual-σ±=3) — no padding to a common length; the full set of centering×parity
     * combinations (4^dim of them) is a compile-time table, indexed at runtime per fine index by
     * centering+parity. Offsets are relative to the anchor I; values are gathered from the coarse
     * field.
     *
     * Stateless, with no virtual of its own: the kernels below are thin IFieldRefineKernel
     * adapters over it, and differ only in the region they hand it. So a kernel costs one virtual
     * call per overlap box and none per fine index.
     *
     * With WenoLimiter the dual rows depend on the data, so they are built per fine index from the
     * coarse line through the anchor along each direction, then tensor-producted at runtime. They
     * read the same ±1 coarse cells as the unlimited rows, and reduce to them exactly on linear
     * data.
     */
    template<typename GridLayoutT, typename FieldT, std::size_t order, typename Limiter = NoLimiter>
    struct CompositeFill
    {
        static constexpr std::size_t dimension = GridLayoutT::dimension;

        static_assert(order == 2, "composite refiner ladder is order 2 (Linear)");
        static_assert(std::is_same_v<Limiter, NoLimiter> or std::is_same_v<Limiter, WenoLimiter>,
                      "Limiter must be NoLimiter or WenoLimiter");

        // order 2 reads ±1 coarse cell (max |offset| over both 1-D primitives).
        static constexpr int coarseStencilWidth() { return order / 2; }

        /**
         * @brief Fill every fine index of `region` from the coarse field.
         *
         * `region` is the caller's to choose: it must lie inside the destination allocation, and
         * the kernel writes only the fine indices it contains.
         */
        static void fill(FieldT const& sourceField, FieldT& destinationField,
                         SAMRAI::hier::Box const& region,
                         std::array<core::QtyCentering, dimension> const& centering,
                         SAMRAI::hier::Box const& destFieldBox,
                         SAMRAI::hier::Box const& sourceFieldBox,
                         SAMRAI::hier::IntVector const& ratio)
        {
            for (std::size_t d = 0; d < dimension; ++d)
                if (ratio(d) != 2)
                    throw std::runtime_error(
                        "composite field refinement supports refinement ratio 2 only");

            // Centering is fixed over the box, parity varies per fine index. The 2-bits-per-
            // direction index built below (bit0 = parity, bit1 = dual) selects the matching
            // compile-time stencil.
            [[maybe_unused]] int const slot = ssi_hack::slotOf(sourceField.physicalQuantity());
            [[maybe_unused]] bool const ssi
                = ssi_hack::ssiOn() and slot >= 0 and ssi_hack::mode != ssi_hack::Mode::Off;
            ssi_hack::curQ = ssi_hack::qIndexOf(sourceField.physicalQuantity());
            [[maybe_unused]] bool const limitThis
                = not((ssi_hack::fillCtx == 0 and not ssi_hack::limitGhosts())
                      or (ssi_hack::stateOnly()
                          and not ssi_hack::isState(sourceField.physicalQuantity())));

            for (auto const fineIndex : phare_box_from<dimension>(region))
            {
                auto const anchor = toCoarseIndex<dimension>(fineIndex);

                if constexpr (std::is_same_v<Limiter, WenoLimiter>)
                    if (ssi and ssi_hack::mode == ssi_hack::Mode::Gather)
                    {
                        auto const aLoc = AMRToLocal(anchor, sourceFieldBox);
                        if (ssi_hack::slope() == 4) // minmax: share the per-direction scale
                        {
                            std::array<bool, dimension> dual{};
                            for (std::size_t d = 0; d < dimension; ++d)
                                dual[d] = centering[d] == core::QtyCentering::dual;
                            auto const sf = minmaxScale_(sourceField, aLoc, dual);
                            for (std::size_t d = 0; d < dimension; ++d)
                                if (dual[d])
                                    ssi_hack::slotsAt(ssi_hack::keyOf(anchor, d))[slot]
                                        = {sf[d], 0.};
                            continue;
                        }
                        [&]<std::size_t... d>(std::index_sequence<d...>) {
                            ((centering[d] == core::QtyCentering::dual
                                  ? gatherIS_<d>(
                                        sourceField, aLoc,
                                        ssi_hack::slotsAt(ssi_hack::keyOf(anchor, d))[slot])
                                  : void()),
                             ...);
                        }(std::make_index_sequence<dimension>{});
                        continue;
                    }

                std::size_t combined = 0; // 2 bits/dir: (dual<<1)|parity
                for (std::size_t d = 0; d < dimension; ++d)
                {
                    auto const parity = static_cast<std::size_t>(fineIndex[d] - 2 * anchor[d]);
                    std::size_t const dualBit
                        = (centering[d] == core::QtyCentering::dual) ? 1u : 0u;
                    combined |= ((dualBit << 1) | parity) << (2u * d);
                }

                auto const anchorLocal = AMRToLocal(anchor, sourceFieldBox);
                auto const fineLocal   = AMRToLocal(fineIndex, destFieldBox);

                double value;
                if constexpr (std::is_same_v<Limiter, WenoLimiter>)
                    value = limitThis ? limitedGather_(sourceField, anchorLocal, combined,
                                                       ssi ? &anchor : nullptr)
                                      : gatherers_[combined](sourceField, anchorLocal);
                else
                    value = gatherers_[combined](sourceField, anchorLocal);

                assignFine_(destinationField, fineLocal, value);
            }
        }

    private:
        using GridLayoutImpl = GridLayoutT::implT;
        using Point_t        = core::Point<int, dimension>;
        using WeightPoint_t  = core::WeightPoint<dimension>;

        // 2 bits of choice per direction (dual<<1 | parity) → 4^dim combinations.
        static constexpr std::size_t nCombined = std::size_t{1} << (2 * dimension);

        // 1-D weight row for direction d and the compile-time choice (dual<<1 | parity).
        template<std::size_t d, std::size_t choice>
        static consteval auto oneDRow_()
        {
            static_assert(choice < 4, "choice is 2 bits: dual << 1 | parity");

            if constexpr (choice == 0) // primal, parity 0: coincident node, exact copy
                return std::array{WeightPoint_t{Point_t{}, 1.0}};
            else if constexpr (choice == 1) // primal, parity 1: half-point midpoint
                return GridLayoutImpl::template directionalInterp<
                    d, GridLayoutImpl::InterpDir::PrimalToDual, order>();
            else if constexpr (choice == 2) // dual, parity 0: σ = −1 child
                return GridLayoutImpl::template directionalProlongation<d, -1, order>();
            else // choice == 3 : dual, parity 1: σ = +1 child
                return GridLayoutImpl::template directionalProlongation<d, +1, order>();
        }

        // full multi-D stencil (exact size) for the combined choice index (2 bits per direction).
        template<std::size_t combined>
        static consteval auto makeStencil_()
        {
            static_assert(dimension >= 1 and dimension <= 3);

            if constexpr (dimension == 1)
                return oneDRow_<0, combined & 3u>();
            else if constexpr (dimension == 2)
                return GridLayoutImpl::template tensorProduct<0, 1>(
                    oneDRow_<0, combined & 3u>(), oneDRow_<1, (combined >> 2) & 3u>());
            else
                return GridLayoutImpl::template tensorProduct<0, 1, 2>(
                    oneDRow_<0, combined & 3u>(), oneDRow_<1, (combined >> 2) & 3u>(),
                    oneDRow_<2, (combined >> 4) & 3u>());
        }

        // gather the coarse contributions for one fine index using the exact stencil of choice
        // `combined` (its size is baked in at compile time — no zero-coef points iterated).
        template<std::size_t combined>
        static double gatherStencil_(FieldT const& src, Point_t const& anchorLocal)
        {
            return core::sum_from(makeStencil_<combined>(), [&](auto const& w) {
                return w.coef * src(anchorLocal + w.indexes);
            });
        }

        // runtime centering+parity → compile-time stencil: one gather fn per combination,
        // dispatched by a plain index.
        using Gatherer_t                 = double (*)(FieldT const&, Point_t const&);
        static constexpr auto gatherers_ = core::for_N_make_array<nCombined>(
            [](auto ic) { return Gatherer_t{&gatherStencil_<ic()>}; });

        // ---- WENO-limited path
        // -------------------------------------------------------------------

        // one 1-D row of at most 3 points (offsets along its own direction)
        struct Row_
        {
            std::array<int, 3> offset{};
            std::array<double, 3> weight{};
            std::size_t size = 0;
        };

        // a compile-time row (copy or half-point) as a Row_
        template<std::size_t d, std::size_t choice>
        static consteval Row_ fixedRow_()
        {
            auto const row = oneDRow_<d, choice>();
            Row_ out;
            for (std::size_t k = 0; k < row.size(); ++k)
            {
                out.offset[k] = row[k].indexes[d];
                out.weight[k] = row[k].coef;
            }
            out.size = row.size();
            return out;
        }

        template<std::size_t d>
        static double alongAxis_(FieldT const& src, Point_t const& anchorLocal, int const k)
        {
            Point_t offset{};
            offset[d] = k;
            return src(anchorLocal + offset);
        }

        // Dual row of child σ: ū + σ·m/4 with m = wL·(ū − ū₋₁) + wR·(ū₊₁ − ū), the WENO blend
        // of the two one-sided slopes. Weights ∝ γ/IS² with γ = ½ each and IS = slope², i.e.
        // wL = IS_R²/(IS_L² + IS_R²); equal slopes give wL = wR = ½, the unlimited row
        // {−σ/8, 1, σ/8}. Two flat slopes (0/0) fall back to that same row.
        // normalised Jiang-Shu indicators of this variable at the anchor, along d (SSI gather)
        template<std::size_t d>
        static void gatherIS_(FieldT const& src, Point_t const& anchorLocal,
                              std::array<double, 2>& out)
        {
            double const L     = alongAxis_<d>(src, anchorLocal, -1);
            double const C     = alongAxis_<d>(src, anchorLocal, 0);
            double const R     = alongAxis_<d>(src, anchorLocal, 1);
            double const norm2 = 1.e-40 + L * L + C * C + R * R;
            out                = {(C - L) * (C - L) / norm2, (R - C) * (R - C) / norm2};
        }

        template<typename AnchorT>
        static double wenoLeftWeight_(double const isL, double const isR)
        {
            double const den = isL * isL + isR * isR;
            return den > 0. ? (isR * isR) / den : 0.5;
        }

        // Row of child σ: ū + σ·m/4 with m = a·(ū − ū₋₁) + b·(ū₊₁ − ū). The slope rule picks (a,
        // b):
        //   weno    : a = wL, b = wR (Jiang-Shu, q=2), per variable or from the shared SSI
        //   minmod  : one-sided slope of smaller magnitude, 0 at an extremum
        //   vanleer : harmonic mean 2AB/(A+B) = (B/(A+B))·A + (A/(A+B))·B, 0 at an extremum
        template<std::size_t d, typename AnchorT>
        static Row_ wenoDualRow_(double const sigma, FieldT const& src, Point_t const& anchorLocal,
                                 AnchorT const* anchorAMR)
        {
            double const left
                = alongAxis_<d>(src, anchorLocal, 0) - alongAxis_<d>(src, anchorLocal, -1);
            double const right
                = alongAxis_<d>(src, anchorLocal, 1) - alongAxis_<d>(src, anchorLocal, 0);

            double a, b;
            if (ssi_hack::slope() == 1) // minmod
            {
                if (left * right <= 0.)
                    a = b = 0.;
                else if (std::abs(left) <= std::abs(right))
                    a = 1., b = 0.;
                else
                    a = 0., b = 1.;
            }
            else if (ssi_hack::slope() == 2) // van Leer
            {
                if (left * right <= 0.)
                    a = b = 0.;
                else
                {
                    a = right / (left + right);
                    b = left / (left + right);
                }
            }
            else
            {
                double isL = left * left;
                double isR = right * right;
                if (anchorAMR)
                {
                    auto const& slots = ssi_hack::slotsAt(ssi_hack::keyOf(*anchorAMR, d));
                    double sL = 0., sR = 0.;
                    bool complete = true;
                    for (auto const& v : slots)
                    {
                        complete = complete and not std::isnan(v[0]);
                        sL += v[0];
                        sR += v[1];
                    }
                    if (complete)
                    {
                        isL = sL / 5.;
                        isR = sR / 5.;
                        ++ssi_hack::hits;
                    }
                    else
                        ++ssi_hack::misses;
                }
                if (ssi_hack::slope() == 3 and not anchorAMR)
                {
                    // WENO3-Z (Don & Borges 2013, JCP 250:347, eq. for r=2): beta_k = slope_k^2,
                    // tau = |beta_L - beta_R|, alpha_k = d_k (1 + (tau/(beta_k + eps))^p),
                    // d_k = 1/2 (the central row), p = 2.
                    double const eps = ssi_hack::weno3zEps();
                    double const tau = std::abs(isL - isR);
                    double const rL  = tau / (isL + eps);
                    double const rR  = tau / (isR + eps);
                    double const aL  = 0.5 * (1. + rL * rL);
                    double const aR  = 0.5 * (1. + rR * rR);
                    a                = aL / (aL + aR);
                }
                else
                    a = wenoLeftWeight_<AnchorT>(isL, isR);
                b = 1. - a;
            }
            ssi_hack::countRow(a, b, left, right);

            return Row_{
                {-1, 0, 1}, {-sigma * a / 4., 1. + sigma * (a - b) / 4., sigma * b / 4.}, 3};
        }

        // Only the dual rows are limited. The primal copy needs nothing, and the order-2
        // half-point is a convex two-point mean with a single admissible stencil, so there is
        // nothing to blend.
        template<std::size_t d, typename AnchorT>
        static Row_ limitedRow_(std::size_t const choice, FieldT const& src,
                                Point_t const& anchorLocal, AnchorT const* anchorAMR)
        {
            static constexpr Row_ copy      = fixedRow_<d, 0>();
            static constexpr Row_ halfPoint = fixedRow_<d, 1>();

            switch (choice)
            {
                case 0: return copy;
                case 1: return halfPoint;
                case 2: return wenoDualRow_<d>(-1., src, anchorLocal, anchorAMR);
                default: return wenoDualRow_<d>(+1., src, anchorLocal, anchorAMR);
            }
        }

        static Row_ fixedRowRt_(std::size_t const d, std::size_t const choice)
        {
            return [&]<std::size_t... dd>(std::index_sequence<dd...>) {
                Row_ r{};
                ((d == dd ? (r = choice == 0 ? fixedRow_<dd, 0>() : fixedRow_<dd, 1>()), 0 : 0),
                 ...);
                return r;
            }(std::make_index_sequence<dimension>{});
        }

        template<std::size_t d>
        static double tensorGather_(std::array<Row_, dimension> const& rows, FieldT const& src,
                                    Point_t const& anchorLocal, Point_t& offset)
        {
            double sum = 0.;
            for (std::size_t k = 0; k < rows[d].size; ++k)
            {
                offset[d] = rows[d].offset[k];
                if constexpr (d + 1 == dimension)
                    sum += rows[d].weight[k] * src(anchorLocal + offset);
                else
                    sum += rows[d].weight[k] * tensorGather_<d + 1>(rows, src, anchorLocal, offset);
            }
            return sum;
        }

        // AMReX-style minmax scale per dual direction (AMReX_MFInterp_3D_C.H:7-117,
        // mf_cell_cons_lin_interp_limit_minmax_llslope): MC slope s_d, then one alpha so that
        // u0 +- sum_d |s_d|/4 stays inside [min, max] of the 3^d neighbourhood (dual directions
        // only), and sf_d = alpha s_d / central_d (sf_d = 1 when the central slope is 0).
        static std::array<double, dimension> minmaxScale_(FieldT const& src,
                                                          Point_t const& anchorLocal,
                                                          std::array<bool, dimension> const& dual)
        {
            std::array<double, dimension> s{}, dc{}, sf{};
            double const u0 = src(anchorLocal);
            double dumax    = 0.;
            for (std::size_t d = 0; d < dimension; ++d)
            {
                sf[d] = 1.;
                if (!dual[d])
                    continue;
                Point_t e{};
                e[d]            = 1;
                double const up = src(anchorLocal + e);
                double const um = src(anchorLocal - e);
                dc[d]           = 0.5 * (up - um);
                double const df = 2. * (up - u0);
                double const db = 2. * (u0 - um);
                double sl       = (df * db >= 0.) ? std::min(std::abs(df), std::abs(db)) : 0.;
                sl              = std::copysign(std::min(sl, std::abs(dc[d])), dc[d]);
                s[d]            = sl;
                dumax += std::abs(sl) / 4.;
            }
            double umin = u0, umax = u0;
            Point_t off{};
            auto scan = [&](auto&& self, std::size_t d) -> void {
                if (d == dimension)
                {
                    double const v = src(anchorLocal + off);
                    umin           = std::min(umin, v);
                    umax           = std::max(umax, v);
                    return;
                }
                int const lo = dual[d] ? -1 : 0, hi = dual[d] ? 1 : 0;
                for (int k = lo; k <= hi; ++k)
                {
                    off[d] = k;
                    self(self, d + 1);
                }
                off[d] = 0;
            };
            scan(scan, 0);
            double alpha = 1.;
            if (dumax > 0.)
            {
                if (dumax * alpha > umax - u0)
                    alpha = (umax - u0) / dumax;
                if (dumax * alpha > u0 - umin)
                    alpha = (u0 - umin) / dumax;
            }
            for (std::size_t d = 0; d < dimension; ++d)
                if (dual[d] and dc[d] != 0.)
                    sf[d] = alpha * s[d] / dc[d];
            return sf;
        }

        template<typename AnchorT>
        static double limitedGather_(FieldT const& src, Point_t const& anchorLocal,
                                     std::size_t const combined, AnchorT const* anchorAMR)
        {
            if (ssi_hack::slope() == 4)
            {
                std::array<bool, dimension> dual{};
                for (std::size_t d = 0; d < dimension; ++d)
                    dual[d] = ((combined >> (2u * d)) & 2u) != 0u;
                auto sf = minmaxScale_(src, anchorLocal, dual);
                if (anchorAMR) // shared over the 5 hydro variables: min of their scales
                    for (std::size_t d = 0; d < dimension; ++d)
                    {
                        if (!dual[d])
                            continue;
                        auto const& slots = ssi_hack::slotsAt(ssi_hack::keyOf(*anchorAMR, d));
                        double m      = 1.;
                        bool complete = true;
                        for (auto const& v : slots)
                        {
                            complete = complete and not std::isnan(v[0]);
                            m        = std::min(m, v[0]);
                        }
                        if (complete)
                        {
                            sf[d] = m;
                            ++ssi_hack::hits;
                        }
                        else
                            ++ssi_hack::misses;
                    }
                std::array<Row_, dimension> rows;
                for (std::size_t d = 0; d < dimension; ++d)
                {
                    auto const choice = (combined >> (2u * d)) & 3u;
                    if (choice < 2)
                        rows[d] = choice == 0 ? fixedRowRt_(d, 0) : fixedRowRt_(d, 1);
                    else
                    {
                        double const sigma = choice == 2 ? -1. : 1.;
                        Point_t e{};
                        e[d]              = 1;
                        double const left = src(anchorLocal) - src(anchorLocal - e);
                        double const right = src(anchorLocal + e) - src(anchorLocal);
                        ssi_hack::countRow(sf[d] / 2., sf[d] / 2., left, right);
                        rows[d] = Row_{{-1, 0, 1}, {-sigma * sf[d] / 8., 1., sigma * sf[d] / 8.}, 3};
                    }
                }
                Point_t offset{};
                return tensorGather_<0>(rows, src, anchorLocal, offset);
            }
            auto const rows = [&]<std::size_t... d>(std::index_sequence<d...>) {
                return std::array<Row_, dimension>{
                    limitedRow_<d>((combined >> (2u * d)) & 3u, src, anchorLocal, anchorAMR)...};
            }(std::make_index_sequence<dimension>{});

            Point_t offset{};
            return tensorGather_<0>(rows, src, anchorLocal, offset);
        }

        // preserve the legacy NaN-guard: only fill fine indices not already set
        static void assignFine_(FieldT& dst, Point_t const& fineLocal, double const value)
        {
            if (auto& dst_val = dst(fineLocal); std::isnan(dst_val))
                dst_val = value;
        }
    };

} // namespace detail


/**
 * @brief Runtime field-refinement kernel: the composite fill over exactly the overlap SAMRAI hands
 * it.
 *
 * Every quantity but B refines this way. B needs its gather widened to whole coarse cells, which
 * is its own kernel (magnetic_composite_refiner.hpp).
 */
template<typename GridLayoutT, typename FieldT, std::size_t order, typename Limiter = NoLimiter>
class CompositeFieldRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using Fill = detail::CompositeFill<GridLayoutT, FieldT, order, Limiter>;

public:
    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        Fill::fill(sourceField, destinationField, intersectionBox, centering, destFieldBox,
                   sourceFieldBox, ratio);
    }

    int coarseStencilWidth() const override { return Fill::coarseStencilWidth(); }
};


// ---- factory (declared in field_refiner_kernel.hpp) ---------------------------------------------

// FieldRefinementOrder has a single enumerator and RefinementConfig::FROM is the only place a
// dict value is validated into it, so there is nothing to branch on. A second order adds a switch
// here, one case per compile-time stencil.
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel([[maybe_unused]] FieldRefinementOrder const order,
                 FieldRefinementLimiter const limiter)
{
    constexpr auto linear = static_cast<std::size_t>(FieldRefinementOrder::Linear);
    if (limiter == FieldRefinementLimiter::WENO)
        return std::make_unique<CompositeFieldRefiner<GridLayoutT, FieldT, linear, WenoLimiter>>();
    return std::make_unique<CompositeFieldRefiner<GridLayoutT, FieldT, linear>>();
}


} // namespace PHARE::amr


#endif
