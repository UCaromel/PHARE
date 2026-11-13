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

#include <algorithm>
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
#include <unordered_map>
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
                   : v == "cweno"   ? 5
                   : v == "central" ? 6
                   : v == "ao5z"    ? 7
                                    : 0;
        }();
        return s;
    }
    // DIAG (exp/refine-diag): fill context (set by the MHD messenger), ablation knobs, counters.
    //   PHARE_REFINE_LIMIT_GHOSTS=0      : ghost / time-interpolated fills (fillCtx 0) unlimited
    //   PHARE_REFINE_LIMIT_SCOPE=state   : only rho, rhoV, Etot, B limited (fluxes, E, J unlimited)
    //   PHARE_REFINE_TRACE=1             : per (fillCtx, quantity) row counters on stderr at exit
    //   PHARE_WENO3Z_EPS                 : epsilon constant E of weno3z / cweno2 (default 1e-40)
    //   PHARE_WENO_EPSMODE               : fixed | rel | dx1 | dx2 | dx1rel | dx2rel
    //                                      eps = E [* h^k] [* (u-^2+u0^2+u+^2)], h = coarse dx
    //   PHARE_CWENO_D0, _P, _Z, _IS0     : cweno (CWENO3 child averages, Semplice, Coco & Russo
    //                                      2016) central weight d0 (0.5), exponent (2), Z weights
    //                                      (CWENOZ3 tau), IS0=slope drops the curvature term
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
    inline int epsMode()
    {
        static int const m = [] {
            char const* e = std::getenv("PHARE_WENO_EPSMODE");
            std::string const v{e ? e : "fixed"};
            return v == "rel"      ? 1
                   : v == "dx1"    ? 2
                   : v == "dx2"    ? 3
                   : v == "dx1rel" ? 4
                   : v == "dx2rel" ? 5
                   : v == "ref"    ? 6
                                   : 0;
        }();
        return m;
    }
    inline double envD_(char const* name, double const dflt)
    {
        char const* s = std::getenv(name);
        return s ? std::atof(s) : dflt;
    }
    inline double cwenoD0()
    {
        static double const v = envD_("PHARE_CWENO_D0", 0.5);
        return v;
    }
    inline double cwenoP()
    {
        static double const v = envD_("PHARE_CWENO_P", 2.);
        return v;
    }
    inline bool cwenoZ()
    {
        static bool const b = envIs_("PHARE_CWENO_Z", "1");
        return b;
    }
    inline bool cwenoSlopeIS0()
    {
        static bool const b = envIs_("PHARE_CWENO_IS0", "slope");
        return b;
    }
    // PHARE_WENO_EPSMODE=ref: eps = (E * ref)^2, ref = a reference magnitude of the row's quantity
    // at the coarse anchor (rho_0; sqrt(2 rho_0 q_0) for rhoV; q_0; Etot_0), set by the fill around
    // each gather on level-creation hydro fills. NaN = no ref: eps = h^2 (dx2 with E = 1).
    inline double epsRef = std::numeric_limits<double>::quiet_NaN();
    // eps of the weno3z / cweno2 weights for the coarse triple (um, u0, up) along direction d
    inline double wenoEps(double const um, double const u0, double const up, std::size_t const d)
    {
        double const E = weno3zEps();
        int const m    = epsMode();
        double const h = refine_diag::coarseDx[d];
        if (m == 6)
        {
            double const e6 = std::isfinite(epsRef) ? (E * epsRef) * (E * epsRef) : h * h;
            return std::max(e6, std::numeric_limits<double>::min());
        }
        double const hk = (m == 2 or m == 4) ? h : (m == 3 or m == 5) ? h * h : 1.;
        double const sc = (m == 1 or m == 4 or m == 5) ? um * um + u0 * u0 + up * up : 1.;
        double const e  = E * hk * sc;
        return std::max(e, std::numeric_limits<double>::min()); // eps = 0 only on all-zero data

    }
    struct Count
    {
        long rows = 0, devW = 0, devS = 0, flat = 0, ext = 0;
        double sumRelDev = 0., maxRelDev = 0.;
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
                                  << " maxRelDev=" << k.maxRelDev
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
        k.maxRelDev = std::max(k.maxRelDev, rel);
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

    // OUT-OF-BOX (exp/refine-diag, 2026-11-13), all default off:
    //   PHARE_REFINE_ENERGY=prim   : on initLevel / regrid hydro fills, Etot children assembled
    //                                from a limited row on q = Etot - |rhoV|^2/(2 rho):
    //                                E_k = E_0 + (q_k - q_0) + v_0.(m_k - m_0) - |v_0|^2/2 (rho_k -
    //                                rho_0), v_0 = m_0/rho_0. Needs coarse rho, rhoV: cached while
    //                                rho and rhoV are refined (they are filled before Etot).
    //   PHARE_REFINE_TENSOR=seq    : 2D all-dual rows: sequential 1D sweeps, second-sweep weights
    //                                from the first-sweep children, mean of both sweep orders
    //   PHARE_REFINE_TENSOR=nocross: child = u0 + sum_d (1D child_d - u0)
    //   PHARE_REFINE_SLOPE=central : a = b = 1/2 through the limited code path (check)
    // prim = exact kinetic form  E_k = E_0 + (q_k - q_0) + KE_k - mean_siblings KE,
    //        KE_k = |m_k|^2/(2 rho_k);  primlin = the linearised form above.
    inline int energyMode()
    {
        static int const m
            = envIs_("PHARE_REFINE_ENERGY", "prim") ? 1 : envIs_("PHARE_REFINE_ENERGY", "primlin") ? 2 : 0;
        return m;
    }
    inline bool energyPrim() { return energyMode() != 0; }
    inline bool nanTrace()
    {
        static bool const b = std::getenv("PHARE_REFINE_NANTRACE") != nullptr;
        return b;
    }
    inline int nanTraced = 0;
    inline bool ao5z() { return slope() == 7; }
    inline long ao5zFallback = 0, ao5zRows = 0;
    inline int tensorMode() // 0 anchor-weight tensor product, 1 seq, 2 nocross
    {
        static int const m
            = envIs_("PHARE_REFINE_TENSOR", "seq") ? 1 : envIs_("PHARE_REFINE_TENSOR", "nocross") ? 2 : 0;
        return m;
    }
    inline bool forceCentral = false; // central rows through the limited path (NoLimiter + prim)
    struct CoarseHydro
    {
        std::array<double, 5> v{std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()};
        unsigned blockDone = 0; // bit per slot: the 3^d block around this cell is stored
    };
    struct Key3Hash
    {
        std::size_t operator()(std::array<int, 3> const& k) const
        {
            std::size_t h = 1469598103934665603ull;
            for (int const i : k)
                h = (h ^ static_cast<std::size_t>(static_cast<unsigned>(i))) * 1099511628211ull;
            return h;
        }
    };
    inline std::unordered_map<std::array<int, 3>, CoarseHydro, Key3Hash> ecache;
    inline long eApplied = 0, eMissed = 0;
    template<typename P>
    std::array<int, 3> key3Of(P const& p)
    {
        std::array<int, 3> k{0, 0, 0};
        for (std::size_t i = 0; i < p.size(); ++i)
            k[i] = p[i];
        return k;
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
            bool allDual = true;
            for (std::size_t d = 0; d < dimension; ++d)
                allDual = allDual and centering[d] == core::QtyCentering::dual;
            bool const hydroCreate = slot >= 0 and allDual
                                     and (ssi_hack::fillCtx == 1 or ssi_hack::fillCtx == 2);
            bool const ePrim       = ssi_hack::energyPrim() and hydroCreate;
            bool const epsRefOn    = ssi_hack::epsMode() == 6 and hydroCreate;

            for (auto const fineIndex : phare_box_from<dimension>(region))
            {
                auto const anchor = toCoarseIndex<dimension>(fineIndex);

                // gather pass without SSI (PHARE_WENO_EPSMODE=ref): cache coarse hydro, no write
                if (ssi_hack::mode == ssi_hack::Mode::Gather and not ssi)
                {
                    if (ePrim or epsRefOn)
                        cacheCoarseBlock_(sourceField, anchor, AMRToLocal(anchor, sourceFieldBox),
                                          slot);
                    continue;
                }

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

                if (ePrim and slot < 4)
                    cacheCoarseBlock_(sourceField, anchor, anchorLocal, slot);
                if (epsRefOn)
                    ssi_hack::epsRef = epsRefOf_(anchor, slot);

                double value;
                bool done = false;
                if (ePrim and slot == 4)
                {
                    bool const limited = std::is_same_v<Limiter, WenoLimiter> and limitThis;
                    done = energyPrim_(sourceField, anchor, anchorLocal, combined, limited, value);
                    ++(done ? ssi_hack::eApplied : ssi_hack::eMissed);
                }
                if (done)
                    ;
                else if constexpr (std::is_same_v<Limiter, WenoLimiter>)
                    value = limitThis ? limitedGather_(sourceField, anchorLocal, combined,
                                                       ssi ? &anchor : nullptr)
                                      : gatherers_[combined](sourceField, anchorLocal);
                else
                    value = gatherers_[combined](sourceField, anchorLocal);

                ssi_hack::epsRef = std::numeric_limits<double>::quiet_NaN();
                if (std::isnan(value) and ssi_hack::nanTrace() and ssi_hack::nanTraced < 12)
                {
                    ++ssi_hack::nanTraced;
                    std::cerr << "NANTRACE ctx=" << ssi_hack::fillCtx << " q=" << ssi_hack::curQ
                              << " fine=";
                    for (std::size_t d = 0; d < dimension; ++d)
                        std::cerr << fineIndex[d] << ",";
                    std::cerr << " anchor=";
                    for (std::size_t d = 0; d < dimension; ++d)
                        std::cerr << anchor[d] << ",";
                    std::cerr << " region=" << region << " srcBox=" << sourceFieldBox
                              << " dstBox=" << destFieldBox << " dstWasNaN="
                              << std::isnan(destinationField(fineLocal)) << " src[-2..2]^d:";
                    if constexpr (dimension == 2)
                        for (int j = -2; j <= 2; ++j)
                        {
                            std::cerr << " |";
                            for (int i = -2; i <= 2; ++i)
                            {
                                Point_t pt = anchorLocal;
                                pt[0] += i;
                                pt[1] += j;
                                bool const in = pt[0] >= 0 and pt[1] >= 0
                                                and pt[0] < static_cast<int>(sourceField.shape()[0])
                                                and pt[1] < static_cast<int>(sourceField.shape()[1]);
                                std::cerr << " " << (in ? sourceField(pt) : -9e99);
                            }
                        }
                    std::cerr << std::endl;
                }
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
            std::array<int, 5> offset{};
            std::array<double, 5> weight{};
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

        // (2R+1)^dim coarse values around an anchor (R = 1, or 2 for ao5z), indexed like a field
        // with the anchor at (2,..,2)
        static constexpr std::size_t nBlock = dimension == 1 ? 5 : dimension == 2 ? 25 : 125;
        static int blockRadius_()
        {
            return (std::is_same_v<Limiter, WenoLimiter> and ssi_hack::ao5z()) ? 2 : 1;
        }
        struct Block_
        {
            std::array<double, nBlock> v{};
            static std::size_t idx(Point_t const& p)
            {
                std::size_t i = 0, stride = 1;
                for (std::size_t d = 0; d < dimension; ++d, stride *= 5)
                    i += static_cast<std::size_t>(p[d]) * stride;
                return i;
            }
            double operator()(Point_t const& p) const { return v[idx(p)]; }
            double& operator()(Point_t const& p) { return v[idx(p)]; }
        };
        static Point_t blockCentre_()
        {
            Point_t c{};
            for (std::size_t d = 0; d < dimension; ++d)
                c[d] = 2;
            return c;
        }
        // visit the offsets in {-R..R}^dim
        template<typename Fn>
        static void forBlock_(Fn&& fn)
        {
            int const R = blockRadius_(), w = 2 * R + 1;
            std::size_t n = 1;
            for (std::size_t d = 0; d < dimension; ++d)
                n *= static_cast<std::size_t>(w);
            Point_t off{};
            for (std::size_t k = 0; k < n; ++k)
            {
                std::size_t r = k;
                for (std::size_t d = 0; d < dimension; ++d, r /= static_cast<std::size_t>(w))
                    off[d] = static_cast<int>(r % static_cast<std::size_t>(w)) - R;
                fn(off);
            }
        }

        // PHARE_REFINE_ENERGY=prim / PHARE_WENO_EPSMODE=ref: remember the coarse values of a hydro
        // variable on the block of each anchor.
        template<typename AnchorT>
        static void cacheCoarseBlock_(FieldT const& src, AnchorT const& anchor,
                                      Point_t const& anchorLocal, int const slot)
        {
            auto& centre = ssi_hack::ecache[ssi_hack::key3Of(anchor)];
            if (centre.blockDone & (1u << slot))
                return;
            centre.blockDone |= (1u << slot);
            forBlock_([&](Point_t const& off) {
                auto key = ssi_hack::key3Of(anchor);
                for (std::size_t d = 0; d < dimension; ++d)
                    key[d] += off[d];
                ssi_hack::ecache[key].v[static_cast<std::size_t>(slot)] = src(anchorLocal + off);
            });
        }

        // reference magnitudes (rho_0, sqrt(2 rho_0 q_0), q_0, Etot_0) of the coarse cell c
        struct Refs_
        {
            double rho, mom, q, etot;
        };
        static Refs_ refsOf_(std::array<double, 5> const& c)
        {
            double const q = c[4] - 0.5 * (c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / c[0];
            return {c[0], std::sqrt(2. * c[0] * q), q, c[4]};
        }
        template<typename AnchorT>
        static double epsRefOf_(AnchorT const& anchor, int const slot)
        {
            auto const it = ssi_hack::ecache.find(ssi_hack::key3Of(anchor));
            if (it == ssi_hack::ecache.end())
                return std::numeric_limits<double>::quiet_NaN();
            auto const r = refsOf_(it->second.v);
            return slot == 0 ? r.rho : slot == 4 ? r.etot : r.mom;
        }

        // PHARE_REFINE_ENERGY: Etot child from limited rows on q = Etot - |m|^2/(2 rho).
        // rho_k, m_k are recomputed here with the same rows as their own fills.
        template<typename AnchorT>
        static bool energyPrim_(FieldT const& src, AnchorT const& anchor,
                                Point_t const& anchorLocal, std::size_t const combined,
                                bool const limited, double& value)
        {
            Block_ rho, q;
            std::array<Block_, 3> m;
            bool ok = true;
            forBlock_([&](Point_t const& off) {
                auto key = ssi_hack::key3Of(anchor);
                for (std::size_t d = 0; d < dimension; ++d)
                    key[d] += off[d];
                auto const it = ssi_hack::ecache.find(key);
                auto const b  = blockCentre_() + off;
                if (it == ssi_hack::ecache.end())
                {
                    bool inner = true;
                    for (std::size_t d = 0; d < dimension; ++d)
                        inner = inner and off[d] >= -1 and off[d] <= 1;
                    if (inner)
                        ok = false;
                    rho(b) = q(b) = std::numeric_limits<double>::quiet_NaN();
                    for (std::size_t i = 0; i < 3; ++i)
                        m[i](b) = std::numeric_limits<double>::quiet_NaN();
                    return;
                }
                auto const& c = it->second.v;
                rho(b)        = c[0];
                for (std::size_t i = 0; i < 3; ++i)
                    m[i](b) = c[i + 1];
                q(b) = src(anchorLocal + off)
                       - 0.5 * (c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / c[0];
                bool inner = true;
                for (std::size_t d = 0; d < dimension; ++d)
                    inner = inner and off[d] >= -1 and off[d] <= 1;
                if (inner)
                    ok = ok and not std::isnan(q(b));
            });
            if (!ok)
                return false;

            auto const c0        = blockCentre_();
            bool const wasForced = ssi_hack::forceCentral;
            ssi_hack::forceCentral = wasForced or not limited;
            double const rho0 = rho(c0), q0 = q(c0), e0 = src(anchorLocal);
            bool const useRef = ssi_hack::epsMode() == 6;
            Refs_ const refs  = refsOf_({rho0, m[0](c0), m[1](c0), m[2](c0), e0});
            auto const child  = [&](Block_ const& u, std::size_t const comb, double const ref) {
                ssi_hack::epsRef = useRef ? ref : std::numeric_limits<double>::quiet_NaN();
                return limitedGather_(u, c0, comb, static_cast<Point_t const*>(nullptr));
            };
            double const dq = child(q, combined, refs.q) - q0;
            if (ssi_hack::energyMode() == 2) // linearised
            {
                double const drho = child(rho, combined, refs.rho) - rho0;
                double vdm = 0., v2 = 0.;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    double const m0 = m[i](c0);
                    double const v0 = m0 / rho0;
                    vdm += v0 * (child(m[i], combined, refs.mom) - m0);
                    v2 += v0 * v0;
                }
                value = e0 + dq + vdm - 0.5 * v2 * drho;
            }
            else // exact kinetic energy of each sibling
            {
                double keSum = 0., keMine = 0.;
                std::size_t const nSib = std::size_t{1} << dimension;
                for (std::size_t j = 0; j < nSib; ++j)
                {
                    std::size_t comb = 0;
                    for (std::size_t d = 0; d < dimension; ++d)
                        comb |= (2u | ((j >> d) & 1u)) << (2u * d);
                    double const rk = child(rho, comb, refs.rho);
                    double m2 = 0.;
                    for (std::size_t i = 0; i < 3; ++i)
                    {
                        double const mk = child(m[i], comb, refs.mom);
                        m2 += mk * mk;
                    }
                    double const ke = 0.5 * m2 / rk;
                    keSum += ke;
                    if (comb == combined)
                        keMine = ke;
                }
                value = e0 + dq + keMine - keSum / static_cast<double>(nSib);
            }
            ssi_hack::forceCentral = wasForced;
            ssi_hack::epsRef       = std::numeric_limits<double>::quiet_NaN();
            return true;
        }

        // ao5z: order-2 slope with WENO-AO(5,3)-Z weights (big = 5-point quartic indicator standing
        // for the central slope, subs = the three 3-point quadratics), eps = round-off guard.
        // slope = A1 s1 + A2 s2 + A3 s3 (s1 left-biased, s2 central, s3 right-biased slopes).
        static std::array<double, 3> ao5zA_(std::array<double, 5> const& u)
        {
            double const um2 = u[0], um1 = u[1], u0 = u[2], u1 = u[3], u2 = u[4];
            double const m1 = 41. * u1 / 60. - 11. * u2 / 120. - 41. * um1 / 60. + 11. * um2 / 120.;
            double const m2
                = -37. * u0 / 28. + 5. * u1 / 7. - 3. * u2 / 56. + 5. * um1 / 7. - 3. * um2 / 56.;
            double const m3 = -u1 / 6. + u2 / 12. + um1 / 6. - um2 / 12.;
            double const ib = (m1 + m3 / 10.) * (m1 + m3 / 10.) + 13. / 3. * m2 * m2
                              + 781. / 20. * m3 * m3;
            std::array<double, 3> const sl{3. * u0 / 2. - 2. * um1 + um2 / 2., u1 / 2. - um1 / 2.,
                                           -3. * u0 / 2. + 2. * u1 - u2 / 2.};
            std::array<double, 3> const cu{u0 / 2. - um1 + um2 / 2., -u0 + u1 / 2. + um1 / 2.,
                                           u0 / 2. - u1 + u2 / 2.};
            std::array<double, 3> is;
            for (std::size_t i = 0; i < 3; ++i)
                is[i] = sl[i] * sl[i] + 13. / 3. * cu[i] * cu[i];
            double const eps = 1e-28 * u0 * u0 + std::numeric_limits<double>::min();
            double const tau
                = (std::abs(ib - is[0]) + std::abs(ib - is[1]) + std::abs(ib - is[2])) / 3.;
            constexpr double g0 = 0.85, g = 0.05;
            double const r0 = tau / (ib + eps);
            double const a0 = g0 * (1. + r0 * r0);
            std::array<double, 3> a;
            double S = a0;
            for (std::size_t i = 0; i < 3; ++i)
            {
                double const r = tau / (is[i] + eps);
                a[i]           = g * (1. + r * r);
                S += a[i];
            }
            double const w0 = a0 / S / g0;
            std::array<double, 3> A{a[0] / S - w0 * g, a[1] / S + w0 * (1. - g), a[2] / S - w0 * g};
            double const slope = A[0] * sl[0] + A[1] * sl[1] + A[2] * sl[2];
            if (!std::isfinite(slope))
                A = {0., 0., 0.};
            return A;
        }

        // limited slope s of the rule (child = u0 +- s/4) from 5 values along d; the 3-point
        // rules read u[1..3] only
        static double vanLeerSlope_(double const um, double const u0, double const up)
        {
            double const l = u0 - um, r = up - u0;
            return l * r <= 0. ? 0. : 2. * l * r / (l + r);
        }
        static double slopeVals_(std::array<double, 5> const& u, std::size_t const d)
        {
            if (ssi_hack::ao5z() and not ssi_hack::forceCentral)
            {
                if (std::isnan(u[0]) or std::isnan(u[4])) // +-2 not provisioned
                {
                    ++ssi_hack::ao5zFallback;
                    return vanLeerSlope_(u[1], u[2], u[3]);
                }
                auto const A = ao5zA_(u);
                return A[0] * (3. * u[2] / 2. - 2. * u[1] + u[0] / 2.) + A[1] * (u[3] / 2. - u[1] / 2.)
                       + A[2] * (-3. * u[2] / 2. + 2. * u[3] - u[4] / 2.);
            }
            auto const [a, b] = ruleAB_(u[1], u[2], u[3], d);
            return a * (u[2] - u[1]) + b * (u[3] - u[2]);
        }
        template<typename Src>
        static double slopeAt_(Src const& src, Point_t const& at, std::size_t const d)
        {
            bool const five = ssi_hack::ao5z() and not ssi_hack::forceCentral;
            Point_t e{};
            e[d] = 1;
            Point_t e2{};
            e2[d] = 2;
            std::array<double, 5> const u{five ? src(at - e2) : 0., src(at - e), src(at),
                                          src(at + e), five ? src(at + e2) : 0.};
            return slopeVals_(u, d);
        }

        template<std::size_t d, typename Src>
        static double alongAxis_(Src const& src, Point_t const& anchorLocal, int const k)
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
        // slope weights (a, b) of the rule for the coarse triple (um, u0, up) along direction d.
        // ssiPath: the kernel runs with shared indicators (weno3z / cweno then fall back to
        // Jiang-Shu); shared: use (sL, sR) as the indicators.
        static std::pair<double, double> ruleAB_(double const um, double const u0, double const up,
                                                 std::size_t const d, bool const ssiPath = false,
                                                 bool const shared = false, double const sL = 0.,
                                                 double const sR = 0.)
        {
            double const left  = u0 - um;
            double const right = up - u0;

            double a, b;
            if (ssi_hack::forceCentral or ssi_hack::slope() == 6)
                a = b = 0.5;
            else if (ssi_hack::slope() == 1) // minmod
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
                if (shared)
                {
                    isL = sL;
                    isR = sR;
                }
                if (ssi_hack::slope() == 5 and not ssiPath)
                {
                    // CWENO3 (Semplice, Coco & Russo 2016, Sec 2.1 eqs 3-4) averaged over the two
                    // children: P_opt = parabola through the 3 averages, P_L/P_R one-sided
                    // linears, P_0 = (P_opt - d_L P_L - d_R P_R)/d_0. The x^2 - 1/12 term averages
                    // to 0 over a child, so child = u0 +- s_eff/4 with
                    // s_eff = w_0 c + w_L left + w_R right, c = central slope.
                    // IS_L = left^2, IS_R = right^2, IS_0 = c^2 + 13/3 d2^2 (d2 = right - left);
                    // Jiang-Shu alpha_k = d_k/(IS_k+eps)^p, or CWENOZ3 (Cravero et al. 2019,
                    // eq 6b, 19): alpha_k = d_k (1 + (tau/(IS_k+eps))^p), tau = |IS_L + IS_R -
                    // 2 IS_opt| = 5/3 d2^2.
                    double const eps = ssi_hack::wenoEps(um, u0, up, d);
                    double const p   = ssi_hack::cwenoP();
                    double const d0  = ssi_hack::cwenoD0();
                    double const dS  = 0.5 * (1. - d0);
                    double const c   = 0.5 * (left + right);
                    double const d2  = right - left;
                    double const isC = ssi_hack::cwenoSlopeIS0() ? c * c : c * c + 13. / 3. * d2 * d2;
                    double aL, aR, aC;
                    if (ssi_hack::cwenoZ())
                    {
                        double const tau = 5. / 3. * d2 * d2;
                        aL = dS * (1. + std::pow(tau / (isL + eps), p));
                        aR = dS * (1. + std::pow(tau / (isR + eps), p));
                        aC = d0 * (1. + std::pow(tau / (isC + eps), p));
                    }
                    else
                    {
                        aL = dS / std::pow(isL + eps, p);
                        aR = dS / std::pow(isR + eps, p);
                        aC = d0 / std::pow(isC + eps, p);
                    }
                    // an infinite alpha (IS + eps underflow) takes all the weight, shared
                    // equally among the infinite ones
                    int const nInf = std::isinf(aL) + std::isinf(aR) + std::isinf(aC);
                    if (nInf > 0)
                    {
                        aL = std::isinf(aL) ? 1. : 0.;
                        aR = std::isinf(aR) ? 1. : 0.;
                        aC = std::isinf(aC) ? 1. : 0.;
                    }
                    a = (0.5 * aC + aL) / (aL + aR + aC);
                }
                else if (ssi_hack::slope() == 3 and not ssiPath)
                {
                    // WENO3-Z (Don & Borges 2013, JCP 250:347, eq. for r=2): beta_k = slope_k^2,
                    // tau = |beta_L - beta_R|, alpha_k = d_k (1 + (tau/(beta_k + eps))^p),
                    // d_k = 1/2 (the central row), p = 2.
                    double const eps = ssi_hack::wenoEps(um, u0, up, d);
                    double const tau = std::abs(isL - isR);
                    double const rL  = tau / (isL + eps);
                    double const rR  = tau / (isR + eps);
                    double const aL  = 0.5 * (1. + rL * rL);
                    double const aR  = 0.5 * (1. + rR * rR);
                    a                = std::isinf(aL) and std::isinf(aR) ? 0.5
                                       : std::isinf(aL)                ? 1.
                                       : std::isinf(aR)                ? 0.
                                                                       : aL / (aL + aR);
                }
                else
                    a = wenoLeftWeight_<int>(isL, isR);
                b = 1. - a;
            }
            ssi_hack::countRow(a, b, left, right);
            return {a, b};
        }

        template<std::size_t d, typename Src, typename AnchorT>
        static Row_ wenoDualRow_(double const sigma, Src const& src, Point_t const& anchorLocal,
                                 AnchorT const* anchorAMR)
        {
            if (ssi_hack::ao5z() and not ssi_hack::forceCentral)
            {
                std::array<double, 5> const u5{
                    alongAxis_<d>(src, anchorLocal, -2), alongAxis_<d>(src, anchorLocal, -1),
                    alongAxis_<d>(src, anchorLocal, 0), alongAxis_<d>(src, anchorLocal, 1),
                    alongAxis_<d>(src, anchorLocal, 2)};
                if (std::isnan(u5[0]) or std::isnan(u5[4])) // +-2 not provisioned: van Leer
                {
                    ++ssi_hack::ao5zFallback;
                    double const l = u5[2] - u5[1], r = u5[3] - u5[2];
                    double const a = l * r <= 0. ? 0. : r / (l + r);
                    double const b = l * r <= 0. ? 0. : l / (l + r);
                    return Row_{{-1, 0, 1},
                                {-sigma * a / 4., 1. + sigma * (a - b) / 4., sigma * b / 4.},
                                3};
                }
                auto const A = ao5zA_(u5);
                double const q = sigma / 4.;
                return Row_{{-2, -1, 0, 1, 2},
                            {q * 0.5 * A[0], q * (-2. * A[0] - 0.5 * A[1]),
                             1. + q * 1.5 * (A[0] - A[2]), q * (0.5 * A[1] + 2. * A[2]),
                             -q * 0.5 * A[2]},
                            5};
            }
            double const um = alongAxis_<d>(src, anchorLocal, -1);
            double const u0 = alongAxis_<d>(src, anchorLocal, 0);
            double const up = alongAxis_<d>(src, anchorLocal, 1);

            double sL = 0., sR = 0.;
            bool shared = false;
            if (anchorAMR and ssi_hack::slope() != 1 and ssi_hack::slope() != 2)
            {
                auto const& slots = ssi_hack::slotsAt(ssi_hack::keyOf(*anchorAMR, d));
                bool complete     = true;
                for (auto const& v : slots)
                {
                    complete = complete and not std::isnan(v[0]);
                    sL += v[0];
                    sR += v[1];
                }
                if (complete)
                {
                    shared = true;
                    sL /= 5.;
                    sR /= 5.;
                    ++ssi_hack::hits;
                }
                else
                    ++ssi_hack::misses;
            }
            auto const [a, b] = ruleAB_(um, u0, up, d, anchorAMR != nullptr, shared, sL, sR);

            return Row_{
                {-1, 0, 1}, {-sigma * a / 4., 1. + sigma * (a - b) / 4., sigma * b / 4.}, 3};
        }

        // Only the dual rows are limited. The primal copy needs nothing, and the order-2
        // half-point is a convex two-point mean with a single admissible stencil, so there is
        // nothing to blend.
        template<std::size_t d, typename Src, typename AnchorT>
        static Row_ limitedRow_(std::size_t const choice, Src const& src,
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

        template<std::size_t d, typename Src>
        static double tensorGather_(std::array<Row_, dimension> const& rows, Src const& src,
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
        template<typename Src>
        static std::array<double, dimension> minmaxScale_(Src const& src,
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

        // PHARE_REFINE_TENSOR=seq, 2D all-dual: sweep along dA on the coarse lines (weights from
        // each line), then along dB with weights from the first-sweep children; eps of each sweep
        // from the coarse h of its direction.
        template<typename Src>
        static double seqChild_(Src const& src, Point_t const& anchorLocal, std::size_t const dA,
                                std::size_t const dB, double const sigmaA, double const sigmaB)
        {
            int const R = (ssi_hack::ao5z() and not ssi_hack::forceCentral) ? 2 : 1;
            std::array<double, 5> v{};
            for (int i = -R; i <= R; ++i)
            {
                Point_t at = anchorLocal;
                at[dB] += i;
                v[static_cast<std::size_t>(i + 2)] = src(at) + sigmaA * slopeAt_(src, at, dA) / 4.;
            }
            return v[2] + sigmaB * slopeVals_(v, dB) / 4.;
        }

        template<typename Src, typename AnchorT>
        static double limitedGather_(Src const& src, Point_t const& anchorLocal,
                                     std::size_t const combined, AnchorT const* anchorAMR)
        {
            if (ssi_hack::tensorMode() != 0 and dimension == 2 and (combined & 2u)
                and ((combined >> 2u) & 2u) and (ssi_hack::slope() != 4 or ssi_hack::forceCentral))
            {
                double const sx = (combined & 1u) ? 1. : -1.;
                double const sy = ((combined >> 2u) & 1u) ? 1. : -1.;
                if (ssi_hack::tensorMode() == 1)
                    return 0.5
                           * (seqChild_(src, anchorLocal, 1, 0, sy, sx)
                              + seqChild_(src, anchorLocal, 0, 1, sx, sy));
                double value = src(anchorLocal);
                for (std::size_t d = 0; d < 2; ++d)
                    value += (d == 0 ? sx : sy) * slopeAt_(src, anchorLocal, d) / 4.;
                return value;
            }
            if (ssi_hack::slope() == 4 and not ssi_hack::forceCentral)
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

    // ao5z reads +-2 coarse cells along each dual direction
    int coarseStencilWidth() const override
    {
        // (declaring 2 for ao5z leaves the coarse temporary's ghost ring unfilled: NaN. ao5z
        // therefore keeps width 1 and falls back to van Leer where a +-2 value is missing.)
        static bool const wide = std::getenv("PHARE_AO5Z_STENCIL2") != nullptr;
        return (wide and std::is_same_v<Limiter, WenoLimiter> and ssi_hack::ao5z())
                   ? 2
                   : Fill::coarseStencilWidth();
    }
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
