#ifndef PHARE_LIMITED_O4_REFINER_HPP
#define PHARE_LIMITED_O4_REFINER_HPP

// EXP (exp/o4-refine-limit-rb, do not merge): env-knob prototype of limited order-4 refinement of
// the cell-centred hydro rows, 2D.
//   PHARE_REFINE_SLOPE=mp|ao5z|lin    5-point rule (lin = unlimited slope through this path),
//                                     symmetrised sequential sweeps
//   PHARE_REFINE_ENERGY=prim|prim4    energy row: q = Etot - KE prolonged, E rebuilt with the
//                                     children's KE; prim4 adds the unresolved-KE term that makes
//                                     the row fourth order (PHARE_REFINE_ENERGY_TSLOPE=mc|minmod|
//                                     central, default mc)
//   PHARE_REFINE_LIMIT_GHOSTS=0       level creation and regrid only (default: ghost fills too)
// Unset = the base operators, bitwise.

#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/point/point.hpp"

#include "amr/data/field/field_data.hpp"
#include "amr/data/field/field_geometry.hpp"
#include "amr/data/field/field_overlap.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "composite_field_refiner.hpp"
#include "field_refiner_kernel.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>
#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/hier/RefineOperator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace PHARE::amr::refine_exp
{
inline std::string env_(char const* name)
{
    char const* e = std::getenv(name);
    return e ? e : "";
}
inline int rule() // 0 off, 1 mp, 2 ao5z, 3 lin
{
    static int const r = [] {
        auto const v = env_("PHARE_REFINE_SLOPE");
        return v == "mp" ? 1 : v == "ao5z" ? 2 : v == "lin" ? 3 : 0;
    }();
    return r;
}
inline int energy() // 0 off, 1 prim (third order), 2 prim4
{
    static int const r = [] {
        auto const v = env_("PHARE_REFINE_ENERGY");
        return v == "prim" ? 1 : v == "prim4" ? 2 : 0;
    }();
    return r;
}
inline int tSlope() // 0 mc, 1 minmod, 2 central
{
    static int const r = [] {
        auto const v = env_("PHARE_REFINE_ENERGY_TSLOPE");
        return v == "minmod" ? 1 : v == "central" ? 2 : 0;
    }();
    return r;
}
inline bool limitGhosts()
{
    static bool const b = env_("PHARE_REFINE_LIMIT_GHOSTS") != "0";
    return b;
}
inline long eApplied = 0, eFallback = 0, nLimited = 0;

using Block    = std::array<std::array<double, 5>, 5>; // [x offset + 2][y offset + 2]
using Children = std::array<std::array<double, 2>, 2>; // [x parity][y parity]

inline double minmod4(double const a, double const b)
{
    double const c1 = 4 * a - b, c2 = 4 * b - a;
    auto const sgn = [](double const x) { return (x > 0) - (x < 0); };
    int const sg   = sgn(a);
    if (sgn(b) != sg or sgn(c1) != sg or sgn(c2) != sg)
        return 0.;
    return sg * std::min(std::min(std::abs(a), std::abs(b)), std::min(std::abs(c1), std::abs(c2)));
}

// m = (upper child) - u0, the lower child is u0 - m
inline double quarter(double const um2, double const um1, double const u0, double const u1,
                      double const u2)
{
    double const sbig = (22. * (u1 - um1) - 3. * (u2 - um2)) / 32.;
    int const r       = rule();
    if (r == 3)
        return sbig / 4;
    if (r == 1) // MP bounds at h/4, curvature factor 3/8
    {
        constexpr double c = 0.375;
        double const L = u0 - um1, R = u1 - u0;
        double const d0 = R - L, dm = u0 - 2 * um1 + um2, dp = u2 - 2 * u1 + u0;
        double const dP = minmod4(d0, dp), dM = minmod4(dm, d0);
        double const a3 = u0 + R / 4 - c * dP, b2 = u0 + L, b3 = u0 + L / 4 + c * dM;
        double const lo = std::max(std::min({u0, u1, a3}), std::min({u0, b2, b3}));
        double const hi = std::min(std::max({u0, u1, a3}), std::max({u0, b2, b3}));
        double v        = u0 + sbig / 4;
        if ((lo - v) * (hi - v) > 0)
            v += std::abs(lo - v) < std::abs(hi - v) ? lo - v : hi - v;
        return v - u0;
    }
    // WENO-AO(5,3)-Z between the quartic slope and the three parabola slopes
    double const m1 = 41 * u1 / 60 - 11 * u2 / 120 - 41 * um1 / 60 + 11 * um2 / 120;
    double const m2 = -37 * u0 / 28 + 5 * u1 / 7 - 3 * u2 / 56 + 5 * um1 / 7 - 3 * um2 / 56;
    double const m3 = -u1 / 6 + u2 / 12 + um1 / 6 - um2 / 12;
    double const t  = m1 + m3 / 10;
    double const ib = t * t + 13. / 3. * m2 * m2 + 781. / 20. * m3 * m3;
    std::array<double, 3> const sl{3 * u0 / 2 - 2 * um1 + um2 / 2, u1 / 2 - um1 / 2,
                                   -3 * u0 / 2 + 2 * u1 - u2 / 2};
    std::array<double, 3> const cv{u0 / 2 - um1 + um2 / 2, -u0 + u1 / 2 + um1 / 2,
                                   u0 / 2 - u1 + u2 / 2};
    std::array<double, 3> is{}, al{};
    for (int i = 0; i < 3; ++i)
        is[i] = sl[i] * sl[i] + 13. / 3. * cv[i] * cv[i];
    double const eps    = 1e-28 * u0 * u0 + 1e-300;
    constexpr double g0 = 0.85, g = 0.05;
    double const tau = (std::abs(ib - is[0]) + std::abs(ib - is[1]) + std::abs(ib - is[2])) / 3;
    double const r0  = tau / (ib + eps);
    double const a0  = g0 * (1 + r0 * r0);
    double S = a0, lin = 0, nl = 0;
    for (int i = 0; i < 3; ++i)
    {
        double const ri = tau / (is[i] + eps);
        al[i]           = g * (1 + ri * ri);
        S += al[i];
        lin += sl[i];
        nl += al[i] * sl[i];
    }
    double const s = (a0 / S / g0) * (sbig - g * lin) + nl / S;
    return std::isfinite(s) ? s / 4 : 0.;
}

// the four children of the block's central cell: mean of the x-then-y and y-then-x sweeps
inline Children children(Block const& b)
{
    Children c{};
    std::array<double, 5> q{}, v{};
    for (int j = 0; j < 5; ++j)
        q[j] = quarter(b[0][j], b[1][j], b[2][j], b[3][j], b[4][j]);
    for (int px = 0; px < 2; ++px)
    {
        double const sx = 2 * px - 1;
        for (int j = 0; j < 5; ++j)
            v[j] = b[2][j] + sx * q[j];
        double const m = quarter(v[0], v[1], v[2], v[3], v[4]);
        c[px][0]       = 0.5 * (v[2] - m);
        c[px][1]       = 0.5 * (v[2] + m);
    }
    for (int i = 0; i < 5; ++i)
        q[i] = quarter(b[i][0], b[i][1], b[i][2], b[i][3], b[i][4]);
    for (int py = 0; py < 2; ++py)
    {
        double const sy = 2 * py - 1;
        for (int i = 0; i < 5; ++i)
            v[i] = b[i][2] + sy * q[i];
        double const m = quarter(v[0], v[1], v[2], v[3], v[4]);
        c[0][py] += 0.5 * (v[2] - m);
        c[1][py] += 0.5 * (v[2] + m);
    }
    return c;
}

// slope of T used by the fourth-order energy row
inline double tSlopeOf(double const tm, double const t0, double const tp)
{
    double const dL = t0 - tm, dR = tp - t0, c = 0.5 * (dL + dR);
    int const mode = tSlope();
    if (mode == 2)
        return c;
    if (dL * dR <= 0)
        return 0.;
    double const lim = std::min(std::abs(dL), std::abs(dR));
    double const sg  = dL > 0 ? 1. : -1.;
    return mode == 1 ? sg * lim : sg * std::min(std::abs(c), 2 * lim);
}

// loop over the coarse anchors of a fine region; fn(anchor (AMR), anchorLocal)
template<typename FieldT, typename Fn>
void forAnchors(SAMRAI::hier::Box const& region, SAMRAI::hier::Box const& sourceFieldBox, Fn&& fn)
{
    using Point_t      = core::Point<int, 2>;
    auto const fineBox = phare_box_from<2>(region);
    auto const lo      = toCoarseIndex<2>(fineBox.lower);
    auto const hi      = toCoarseIndex<2>(fineBox.upper);
    for (int i = lo[0]; i <= hi[0]; ++i)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Point_t const anchor{i, j};
            fn(anchor, AMRToLocal(anchor, sourceFieldBox));
        }
}

template<typename FieldT>
Block blockOf(FieldT const& src, core::Point<int, 2> const& anchorLocal)
{
    Block u;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            u[i][j] = src(core::Point<int, 2>{anchorLocal[0] + i - 2, anchorLocal[1] + j - 2});
    return u;
}

// write the children that lie in the region, where the destination is still unset
template<typename FieldT>
void writeChildren(FieldT& dst, Children const& c, core::Point<int, 2> const& anchor,
                   SAMRAI::hier::Box const& region, SAMRAI::hier::Box const& destFieldBox)
{
    auto const fineBox = phare_box_from<2>(region);
    for (int px = 0; px < 2; ++px)
        for (int py = 0; py < 2; ++py)
        {
            core::Point<int, 2> const fine{2 * anchor[0] + px, 2 * anchor[1] + py};
            if (fine[0] < fineBox.lower[0] or fine[0] > fineBox.upper[0]
                or fine[1] < fineBox.lower[1] or fine[1] > fineBox.upper[1])
                continue;
            if (auto& v = dst(AMRToLocal(fine, destFieldBox)); std::isnan(v))
                v = c[px][py];
        }
}
} // namespace PHARE::amr::refine_exp


namespace PHARE::amr
{

/** EXP: limited order-4 kernel of one cell-centred row (density, a momentum component). */
template<typename GridLayoutT, typename FieldT>
class LimitedO4FieldRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    using Base                             = detail::CompositeFill<GridLayoutT, FieldT, 4>;

public:
    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        bool cellCentred = dimension == 2;
        for (auto const c : centering)
            cellCentred = cellCentred and c == core::QtyCentering::dual;
        if (!cellCentred or intersectionBox.empty())
        {
            Base::fill(sourceField, destinationField, intersectionBox, centering, destFieldBox,
                       sourceFieldBox, ratio);
            return;
        }
        if constexpr (dimension == 2)
            refine_exp::forAnchors<FieldT>(
                intersectionBox, sourceFieldBox, [&](auto const& anchor, auto const& anchorLocal) {
                    auto const c
                        = refine_exp::children(refine_exp::blockOf(sourceField, anchorLocal));
                    refine_exp::writeChildren(destinationField, c, anchor, intersectionBox,
                                              destFieldBox);
                    ++refine_exp::nLimited;
                });
    }

    int coarseStencilWidth() const override { return 2; }
};


/**
 * EXP: total-energy refine operator of the energy row. Reads the density and the momentum of the
 * same schedule on the coarse patch SAMRAI hands it (items are registered with scratch ==
 * destination, so they sit there under the destination ids).
 */
template<typename GridLayoutT, typename FieldT, typename VectorFieldDataT>
class EnergyRowO4RefineOperator : public SAMRAI::hier::RefineOperator
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    using FieldDataT                       = FieldData<GridLayoutT, FieldT>;

    EnergyRowO4RefineOperator(int const rhoId, int const rhoVId)
        : SAMRAI::hier::RefineOperator{"EnergyRowO4RefineOperator"}
        , rhoId_{rhoId}
        , rhoVId_{rhoVId}
    {
    }

    int getOperatorPriority() const override { return 0; }

    SAMRAI::hier::IntVector getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector(dim, 2);
    }

    void refine(SAMRAI::hier::Patch& destination, SAMRAI::hier::Patch const& source,
                int const destinationId, int const sourceId,
                SAMRAI::hier::BoxOverlap const& destinationOverlap,
                SAMRAI::hier::IntVector const& /*ratio*/) const override
    {
        if constexpr (dimension == 2)
        {
            for (auto const id : {rhoId_, rhoVId_})
                if (!source.checkAllocated(id))
                    throw std::runtime_error("EnergyRowO4RefineOperator: density or momentum is "
                                             "not allocated on the coarse patch");

            auto const& overlapBoxes = dynamic_cast<FieldOverlap const&>(destinationOverlap)
                                           .getDestinationBoxContainer();

            auto& fineEtot         = FieldDataT::getField(destination, destinationId);
            auto const& destLayout = FieldDataT::getLayout(destination, destinationId);
            auto const& coarseEtot = FieldDataT::getField(source, sourceId);
            auto const& srcLayout  = FieldDataT::getLayout(source, sourceId);
            auto const& coarseRho  = FieldDataT::getField(source, rhoId_);
            auto const& coarseRhoV = VectorFieldDataT::getFields(source, rhoVId_);

            auto const& qty         = fineEtot.physicalQuantity();
            auto const destFieldBox = FieldDataT::Geometry::toFieldBox(
                destination.getPatchData(destinationId)->getGhostBox(), qty, destLayout);
            auto const sourceFieldBox = FieldDataT::Geometry::toFieldBox(
                source.getPatchData(sourceId)->getGhostBox(), qty, srcLayout);
            if (!source.getPatchData(rhoId_)->getGhostBox().isSpatiallyEqual(
                    source.getPatchData(sourceId)->getGhostBox())
                or !source.getPatchData(rhoVId_)->getGhostBox().isSpatiallyEqual(
                    source.getPatchData(sourceId)->getGhostBox()))
                throw std::runtime_error(
                    "EnergyRowO4RefineOperator: density, momentum and Etot ghost boxes differ");

            for (auto const& box : overlapBoxes)
            {
                auto const region = destFieldBox * box;
                if (region.empty())
                    continue;
                refine_exp::forAnchors<FieldT>(
                    region, sourceFieldBox, [&](auto const& anchor, auto const& anchorLocal) {
                        refine_exp::writeChildren(
                            fineEtot, childrenOf_(coarseRho, coarseRhoV, coarseEtot, anchorLocal),
                            anchor, region, destFieldBox);
                    });
            }
        }
        else
            throw std::runtime_error("EnergyRowO4RefineOperator is 2D only");
    }

private:
    template<typename Fields>
    static refine_exp::Children childrenOf_(FieldT const& rhoF, Fields const& rhoVF,
                                            FieldT const& etotF,
                                            core::Point<int, 2> const& anchorLocal)
    {
        using refine_exp::Block;
        using refine_exp::Children;
        Block const e   = refine_exp::blockOf(etotF, anchorLocal);
        Block const rho = refine_exp::blockOf(rhoF, anchorLocal);
        std::array<Block, 3> const m{refine_exp::blockOf(rhoVF[0], anchorLocal),
                                     refine_exp::blockOf(rhoVF[1], anchorLocal),
                                     refine_exp::blockOf(rhoVF[2], anchorLocal)};
        bool ok = true;
        Block q;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
            {
                ok      = ok and rho[i][j] > 0;
                q[i][j] = e[i][j]
                          - 0.5
                                * (m[0][i][j] * m[0][i][j] + m[1][i][j] * m[1][i][j]
                                   + m[2][i][j] * m[2][i][j])
                                / rho[i][j];
            }
        Children const cr = refine_exp::children(rho);
        for (auto const& row : cr)
            for (auto const v : row)
                ok = ok and v > 0;
        if (!ok) // no kinetic energy to split: the row by itself
        {
            ++refine_exp::eFallback;
            return refine_exp::children(e);
        }
        std::array<Children, 3> const cm{refine_exp::children(m[0]), refine_exp::children(m[1]),
                                         refine_exp::children(m[2])};
        Children const cq = refine_exp::children(q);
        Children ke{}, out{};
        double keMean = 0;
        for (int px = 0; px < 2; ++px)
            for (int py = 0; py < 2; ++py)
            {
                double m2 = 0;
                for (int k = 0; k < 3; ++k)
                    m2 += cm[k][px][py] * cm[k][px][py];
                ke[px][py] = 0.5 * m2 / cr[px][py];
                keMean += 0.25 * ke[px][py];
            }
        // fourth-order row: T = mean(KE) - KE(means) ~ rho/24 sum_d |central difference of v|^2 is
        // four times smaller on the children, so they carry -(3/4) of the coarse T's variation
        double sTx = 0, sTy = 0;
        if (refine_exp::energy() == 2)
        {
            auto const T = [&](int const i, int const j) {
                double s = 0;
                for (int k = 0; k < 3; ++k)
                {
                    double const dx
                        = 0.5 * (m[k][i + 1][j] / rho[i + 1][j] - m[k][i - 1][j] / rho[i - 1][j]);
                    double const dy
                        = 0.5 * (m[k][i][j + 1] / rho[i][j + 1] - m[k][i][j - 1] / rho[i][j - 1]);
                    s += dx * dx + dy * dy;
                }
                return rho[i][j] * s / 24.;
            };
            double const t0 = T(2, 2);
            sTx             = refine_exp::tSlopeOf(T(1, 2), t0, T(3, 2));
            sTy             = refine_exp::tSlopeOf(T(2, 1), t0, T(2, 3));
        }
        for (int px = 0; px < 2; ++px)
            for (int py = 0; py < 2; ++py)
                out[px][py] = e[2][2] + (cq[px][py] - q[2][2]) + ke[px][py] - keMean
                              - (3. / 16.) * ((2 * px - 1) * sTx + (2 * py - 1) * sTy);
        ++refine_exp::eApplied;
        return out;
    }

    int rhoId_;
    int rhoVId_;
};

} // namespace PHARE::amr

#endif
