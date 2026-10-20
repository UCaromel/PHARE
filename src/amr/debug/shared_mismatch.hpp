#ifndef PHARE_AMR_DEBUG_SHARED_MISMATCH_HPP
#define PHARE_AMR_DEBUG_SHARED_MISMATCH_HPP

// INVESTIGATION-ONLY instrumentation: detects values that differ between patches on
// AMR indexes that several patches hold (shared border nodes, overlapping interiors, ghosts).
// Enabled by env PHARE_MM_LOG=<path>. Compares the local patches of each rank; with
// PHARE_MM_MPI=1 the field probes gather all ranks to rank 0 (collective call sites).

#include "amr/resources_manager/amr_utils.hpp"
#include <mpi.h>

#include <SAMRAI/hier/PatchLevel.h>
#include <SAMRAI/geom/CartesianGridGeometry.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace PHARE::amr::debug
{
inline std::ofstream* mm_log()
{
    static std::ofstream* out = []() -> std::ofstream* {
        if (auto const* p = std::getenv("PHARE_MM_LOG"))
        {
            int rank = 0;
            int init = 0;
            MPI_Initialized(&init);
            if (init)
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return new std::ofstream{std::string{p} + "." + std::to_string(rank)};
        }
        return nullptr;
    }();
    return out;
}

inline bool mm_delta_enabled()
{
    return mm_log() != nullptr;
}

// full per-call cross-patch reports; PHARE_MM_DELTA_ONLY=1 keeps only the magMax deltas
inline bool mm_enabled()
{
    static bool const deltaOnly = [] {
        auto const* p = std::getenv("PHARE_MM_DELTA_ONLY");
        return p and std::string{p} == "1";
    }();
    return mm_log() != nullptr and !deltaOnly;
}

// optional env PHARE_MM_DISABLE_MAGMAX=1
inline bool mm_disable_magmax()
{
    static bool const v = [] {
        auto const* p = std::getenv("PHARE_MM_DISABLE_MAGMAX");
        return p and std::string{p} == "1";
    }();
    return v;
}

// PHARE_MM_ONLY=<label>: only report that probe label
inline bool mm_label_on(std::string const& label)
{
    static std::string const only = [] {
        auto const* p = std::getenv("PHARE_MM_ONLY");
        return p ? std::string{p} : std::string{};
    }();
    return only.empty() or only == label;
}

// PHARE_MM_LEVEL=<n>: only report that level
inline bool mm_level_on(int lvl)
{
    static int const only = [] {
        auto const* p = std::getenv("PHARE_MM_LEVEL");
        return p ? std::stoi(p) : -1;
    }();
    return only < 0 or only == lvl;
}

inline bool mm_flag(char const* name)
{
    auto const* p = std::getenv(name);
    return p and std::string{p} == "1";
}

inline std::size_t& mm_counter()
{
    static std::size_t c = 0;
    return c;
}

struct MMEntry
{
    int patch; // rank * 1000000 + local patch index
    int rank;
    bool interior;
    double value;
};

// one ghost-box value of one patch, as exchanged by the cross-rank probe
struct MMRec
{
    std::int64_t key;
    std::array<int, 3> pt;
    int patch;
    int rank;
    int interior;
    double value;
};

struct MMStats
{
    double maxShared    = 0; // interior vs interior
    std::size_t nShared = 0, nSharedPts = 0;
    double maxGhostOwner     = 0; // ghost vs some interior
    std::size_t nGhostOwner  = 0;
    double maxGhostGhost     = 0; // ghost vs ghost, no interior holder
    std::size_t nGhostGhost  = 0;
    std::size_t nNaNGhost    = 0; // ghost NaN while an interior holds a number
    std::size_t nNaNInterior = 0; // NaN in interior
    std::array<int, 3> worstShared{};
    double worstSharedA = 0, worstSharedB = 0;
    std::array<int, 3> worstGG{};
    double worstGGa = 0, worstGGb = 0;
    int worstGGpa = -1, worstGGpb = -1;
    std::array<int, 3> worstGO{};
    double worstGOghost = 0, worstGOowner = 0;

    // cross-rank subset: pairs whose two holders live on different ranks; a ghost-owner
    // mismatch is cross-rank when no interior holder of that index is on the ghost's rank
    bool mpi = false;
    double xShared = 0, xGhostOwner = 0, xGhostGhost = 0;
    std::size_t nxShared = 0, nxGhostOwner = 0, nxGhostGhost = 0;
    std::array<int, 3> xWorst{};
    int xWorstRa = -1, xWorstRb = -1;
    double xWorstD = 0;
    char const* xWorstKind = "";
};

// PHARE_MM_MPI=1: mm_field/mm_vec gather every rank's records to rank 0 (collective)
inline bool mm_mpi()
{
    static bool const v = [] {
        auto const* p = std::getenv("PHARE_MM_MPI");
        return p and std::string{p} == "1";
    }();
    return v;
}

inline int mm_rank()
{
    int rank = 0, init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

// all records to rank 0; other ranks get an empty vector
inline std::vector<MMRec> mm_gather(std::vector<MMRec> const& local)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int const rank = mm_rank();
    int const nb   = static_cast<int>(local.size() * sizeof(MMRec));
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&nb, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<MMRec> all;
    if (rank == 0)
    {
        std::size_t tot = 0;
        for (int r = 0; r < size; ++r)
        {
            displs[r] = static_cast<int>(tot);
            tot += counts[r];
        }
        all.resize(tot / sizeof(MMRec));
    }
    MPI_Gatherv(local.data(), nb, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
    return all;
}

template<std::size_t dim>
MMStats mm_stats(std::vector<MMRec> const& recs)
{
    std::unordered_map<std::int64_t, std::vector<MMEntry>> map;
    std::unordered_map<std::int64_t, std::array<int, 3>> pts;
    for (auto const& r : recs)
    {
        map[r.key].push_back({r.patch, r.rank, r.interior != 0, r.value});
        pts[r.key] = r.pt;
    }

    MMStats s;
    auto cross = [&](double d, std::int64_t k, int ra, int rb, char const* kind) {
        if (d > s.xWorstD)
        {
            s.xWorstD = d;
            s.xWorst  = pts[k];
            s.xWorstRa = ra;
            s.xWorstRb = rb;
            s.xWorstKind = kind;
        }
    };
    for (auto const& [k, entries] : map)
    {
        MMEntry const* firstInt = nullptr;
        for (auto const& e : entries)
            if (e.interior)
            {
                if (std::isnan(e.value))
                    ++s.nNaNInterior;
                else if (!firstInt)
                    firstInt = &e;
            }

        // interior vs interior
        std::size_t nInt = 0;
        for (auto const& e : entries)
            if (e.interior and !std::isnan(e.value))
            {
                ++nInt;
                double d = std::abs(e.value - firstInt->value);
                if (d > 0)
                {
                    ++s.nShared;
                    if (d > s.maxShared)
                    {
                        s.maxShared = d;
                        for (std::size_t i = 0; i < dim; ++i)
                            s.worstShared[i] = pts[k][i];
                        s.worstSharedA = firstInt->value;
                        s.worstSharedB = e.value;
                    }
                    if (e.rank != firstInt->rank)
                    {
                        ++s.nxShared;
                        s.xShared = std::max(s.xShared, d);
                        cross(d, k, firstInt->rank, e.rank, "shared");
                    }
                }
            }
        if (nInt > 1)
            ++s.nSharedPts;

        if (firstInt)
        {
            for (auto const& e : entries)
                if (!e.interior)
                {
                    if (std::isnan(e.value))
                    {
                        ++s.nNaNGhost;
                        continue;
                    }
                    // compare against all interiors, take the min distance (multiple owners may
                    // disagree; shared mismatch already counted)
                    double dmin    = std::numeric_limits<double>::max();
                    double own     = 0;
                    int ownRank    = -1;
                    bool sameOwner = false;
                    for (auto const& o : entries)
                        if (o.interior and !std::isnan(o.value))
                        {
                            sameOwner = sameOwner or o.rank == e.rank;
                            double d  = std::abs(e.value - o.value);
                            if (d < dmin)
                            {
                                dmin    = d;
                                own     = o.value;
                                ownRank = o.rank;
                            }
                        }
                    if (dmin > 0)
                    {
                        ++s.nGhostOwner;
                        if (dmin > s.maxGhostOwner)
                        {
                            s.maxGhostOwner = dmin;
                            for (std::size_t i = 0; i < dim; ++i)
                                s.worstGO[i] = pts[k][i];
                            s.worstGOghost = e.value;
                            s.worstGOowner = own;
                        }
                        if (!sameOwner)
                        {
                            ++s.nxGhostOwner;
                            s.xGhostOwner = std::max(s.xGhostOwner, dmin);
                            cross(dmin, k, e.rank, ownRank, "ghostOwner");
                        }
                    }
                }
        }
        else
        {
            MMEntry const* firstG = nullptr;
            for (auto const& e : entries)
                if (!std::isnan(e.value))
                {
                    if (!firstG)
                        firstG = &e;
                    else
                    {
                        double d = std::abs(e.value - firstG->value);
                        if (d > 0)
                        {
                            ++s.nGhostGhost;
                            if (d > s.maxGhostGhost)
                            {
                                s.maxGhostGhost = d;
                                for (std::size_t i = 0; i < dim; ++i)
                                    s.worstGG[i] = pts[k][i];
                                s.worstGGa  = firstG->value;
                                s.worstGGb  = e.value;
                                s.worstGGpa = firstG->patch;
                                s.worstGGpb = e.patch;
                            }
                            if (e.rank != firstG->rank)
                            {
                                ++s.nxGhostGhost;
                                s.xGhostGhost = std::max(s.xGhostGhost, d);
                                cross(d, k, firstG->rank, e.rank, "ghostGhost");
                            }
                        }
                    }
                }
        }
    }
    return s;
}

template<typename GridLayoutT, typename ResourcesManager, typename Holder, typename Getter>
MMStats mm_check_field(ResourcesManager& rm, Holder& holder, Getter&& getter,
                       SAMRAI::hier::PatchLevel const& level)
{
    constexpr auto dim = GridLayoutT::dimension;

    auto const ratio = level.getRatioToLevelZero();
    auto const& geom = *level.getGridGeometry();
    auto const shift = geom.getPeriodicShift(ratio);
    auto const dombb = level.getPhysicalDomain(SAMRAI::hier::BlockId::zero()).getBoundingBox();

    std::array<int, dim> period{}, lo{};
    for (std::size_t i = 0; i < dim; ++i)
    {
        period[i] = shift(i);
        lo[i]     = dombb.lower(i);
    }

    auto canon = [&](core::Point<int, dim> p) {
        for (std::size_t i = 0; i < dim; ++i)
            if (period[i] > 0)
            {
                int x = (p[i] - lo[i]) % period[i];
                if (x < 0)
                    x += period[i];
                p[i] = x + lo[i];
            }
        return p;
    };

    auto key = [&](core::Point<int, dim> const& p) {
        std::int64_t k = 0;
        for (std::size_t i = 0; i < dim; ++i)
            k = k * 1000003 + (p[i] + 100000);
        return k;
    };

    int const rank = mm_rank();
    std::vector<MMRec> recs;

    int ip = 0;
    static bool const printBoxes = [] {
        auto const* p = std::getenv("PHARE_MM_BOXES");
        return p and std::string{p} == "1";
    }();
    for (auto& patch : rm.enumerate(level, holder))
    {
        auto& field       = getter();
        auto const layout = layoutFromPatch<GridLayoutT>(*patch);
        if (printBoxes)
            *mm_log() << "   box p" << ip << " L" << level.getLevelNumber() << " "
                      << patch->getBox() << "\n";
        auto const ibox   = layout.AMRBoxFor(field);
        auto const gbox   = layout.AMRGhostBoxFor(field);
        for (auto const& amrIdx : gbox)
        {
            auto const lcl = layout.AMRToLocal(amrIdx);
            auto c         = canon(amrIdx);
            MMRec r{key(c), {}, rank * 1000000 + ip, rank,
                    core::isIn(amrIdx, ibox) ? 1 : 0, field(lcl)};
            for (std::size_t i = 0; i < dim; ++i)
                r.pt[i] = c[i];
            recs.push_back(r);
        }
        ++ip;
    }

    if (mm_mpi())
    {
        auto all = mm_gather(recs);
        auto s   = mm_stats<dim>(all);
        s.mpi    = true;
        return s;
    }
    return mm_stats<dim>(recs);
}

template<typename GridLayoutT>
void mm_print(std::string const& label, std::string const& name, MMStats const& s,
              SAMRAI::hier::PatchLevel const& level, double time)
{
    auto& out      = *mm_log();
    auto const dim = GridLayoutT::dimension;
    out.precision(17);
    out << mm_counter() << " L" << level.getLevelNumber() << " t=" << time << " " << label << " "
        << name << " shared=" << s.maxShared << "(" << s.nShared << "/" << s.nSharedPts << ")";
    if (s.nShared)
    {
        out << "@[";
        for (std::size_t i = 0; i < dim; ++i)
            out << s.worstShared[i] << (i + 1 < dim ? "," : "");
        out << "](" << s.worstSharedA << " vs " << s.worstSharedB << ")";
    }
    out << " ghostOwner=" << s.maxGhostOwner << "(" << s.nGhostOwner << ")";
    if (s.nGhostOwner)
    {
        out << "@[";
        for (std::size_t i = 0; i < dim; ++i)
            out << s.worstGO[i] << (i + 1 < dim ? "," : "");
        out << "](g=" << s.worstGOghost << " own=" << s.worstGOowner << ")";
    }
    out << " ghostGhost=" << s.maxGhostGhost << "(" << s.nGhostGhost << ")";
    if (s.nGhostGhost)
    {
        out << "@[";
        for (std::size_t i = 0; i < dim; ++i)
            out << s.worstGG[i] << (i + 1 < dim ? "," : "");
        out << "](p" << s.worstGGpa << "=" << s.worstGGa << " p" << s.worstGGpb << "="
            << s.worstGGb << ")";
    }
    out << " nanGhost=" << s.nNaNGhost << " nanInt=" << s.nNaNInterior;
    if (s.mpi)
    {
        out << " X:shared=" << s.xShared << "(" << s.nxShared << ") gO=" << s.xGhostOwner << "("
            << s.nxGhostOwner << ") gG=" << s.xGhostGhost << "(" << s.nxGhostGhost << ")";
        if (s.xWorstD > 0)
        {
            out << " worst=" << s.xWorstKind << "@[";
            for (std::size_t i = 0; i < dim; ++i)
                out << s.xWorst[i] << (i + 1 < dim ? "," : "");
            out << "] r" << s.xWorstRa << "/r" << s.xWorstRb;
        }
    }
    out << "\n";
    out.flush();
}

template<typename GridLayoutT, typename ResourcesManager, typename Field>
void mm_field(std::string const& label, ResourcesManager& rm, Field& field,
              SAMRAI::hier::PatchLevel const& level, double time)
{
    if (!mm_enabled() or !mm_label_on(label) or !mm_level_on(level.getLevelNumber()))
        return;
    ++mm_counter();
    auto s = mm_check_field<GridLayoutT>(rm, field, [&]() -> auto& { return field; }, level);
    if (!mm_mpi() or mm_rank() == 0)
        mm_print<GridLayoutT>(label, field.name(), s, level, time);
}

template<typename GridLayoutT, typename ResourcesManager, typename VecField>
void mm_vec(std::string const& label, ResourcesManager& rm, VecField& vf,
            SAMRAI::hier::PatchLevel const& level, double time)
{
    if (!mm_enabled() or !mm_label_on(label) or !mm_level_on(level.getLevelNumber()))
        return;
    ++mm_counter();
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto s = mm_check_field<GridLayoutT>(
            rm, vf, [&]() -> auto& { return vf.getComponent(static_cast<core::Component>(c)); }, level);
        if (!mm_mpi() or mm_rank() == 0)
            mm_print<GridLayoutT>(label, vf.getComponentName(static_cast<core::Component>(c)), s,
                                  level, time);
    }
}


// snapshot of a vecfield on all local patches (ghost box), for before/after comparisons
struct MMSnap
{
    std::vector<std::array<std::vector<double>, 3>> vals;
    std::vector<std::array<std::vector<char>, 3>> interior;
};

template<typename GridLayoutT, typename ResourcesManager, typename VecField>
MMSnap mm_snap(ResourcesManager& rm, VecField& vf, SAMRAI::hier::PatchLevel const& level)
{
    MMSnap snap;
    for (auto& patch : rm.enumerate(level, vf))
    {
        auto const layout = layoutFromPatch<GridLayoutT>(*patch);
        auto& v           = snap.vals.emplace_back();
        auto& in          = snap.interior.emplace_back();
        for (std::size_t c = 0; c < 3; ++c)
        {
            auto& f         = vf.getComponent(static_cast<core::Component>(c));
            auto const ibox = layout.AMRBoxFor(f);
            for (auto const& amrIdx : layout.AMRGhostBoxFor(f))
            {
                v[c].push_back(f(layout.AMRToLocal(amrIdx)));
                in[c].push_back(core::isIn(amrIdx, ibox));
            }
        }
    }
    return snap;
}

template<typename GridLayoutT, typename ResourcesManager, typename Field>
MMSnap mm_snap_field(ResourcesManager& rm, Field& f, SAMRAI::hier::PatchLevel const& level)
{
    MMSnap snap;
    for (auto& patch : rm.enumerate(level, f))
    {
        auto const layout = layoutFromPatch<GridLayoutT>(*patch);
        auto& v           = snap.vals.emplace_back();
        auto& in          = snap.interior.emplace_back();
        auto const ibox   = layout.AMRBoxFor(f);
        for (auto const& amrIdx : layout.AMRGhostBoxFor(f))
        {
            v[0].push_back(f(layout.AMRToLocal(amrIdx)));
            in[0].push_back(core::isIn(amrIdx, ibox));
        }
    }
    return snap;
}

inline void mm_snap_diff(std::string const& label, MMSnap const& a, MMSnap const& b,
                         SAMRAI::hier::PatchLevel const& level, double time)
{
    ++mm_counter();
    std::array<double, 3> dInt{}, dGh{};
    std::array<std::size_t, 3> nInt{}, nGh{}, nNaNfill{};
    for (std::size_t p = 0; p < a.vals.size(); ++p)
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < a.vals[p][c].size(); ++i)
            {
                double x = a.vals[p][c][i], y = b.vals[p][c][i];
                if (std::isnan(x) and !std::isnan(y))
                {
                    ++nNaNfill[c];
                    continue;
                }
                if (std::isnan(x) and std::isnan(y))
                    continue;
                double d = std::abs(x - y);
                if (d > 0)
                {
                    if (a.interior[p][c][i])
                    {
                        ++nInt[c];
                        dInt[c] = std::max(dInt[c], d);
                    }
                    else
                    {
                        ++nGh[c];
                        dGh[c] = std::max(dGh[c], d);
                    }
                }
            }
    auto& out = *mm_log();
    out.precision(6);
    out << mm_counter() << " L" << level.getLevelNumber() << " t=" << time << " " << label
        << " DELTA";
    for (std::size_t c = 0; c < 3; ++c)
        out << " c" << c << ":int=" << dInt[c] << "(" << nInt[c] << ") gh=" << dGh[c] << "("
            << nGh[c] << ") nanfill=" << nNaNfill[c];
    out << "\n";
    out.flush();
}

// PHARE_MM_TIME=1: accumulate wall time of the max schedules, printed at exit per rank
struct MMTimer
{
    double seconds = 0;
    std::size_t calls = 0;
    ~MMTimer()
    {
        if (calls and mm_flag("PHARE_MM_TIME"))
        {
            int rank = 0, fin = 0;
            MPI_Finalized(&fin);
            std::fprintf(stderr, "MMTIME magMax calls=%zu seconds=%.3f\n", calls, seconds);
            (void)rank;
        }
    }
};
inline MMTimer& mm_timer()
{
    static MMTimer t;
    return t;
}

} // namespace PHARE::amr::debug

#endif
