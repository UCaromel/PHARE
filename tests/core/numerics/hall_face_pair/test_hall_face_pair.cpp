// The F12 face-pair Hall field on a periodic patch: fourth order against the exact edge average
// of J x B / rho, and zero discrete work against its own current, sum_e Ebar_H . J~ = 0.
// With HALL_F12_DUMP=<prefix> the inputs and the point-value field are written as text, for a
// bitwise comparison with the numpy reference.

#include "phare_core.hpp"
#include "simulator/simulator_def.hpp"
#include "core/numerics/constrained_transport/hall_face_pair.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

namespace PHARE::core
{
namespace
{
    constexpr double twoPi = 2 * std::numbers::pi;

    struct Mode
    {
        std::array<double, 3> k, a;
        double phase;
    };

    std::array<double, 3> cross(std::array<double, 3> const& u, std::array<double, 3> const& v)
    {
        return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    }

    template<std::size_t dim>
    struct Fields
    {
        std::vector<Mode> modes;
        std::array<double, 3> B0{0.4, 0.3, 1.0};
        std::array<double, 3> kr;

        Fields()
        {
            std::vector<std::array<double, 3>> ks;
            if constexpr (dim == 3)
            {
                ks = {{0, 1, 1}, {1, 0, 2}, {2, 1, 0}, {1, -1, 1}, {1, 2, -1}};
                kr = {1, 2, -1};
            }
            else
            {
                ks = {{0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {2, -1, 0}, {1, 2, 0}};
                kr = {1, 2, 0};
            }
            double s = 0.3;
            for (auto const& k : ks)
            {
                Mode m;
                for (int d = 0; d < 3; ++d)
                {
                    m.k[d] = twoPi * k[d];
                    s      = std::fmod(s * 7.31 + 0.17, 1.0);
                    m.a[d] = 0.12 * (2 * s - 1);
                }
                s       = std::fmod(s * 7.31 + 0.17, 1.0);
                m.phase = twoPi * s;
                modes.push_back(m);
            }
            for (auto& v : kr)
                v *= twoPi;
        }

        std::array<double, 3> B(std::array<double, 3> const& x) const
        {
            auto out = B0;
            for (auto const& m : modes)
            {
                auto const c = std::cos(m.k[0] * x[0] + m.k[1] * x[1] + m.k[2] * x[2] + m.phase);
                auto const v = cross(m.k, m.a);
                for (int i = 0; i < 3; ++i)
                    out[i] += v[i] * c;
            }
            return out;
        }

        std::array<double, 3> J(std::array<double, 3> const& x) const
        {
            std::array<double, 3> out{};
            for (auto const& m : modes)
            {
                auto const s = std::sin(m.k[0] * x[0] + m.k[1] * x[1] + m.k[2] * x[2] + m.phase);
                auto const v = cross(m.k, cross(m.k, m.a));
                for (int i = 0; i < 3; ++i)
                    out[i] -= v[i] * s;
            }
            return out;
        }

        double rho(std::array<double, 3> const& x) const
        {
            return 1.0 + 0.3 * std::sin(kr[0] * x[0] + kr[1] * x[1] + kr[2] * x[2] + 0.4);
        }

        std::array<double, 3> EH(std::array<double, 3> const& x) const
        {
            auto const j = J(x);
            auto const b = B(x);
            auto const r = rho(x);
            auto const e = cross(j, b);
            return {e[0] / r, e[1] / r, e[2] / r};
        }
    };

    // average of fn over the cell-sized interval along `axes` around x (6-point Gauss)
    template<typename Fn>
    double average(Fn&& fn, std::array<double, 3> const& x, std::array<bool, 3> const& axes,
                   double const h)
    {
        static constexpr std::array<double, 6> q{-0.9324695142031521, -0.6612093864662645,
                                                 -0.2386191860831969, 0.2386191860831969,
                                                 0.6612093864662645,  0.9324695142031521};
        static constexpr std::array<double, 6> w{0.1713244923791704, 0.3607615730481386,
                                                 0.4679139345726910, 0.4679139345726910,
                                                 0.3607615730481386, 0.1713244923791704};
        double acc   = 0.;
        int const nx = axes[0] ? 6 : 1, ny = axes[1] ? 6 : 1, nz = axes[2] ? 6 : 1;
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    auto p   = x;
                    double c = 1.;
                    if (axes[0])
                        p[0] += q[i] * h / 2, c *= w[i] / 2;
                    if (axes[1])
                        p[1] += q[j] * h / 2, c *= w[j] / 2;
                    if (axes[2])
                        p[2] += q[k] * h / 2, c *= w[k] / 2;
                    acc += c * fn(p);
                }
        return acc;
    }

    template<std::size_t dim>
    struct Result
    {
        double error;
        double work;
    };

    template<std::size_t dim>
    Result<dim> run(std::uint32_t const N)
    {
        static constexpr PHARE::SimOpts opts{dim,
                                             0,
                                             0,
                                             PHARE::MHDOpts::MHDOrder::O4,
                                             PHARE::MHDOpts::ReconstructionType::WENOZ,
                                             PHARE::MHDOpts::SlopeLimiterType::None,
                                             PHARE::MHDOpts::RiemannSolverType::Rusanov};
        using Layout = PHARE_Types<opts>::MHD::GridLayout_t;
        static_assert(Layout::options.field_ghost_width == 6);

        double const h = 1. / N;
        std::array<double, dim> dl;
        std::array<std::uint32_t, dim> nc;
        dl.fill(h);
        nc.fill(N);
        Point<double, dim> origin;
        for (std::size_t d = 0; d < dim; ++d)
            origin[d] = 0.;
        Layout layout{dl, nc, origin};

        using Vector = MHDQuantity::Vector;
        UsableVecFieldMHD<dim> Bavg{"Bavg", layout, Vector::B}, Bpv{"Bpv", layout, Vector::B},
            MB{"MB", layout, Vector::B}, Jt{"Jt", layout, Vector::J}, Js{"Js", layout, Vector::J},
            E{"E", layout, Vector::E};
        UsableFieldMHD<dim> rho{"rho", layout, MHDQuantity::Scalar::rho},
            rhopv{"rhopv", layout, MHDQuantity::Scalar::rho};

        Fields<dim> const fields;

        auto position = [&](auto const& field, MeshIndex<dim> const& idx) {
            std::array<double, 3> x{};
            auto const centering = layout.centering(field.physicalQuantity());
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto const start
                    = layout.physicalStartIndex(centering[d], static_cast<Direction>(d));
                x[d] = (static_cast<double>(idx[d]) - static_cast<double>(start)
                        + (centering[d] == QtyCentering::dual ? 0.5 : 0.))
                       * h;
            }
            return x;
        };
        auto axesOf = [&](auto const& field) {
            std::array<bool, 3> axes{};
            auto const centering = layout.centering(field.physicalQuantity());
            for (std::size_t d = 0; d < dim; ++d)
                axes[d] = centering[d] == QtyCentering::dual;
            return axes;
        };
        auto edgeAxes = [&](std::size_t c) {
            std::array<bool, 3> axes{};
            if (c < dim)
                axes[c] = true;
            return axes;
        };

        // averages on the whole ghost box: periodic images of the analytic fields
        for (std::size_t b = 0; b < 3; ++b)
        {
            auto& Bb        = Bavg.super()[b];
            auto const axes = axesOf(Bb);
            layout.evalOnGhostBox(Bb, [&](auto const&... args) {
                MeshIndex<dim> idx{args...};
                Bb(idx) = average([&](auto const& p) { return fields.B(p)[b]; }, position(Bb, idx),
                                  axes, h);
            });
        }
        {
            auto const axes = axesOf(rho.super());
            layout.evalOnGhostBox(rho.super(), [&](auto const&... args) {
                MeshIndex<dim> idx{args...};
                rho(idx) = average([&](auto const& p) { return fields.rho(p); },
                                   position(rho.super(), idx), axes, h);
            });
        }

        // point values as PointValueConverter writes them
        Point<std::uint32_t, dim> one;
        for (std::size_t d = 0; d < dim; ++d)
            one[d] = 1;
        layout.evalOnShrinkedGhostBox(rhopv.super(), one, [&](auto const&... args) {
            MeshIndex<dim> idx{args...};
            rhopv(idx) = rho(idx) - layout.lapl(rho.super(), idx) / 24.;
        });
        for_N<3>([&](auto i) {
            constexpr auto b = static_cast<Direction>(i());
            auto const& src  = Bavg.super()[i()];
            auto& dst        = Bpv.super()[i()];
            layout.evalOnShrinkedGhostBox(dst, one, [&](auto const&... args) {
                MeshIndex<dim> idx{args...};
                dst(idx) = src(idx) - layout.template transverseLapl<b>(src, idx) / 24.;
            });
        });

        HallFacePairO4<Layout>{layout}(MB.super(), Jt.super(), Js.super(), Bavg.super(),
                                       Bpv.super(), rhopv.super(), E.super());

        // edge average along each edge, as PointValueConverter::fluxesToAverages
        double errMax = 0., refMax = 0., work = 0., workAbs = 0.;
        for_N<3>([&](auto i) {
            constexpr std::size_t c = i();
            auto const& Ec          = E.super()[c];
            auto const& Jc          = Jt.super()[c];
            auto const centering    = layout.centering(Ec.physicalQuantity());
            layout.evalOnBox(Ec, [&](auto const&... args) {
                MeshIndex<dim> idx{args...};
                double ebar = Ec(idx);
                if constexpr (c < dim)
                    ebar += layout.template directionalLapl<static_cast<Direction>(c)>(Ec, idx)
                            / 24.;
                auto const exact = average([&](auto const& p) { return fields.EH(p)[c]; },
                                           position(Ec, idx), edgeAxes(c), h);
                errMax           = std::max(errMax, std::abs(ebar - exact));
                refMax           = std::max(refMax, std::abs(exact));

                // each periodic edge once: drop the last primal node
                for (std::size_t d = 0; d < dim; ++d)
                    if (centering[d] == QtyCentering::primal
                        && idx[d]
                               == layout.physicalEndIndex(centering[d], static_cast<Direction>(d)))
                        return;
                work += ebar * Jc(idx);
                workAbs += std::abs(ebar * Jc(idx));
            });
        });

        if (auto const* prefix = std::getenv("HALL_F12_DUMP"))
        {
            auto dump = [&](auto const& field, std::string const& name) {
                auto const path = std::string{prefix} + "_" + std::to_string(dim) + "d_"
                                  + std::to_string(N) + "_" + name + ".txt";
                auto* f              = std::fopen(path.c_str(), "w");
                auto const centering = layout.centering(field.physicalQuantity());
                std::array<std::uint32_t, 3> s{0, 0, 0}, n{1, 1, 1};
                for (std::size_t d = 0; d < dim; ++d)
                {
                    s[d] = layout.physicalStartIndex(centering[d], static_cast<Direction>(d));
                    n[d] = N;
                }
                for (std::uint32_t a = 0; a < n[0]; ++a)
                    for (std::uint32_t b = 0; b < n[1]; ++b)
                        for (std::uint32_t c = 0; c < n[2]; ++c)
                        {
                            MeshIndex<dim> idx;
                            std::array<std::uint32_t, 3> v{s[0] + a, s[1] + b, s[2] + c};
                            for (std::size_t d = 0; d < dim; ++d)
                                idx[d] = v[d];
                            std::fprintf(f, "%.17g\n", field(idx));
                        }
                std::fclose(f);
            };
            dump(rho.super(), "rho");
            for (std::size_t k = 0; k < 3; ++k)
            {
                dump(Bavg.super()[k], "B" + std::to_string(k));
                dump(E.super()[k], "Epv" + std::to_string(k));
                dump(Jt.super()[k], "Jt" + std::to_string(k));
            }
        }

        return {errMax / refMax, work / workAbs};
    }

    template<std::size_t dim>
    void checkOrderAndWork(std::vector<std::uint32_t> const& Ns)
    {
        std::vector<double> errs;
        for (auto N : Ns)
        {
            auto const r = run<dim>(N);
            errs.push_back(r.error);
            std::printf("F12 %zuD N=%u  max rel err %.3e  work/|work| %+.2e\n", dim, N, r.error,
                        r.work);
            EXPECT_LT(std::abs(r.work), 1e-13);
        }
        for (std::size_t k = 0; k + 1 < errs.size(); ++k)
        {
            auto const order = std::log2(errs[k] / errs[k + 1]);
            std::printf("F12 %zuD order %u->%u: %.3f\n", dim, Ns[k], Ns[k + 1], order);
            if (k + 2 == errs.size())
                EXPECT_GT(order, 3.8);
        }
    }
} // namespace

TEST(HallFacePairO4, fourthOrderAndZeroWork2D)
{
    checkOrderAndWork<2>({16, 32, 64, 128});
}

TEST(HallFacePairO4, fourthOrderAndZeroWork3D)
{
    checkOrderAndWork<3>({12, 24, 48});
}

} // namespace PHARE::core
