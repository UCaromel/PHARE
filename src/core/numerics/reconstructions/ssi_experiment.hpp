#ifndef PHARE_CORE_NUMERICS_RECONSTRUCTIONS_SSI_EXPERIMENT_HPP
#define PHARE_CORE_NUMERICS_RECONSTRUCTIONS_SSI_EXPERIMENT_HPP

// EXPERIMENT ONLY (delete when the SSI design is frozen): runtime switches for the test matrix.
//   PHARE_SSI_VARS = rhoP | rhoPB | rhoPV | all | rhoB | rhoV | rhoVB   reference variables
//   PHARE_SSI_NORM = own | energy                   per-variable normalisation scale
//   PHARE_SSI_P    = 1 | 2                          Z-weight exponent on the shared path
//   PHARE_SSI_LEGACY = 1                            old-branch combine: mean, no round-off floor
//   PHARE_SSI_W3EPS  = <eps>                        WENO3 shared path: JS weights with this eps

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace PHARE::core::ssi_experiment
{
struct Options
{
    bool P, B, V, energy, legacy;
    int p;
    double w3_js_eps; // <= 0: Z3 weights
};

inline Options const& options()
{
    static Options const opts = [] {
        auto get = [](char const* name, char const* dflt) {
            auto const* env = std::getenv(name);
            return std::string{env ? env : dflt};
        };
        auto const vars = get("PHARE_SSI_VARS", "rhoPB");
        auto const norm = get("PHARE_SSI_NORM", "own");
        auto const p    = get("PHARE_SSI_P", "1");
        auto const known = {"rhoP", "rhoPB", "rhoPV", "all", "rhoB", "rhoV", "rhoVB"};
        bool ok          = false;
        for (auto const* k : known)
            ok |= vars == k;
        if (!ok)
            throw std::runtime_error("PHARE_SSI_VARS: unknown value " + vars);
        if (norm != "own" && norm != "energy")
            throw std::runtime_error("PHARE_SSI_NORM: unknown value " + norm);
        if (p != "1" && p != "2")
            throw std::runtime_error("PHARE_SSI_P: unknown value " + p);
        bool const hasP = vars == "rhoP" || vars == "rhoPB" || vars == "rhoPV" || vars == "all";
        bool const hasB = vars == "rhoPB" || vars == "all" || vars == "rhoB" || vars == "rhoVB";
        bool const hasV = vars == "rhoPV" || vars == "all" || vars == "rhoV" || vars == "rhoVB";
        return Options{hasP, hasB, hasV, norm == "energy", get("PHARE_SSI_LEGACY", "0") == "1",
                       std::stoi(p), std::stod(get("PHARE_SSI_W3EPS", "0"))};
    }();
    return opts;
}
} // namespace PHARE::core::ssi_experiment

#endif
