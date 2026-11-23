#ifndef PHARE_CORE_NUMERICS_RECONSTRUCTIONS_SSI_EXPERIMENT_HPP
#define PHARE_CORE_NUMERICS_RECONSTRUCTIONS_SSI_EXPERIMENT_HPP

// EXPERIMENT ONLY (delete when the SSI design is frozen): runtime switches for the test matrix.
//   PHARE_SSI_VARS = rhoP | rhoPB | rhoPV | all     reference variables
//   PHARE_SSI_NORM = own | energy                   per-variable normalisation scale
//   PHARE_SSI_P    = 1 | 2                          Z-weight exponent on the shared path

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace PHARE::core::ssi_experiment
{
struct Options
{
    bool B, V, energy;
    int p;
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
        if (vars != "rhoP" && vars != "rhoPB" && vars != "rhoPV" && vars != "all")
            throw std::runtime_error("PHARE_SSI_VARS: unknown value " + vars);
        if (norm != "own" && norm != "energy")
            throw std::runtime_error("PHARE_SSI_NORM: unknown value " + norm);
        if (p != "1" && p != "2")
            throw std::runtime_error("PHARE_SSI_P: unknown value " + p);
        return Options{vars == "rhoPB" || vars == "all", vars == "rhoPV" || vars == "all",
                       norm == "energy", std::stoi(p)};
    }();
    return opts;
}
} // namespace PHARE::core::ssi_experiment

#endif
