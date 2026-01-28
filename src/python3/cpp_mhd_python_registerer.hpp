#ifndef PHARE_CPP_MHD_PYTHON_REGISTERER_HPP
#define PHARE_CPP_MHD_PYTHON_REGISTERER_HPP

#include "phare_simulator_options.hpp"
#ifndef PHARE_SIM_STR
#define PHARE_SIM_STR 1, 1, 2 // mostly for clangformat - errors in cpp file if define is missing
#endif

#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "amr/samrai.hpp" // IWYU pragma: keep
#include "amr/wrappers/hierarchy.hpp"

#include "simulator/simulator.hpp" // IWYU pragma: keep

#include "python3/pybind_def.hpp" // IWYU pragma: keep
#include "pybind11/stl.h"         // IWYU pragma: keep
#include "pybind11/numpy.h"       // IWYU pragma: keep
#include "pybind11/chrono.h"      // IWYU pragma: keep
#include "pybind11/complex.h"     // IWYU pragma: keep
#include "pybind11/functional.h"  // IWYU pragma: keep

#include "python3/particles.hpp"     // IWYU pragma: keep
#include "python3/patch_level.hpp"   // IWYU pragma: keep
#include "python3/data_wrangler.hpp" // IWYU pragma: keep

namespace PHARE::pydata
{

namespace py = pybind11;

template<typename Simulator, typename PyClass>
void declareSimulator(PyClass&& sim)
{
    sim.def("initialize", &Simulator::initialize)
        .def("advance", &Simulator::advance)
        .def("startTime", &Simulator::startTime)
        .def("currentTime", &Simulator::currentTime)
        .def("endTime", &Simulator::endTime)
        .def("timeStep", &Simulator::timeStep)
        .def("to_str", &Simulator::to_str)
        .def("domain_box", &Simulator::domainBox)
        .def("cell_width", &Simulator::cellWidth)
        .def("dump_diagnostics", &Simulator::dump_diagnostics, py::arg("timestamp"),
             py::arg("timestep"))
        .def("dump_restarts", &Simulator::dump_restarts, py::arg("timestamp"), py::arg("timestep"));
}

template<typename Dimension, typename InterpOrder, typename NbRefinedPart,
         MHDOpts::TimeIntegratorType TI, MHDOpts::ReconstructionType RC,
         MHDOpts::SlopeLimiterType SL, MHDOpts::RiemannSolverType RS, bool Hall, bool Resistivity,
         bool HyperResistivity>
class Registerer
{
    static constexpr auto dim           = Dimension{}();
    static constexpr auto interp        = InterpOrder{}();
    static constexpr auto nbRefinedPart = NbRefinedPart{}();

    static constexpr SimOpts opts{
        dim, interp, nbRefinedPart, {TI, RC, SL, RS, Hall, Resistivity, HyperResistivity}};

    using Sim = Simulator<opts>;
    using DW  = DataWrangler<opts>;

public:
    static constexpr void declare_etc(py::module& m)
    {
        constexpr auto opts = SimOpts{PHARE_SIM_STR};

        std::string name = "DataWrangler";

        py::class_<DW, py::smart_holder>(m, name.c_str())
            .def(py::init<std::shared_ptr<Sim> const&, std::shared_ptr<amr::Hierarchy> const&>())
            .def(py::init<std::shared_ptr<ISimulator> const&,
                          std::shared_ptr<amr::Hierarchy> const&>())
            .def("sync_merge", &DW::sync_merge)
            .def("getPatchLevel", &DW::getPatchLevel)
            .def("getNumberOfLevels", &DW::getNumberOfLevels);

        using PL = PatchLevel<opts>;
        name     = "PatchLevel";

        py::class_<PL, py::smart_holder>(m, name.c_str())
            .def("getEM", &PL::getEM)
            .def("getE", &PL::getE)
            .def("getB", &PL::getB)
            .def("getBx", &PL::getBx)
            .def("getBy", &PL::getBy)
            .def("getBz", &PL::getBz)
            .def("getEx", &PL::getEx)
            .def("getEy", &PL::getEy)
            .def("getEz", &PL::getEz)
            .def("getVix", &PL::getVix)
            .def("getViy", &PL::getViy)
            .def("getViz", &PL::getViz)
            .def("getDensity", &PL::getDensity)
            .def("getBulkVelocity", &PL::getBulkVelocity)
            .def("getPopDensities", &PL::getPopDensities)
            .def("getPopFluxes", &PL::getPopFlux)
            .def("getFx", &PL::getFx)
            .def("getFy", &PL::getFy)
            .def("getFz", &PL::getFz)
            .def("getParticles", &PL::getParticles, py::arg("userPopName") = "all");
    }

    static constexpr void declare_macro_sim(py::module& m)
    {
        using Sim = Simulator<SimOpts{PHARE_SIM_STR}>;

        std::string name = "Simulator";
        declareSimulator<Sim>(
            py::class_<Sim, py::smart_holder>(m, name.c_str())
                .def_property_readonly_static("dims", [](py::object) { return Sim::dimension; })
                .def_property_readonly_static("interp_order",
                                              [](py::object) { return Sim::interp_order; })
                .def_property_readonly_static("refined_particle_nbr",
                                              [](py::object) { return Sim::nbRefinedPart; }));

        name = "make_simulator";
        m.def(name.c_str(), [](std::shared_ptr<PHARE::amr::Hierarchy> const& hier) {
            return makeSimulator<Sim>(hier);
        });


        declare_etc(m);
    }
};

} // namespace PHARE::pydata

#endif
