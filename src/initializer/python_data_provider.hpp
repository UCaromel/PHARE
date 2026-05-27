// this file includes python which has conflicting #defines with SAMRAI
//  so include it last

#ifndef PHARE_PYTHON_DATA_PROVIDER_HPP
#define PHARE_PYTHON_DATA_PROVIDER_HPP

#include "initializer/data_provider.hpp"

#include <filesystem>

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/functional.h>

namespace py = pybind11;

namespace PHARE
{
namespace initializer
{
    namespace detail
    {
        inline py::scoped_interpreter& pythonInterpreter()
        {
            // Reinitializing the embedded interpreter between GTest cases is not
            // reliable here. Keep one interpreter alive for the whole process so
            // successive PythonDataProvider instances reuse the same runtime.
            static py::scoped_interpreter guard{};
            return guard;
        }
    } // namespace detail

    class __attribute__((visibility("hidden"))) PythonDataProvider : public DataProvider
    {
    public:
        PythonDataProvider()
        {
            detail::pythonInterpreter();
        }
        PythonDataProvider(std::string moduleName)
            : moduleName_{moduleName}
        {
            detail::pythonInterpreter();
        }

        /**
         * @brief read overrides the abstract DataProvider::read method. This method basically
         * executes the user python script that fills the dictionnary.
         */
        void read() override
        {
            auto sys  = py::module::import("sys");
            auto path = sys.attr("path");

            // CTest may launch the executable by absolute path from a parent build
            // directory, while local PHARE entry points also expect `job.py` to be
            // discoverable from the executable directory. Make both search roots
            // explicit for embedded Python before importing the user job module.
            auto addPath = [&](std::filesystem::path const& candidate) {
                if (!candidate.empty())
                    path.attr("insert")(0, candidate.string());
            };

            addPath(std::filesystem::current_path());
            addPath(std::filesystem::read_symlink("/proc/self/exe").parent_path());

            auto module = py::module::import(initModuleName_.c_str());
            module.attr("get_user_inputs")(moduleName_);
        }



    private:
        std::string moduleName_{"job"};
        std::string initModuleName_{"pyphare.pharein.init"};
    };

} // namespace initializer

} // namespace PHARE

#endif // DATA_PROVIDER_HPP
