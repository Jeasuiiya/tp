#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <common/util.hpp>
#include <cost_graph/cost_graph.hpp>

#ifndef FRAMEWORK_TRINITY_INTERFACE_H
#define FRAMEWORK_TRINITY_INTERFACE_H

namespace py = pybind11;

namespace framework {
class [[gnu::visibility("hidden")]] TrinityInterface {
  private:
    py::object networkx_graph;
    py::object gcluster;
    py::object hparams;
    py::bool_ verbose;
    py::int_ step;
    py::object py_trinity_interface;

  public:
    TrinityInterface() = default;
    TrinityInterface(py::object _networkx_graph, py::object _gcluster, py::object _hparams, py::bool_ _verbose,
                     py::int_ _step)
        : networkx_graph(std::move(_networkx_graph)),
          gcluster(std::move(_gcluster)),
          hparams(std::move(_hparams)),
          verbose(std::move(_verbose)),
          step(std::move(_step)) {
        py_trinity_interface = py::module::import("framework.trinity.interface")
                                   .attr("PyTrinityInterface")(networkx_graph, gcluster, hparams, verbose, step);
    };

    bool StartReinLearningModule();
    void GetReinLearningBestPlacement(std::map<std::string, std::string>* best_placement);
};
}  // namespace framework

#endif
