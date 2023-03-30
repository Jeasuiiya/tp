#include "common/util.hpp"
#include "cost_graph/cost_graph.hpp"
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"

#ifndef FRAMEWORK_AWARE_INTERFACE_H
#define FRAMEWORK_AWARE_INTERFACE_H

namespace py = pybind11;

namespace framework {
class [[gnu::visibility("hidden")]] AwareInterface {
  private:
    py::dict config_params;
    py::dict simulator_params;
    py::object py_aware_interface;
    py::object networkx_graph;

  public:
    AwareInterface() = default;
    AwareInterface(py::dict _config_params, py::dict _simulator_params, py::object _networkx_graph)
        : config_params(std::move(_config_params)),
          simulator_params(std::move(_simulator_params)),
          networkx_graph(std::move(_networkx_graph)) {
        py_aware_interface = py::module::import("framework.aware.interface")
                                 .attr("PyAwareInterface")(config_params, simulator_params, networkx_graph);
    };

    bool StartReinLearningModule();
    void GetReinLearningBestPlacement(std::map<std::string, std::string>* best_placement);
};
}  // namespace framework

#endif
