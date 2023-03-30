#include "cost_graph/cost_graph.hpp"
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"

#ifndef FRAMEWORK_NETWORKX_GENERATOR_H
#define FRAMEWORK_NETWORKX_GENERATOR_H

namespace py = pybind11;

namespace framework {
class [[gnu::visibility("hidden")]] NetworkxGenerator {
  private:
    MergedCostGraph merged_cost_graph;
    // networkx_graph = py::module::import("policy.python.graph").attr("Graph");
    py::object networkx_graph;

  public:
    NetworkxGenerator() = default;
    explicit NetworkxGenerator(MergedCostGraph& _merged_cost_graph) : merged_cost_graph(_merged_cost_graph) {
        networkx_graph = py::module::import("framework.aware.graph").attr("Graph")();
    }

    virtual ~NetworkxGenerator() = default;

    void ConvertMergedCostGraph();
    void NetworkxGraphAddNode(std::string& name, int64_t cost, int64_t mem, std::string& device) {
        networkx_graph.attr("add_node")(name, cost, mem, device);
    };
    void NetworkxGraphAddEdge(std::string& left_node, std::string& right_node, int64_t output_memory) {
        networkx_graph.attr("add_edge")(left_node, right_node, output_memory);
    };

    DECL_ACCESSOR(GetMergedCostGraph, SetMergedCostGraph, merged_cost_graph, M)
    DECL_ACCESSOR(GetNetworkxGraph, SetNetworkxGraph, networkx_graph, M)
};

}  // namespace framework

#endif