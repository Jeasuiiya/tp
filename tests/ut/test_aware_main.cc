#include "cost_graph/common.hpp"
#include "fusion/aware_fusion.h"
#include "gtest/gtest.h"
#include "policy/aware/aware_interface.h"
#include "policy/aware/networkx_generator.h"

using namespace framework;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int main() {
    py::scoped_interpreter python;
    py::module::import("sys").attr("path").attr("append")(MACRO_STRINGIFY(SOURCE_PYTHONPATH));

    py::dict config_params;
    config_params["seed"] = 42;
    config_params["n_devs"] = 4;
    config_params["radial_mp"] = 1;
    config_params["sage_sample_ratio"] = 1.0;
    config_params["pgnn_c"] = 0.2;
    config_params["pgnn_neigh_cutoff"] = 4;
    config_params["pgnn_anchor_exponent"] = 4;
    config_params["pgnn_aggregation"] = "max";
    config_params["n_eps"] = 20;
    config_params["disc_factor"] = 1.0;
    config_params["output_save_path"] = "/home/DatasetHouse/AwareOutput";
    config_params["turn_based_baseline"] = true;
    config_params["print_freq"] = 1;
    config_params["save_freq"] = 1;
    config_params["eval_freq"] = 5;
    config_params["bl_n_rnds"] = 1000;
    config_params["mem_penalty"] = 3.0;
    config_params["max_mem"] = 10.0;
    config_params["max_runtime_mem_penalized"] = 10.0;
    config_params["node_traversal_order"] = "random";

    py::dict simulator_params;
    simulator_params["lr_init"] = 1e-3;
    simulator_params["lr_dec"] = 0.95;
    simulator_params["lr_start_decay_step"] = 1e9;
    simulator_params["lr_decay_steps"] = 100;
    simulator_params["lr_min"] = 1e-3;
    simulator_params["lr_dec_approach"] = "exponential";
    simulator_params["ent_dec_init"] = 1.0;
    simulator_params["ent_dec"] = 0.95;
    simulator_params["ent_start_dec_step"] = 1e9;
    simulator_params["ent_dec_steps"] = 100;
    simulator_params["ent_dec_min"] = 0.0;
    simulator_params["ent_dec_lin_steps"] = 0;
    simulator_params["ent_dec_approach"] = "linear";
    simulator_params["optimizer_type"] = "adam";

    NodeBase node_1;
    node_1.Name("input");
    node_1.Device("/GPU:0");
    node_1.ComputeCost(2);
    node_1.PersistentMemory(1);
    node_1.Outputs({"conv1", "conv2"});
    node_1.OutputMemory(10);

    NodeBase node_2;
    node_2.Name("conv1");
    node_2.Device("/GPU:1");
    node_2.ComputeCost(3);
    node_2.Inputs({"input"});
    node_2.InputMemory(2);
    node_2.PersistentMemory(3);
    node_2.Outputs({"conv2"});
    node_2.OutputMemory(3);

    NodeBase node_3;
    node_3.Name("conv2");
    node_3.Device("/GPU:2");
    node_3.ComputeCost(3);
    node_3.Inputs({"input", "conv1"});
    node_3.InputMemory(8);
    node_3.PersistentMemory(4);
    node_3.Outputs({"conv3"});
    node_3.OutputMemory(3);

    NodeBase node_4;
    node_4.Name("conv3");
    node_4.Device("/GPU:1");
    node_4.ComputeCost(3);
    node_4.Inputs({"conv2"});
    node_4.InputMemory(3);
    node_4.PersistentMemory(4);
    node_4.Outputs({"relu"});
    node_4.OutputMemory(3);

    NodeBase node_5;
    node_5.Name("relu");
    node_5.Device("/GPU:1");
    node_5.ComputeCost(3);
    node_5.Inputs({"conv3"});
    node_5.InputMemory(3);
    node_5.PersistentMemory(4);
    node_5.Outputs({"output"});
    node_5.OutputMemory(3);

    NodeBase node_6;
    node_6.Name("output");
    node_6.Device("/GPU:3");
    node_6.ComputeCost(3);
    node_6.Inputs({"relu"});
    node_6.InputMemory(3);
    node_6.PersistentMemory(6);

    Graph graph;
    graph.AddNode(node_1);
    graph.AddNode(node_2);
    graph.AddNode(node_3);
    graph.AddNode(node_4);
    graph.AddNode(node_5);
    graph.AddNode(node_6);

    CostGraph cost_graph = ConvertGraphToCostGraph(graph);
    MergedCostGraph merged_cost_graph = InitMergedCostGraph(cost_graph);
    AwareFusion aware_fusion(cost_graph, merged_cost_graph);
    // MergedCostGraph new_merged_cost_graph = aware_fusion.GenerateFusedGraph();
    NetworkxGenerator networkx_generator(merged_cost_graph);
    networkx_generator.ConvertMergedCostGraph();
    py::object networkx_graph = networkx_generator.GetNetworkxGraph();

    AwareInterface aware_interface(config_params, simulator_params, networkx_graph);
    aware_interface.StartReinLearningModule();
    std::map<std::string, std::string> best_placement;
    aware_interface.GetReinLearningBestPlacement(&best_placement);

    std::vector<MergedCostNode>& merged_cost_nodes = merged_cost_graph.GetMergedCostNodes();
    for (auto& merged_cost_node : merged_cost_nodes) {
        std::string& node_name = merged_cost_node.GetName();
        merged_cost_node.SetDevice(best_placement[node_name]);
    }
}