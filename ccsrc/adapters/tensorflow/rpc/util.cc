#include "adapters/tensorflow/rpc/util.h"

#include "range/v3/all.hpp"

namespace framework {

std::map<std::string, std::string> GetDeviceMapFromMessage(framework::rpc::Graph const& graph) {
    return graph.node() | ranges::views::transform([](auto& a) { return std::make_pair(a.name(), a.device()); })
           | ranges::to<std::map<std::string, std::string>>();
}

framework::rpc::Graph ConvertGraphToMessage(framework::Graph& graph) {
    framework::rpc::Graph rpc_graph;
    for (const auto& n : graph.Nodes()) {
        framework::rpc::Node rpc_node;
        rpc_node.set_name(n->Name());
        rpc_node.set_op(n->Op());
        rpc_node.set_device(n->Device());
        rpc_node.set_computecost(n->ComputeCost());
        rpc_node.set_persistent_memory(n->PersistentMemory());
        rpc_node.set_output_memory(n->OutputMemory());
        for (const auto& i : n->Inputs()) {
            rpc_node.add_inputs(i);
        }
        // for (const auto& i : n->InputsData()) {
        //     rpc_node.add_inputs_data(i);
        // }
        for (const auto& i : n->Outputs()) {
            rpc_node.add_outputs(i);
        }
        // for (const auto& i : n->OutputsData()) {
        //     rpc_node.add_outputs_data(i);
        // }
        rpc_node.mutable_attr()->insert({"shape", n->Attrs()["shape"]});
        rpc_graph.mutable_node()->Add(std::move(rpc_node));
    }
    return rpc_graph;
}
framework::Graph ConvertMessageToGraph(const framework::rpc::Graph& rpc_graph) {
    framework::Graph graph;
    for (const auto& n : rpc_graph.node()) {
        framework::NodeBase node;
        node.Name(n.name());
        node.Op(n.op());
        node.Device(n.device());
        node.ComputeCost(n.computecost());
        node.PersistentMemory(n.persistent_memory());
        node.OutputMemory(n.output_memory());
        for (const auto& i : n.inputs()) {
            node.AddInput(i);
        }
        // for (const auto& i : n.inputs_data()) {
        //     node.AddInputsData(i);
        // }
        for (const auto& i : n.outputs()) {
            node.AddOutput(i);
        }
        // for (const auto& i : n.outputs_data()) {
        //     node.AddOutputsData(i);
        // }
        auto shape_find = n.attr().find("shape");
        std::string shape;
        if (shape_find != n.attr().end()) {
            shape = shape_find->second;
        }
        node.Attrs().insert({"shape", shape});
        graph.AddNode(std::move(node));
    }
    return graph;
}
}  // namespace framework
