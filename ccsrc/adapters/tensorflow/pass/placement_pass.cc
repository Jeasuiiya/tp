#include "adapters/tensorflow/pass/placement_pass.h"

#include <map>
#include <string>
#include <utility>

#include "DistributedIR/graph.hpp"
#include "adapters/tensorflow/rpc/client.h"
#include "adapters/tensorflow/rpc/graph.pb.h"
#include "adapters/tensorflow/rpc/util.h"
#include "common/fmt.hpp"
#include "cost_graph/common.hpp"
#include "policy/fd-dps/fddps_algorithm.h"
#include "range/v3/all.hpp"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
std::string GetPlacementConfPrefixEnvVar() {
    const char* prefix_env = getenv("TF_PLACEMENT_PREFIX");
    if (!prefix_env) {
        LOG(WARNING) << "TF_PLACEMENT_PREFIX is not set";
        return "";
    }

    std::string result = prefix_env;

    return result;
}

std::string GetPlacementConfPathEnvVar() {
    const char* prefix_env = getenv("TF_PLACEMENT_PATH");
    if (!prefix_env) {
        LOG(WARNING) << "TF_PLACEMENT_PATH is not set";
        return "";
    }

    std::string result = prefix_env;

    return result;
}

std::string GetPlacementRpcAddressEnvVar() {
    const char* address = getenv("TF_PLACEMENT_RPC_ADDRESS");
    if (!address) {
        LOG(WARNING) << "TF_PLACEMENT_RPC_ADDRESS is not set";
        return "";
    }
    return std::string{address};
}

enum class PolicyType {
    None,
    Aware,
    FdDps,
};

PolicyType GetPlacementPolicyVar() {
    const char* policy = getenv("TF_PLACEMENT_POLICY");
    if (!policy) {
        return PolicyType::None;
    }
    std::string policy_str{policy};
    if (policy_str == "aware") {
        return PolicyType::Aware;
    }
    if (policy_str == "fddps") {
        return PolicyType::FdDps;
    }
    return PolicyType::None;
}

struct ConvertContext {
    std::map<std::string, const NodeDef*> name_to_node;
    std::map<std::string, std::vector<std::pair<const NodeDef*, int>>> name_to_output;

    explicit ConvertContext(const GraphDef& graphdef) {
        for (const auto& node_def : graphdef.node()) {
            name_to_node.insert({node_def.name(), &node_def});
        }
        for (const auto& node_def : graphdef.node()) {
            for (const auto& it : node_def.input()) {
                auto view = it | ranges::views::split(':') | ranges::to<std::vector<std::string>>();
                auto output = name_to_output.find(view[0]);
                if (output != name_to_output.end()) {
                    auto& outputs = output->second;
                    outputs.emplace_back(name_to_node[node_def.name()], view.size() == 1 ? 0 : std::stoi(view[1]));
                } else {
                    name_to_output.insert(
                        {view[0], {{name_to_node[node_def.name()], view.size() == 1 ? 0 : std::stoi(view[1])}}});
                }
            }
        }
    }
};

void SetDevice(Graph& g, std::map<std::string, std::string> node_to_device) {
    for (int i = 0; i < g.num_node_ids(); ++i) {
        auto* node = g.FindNodeId(i);
        auto find_iter = node_to_device.find(node->name());
        if (find_iter != node_to_device.end()) {
            // VLOG(0) << "before assigned:" << node->assigned_device_name();
            // VLOG(0) << "before requested:" << node->requested_device();
            node->set_requested_device(find_iter->second);
            // node->set_assigned_device_name(device_num);
            // VLOG(0) << "after assigned:" << node->assigned_device_name();
            // VLOG(0) << "after requested:" << node->requested_device();
        }
    }
}

std::map<std::string, std::string> GetDeviceMapFromGraph(framework::Graph& graph) {
    return graph.Nodes() | ranges::views::transform([](auto& a) { return std::make_pair(a->Name(), a->Device()); })
           | ranges::to<std::map<std::string, std::string>>();
}

std::map<std::string, std::string> GetDeviceMapFromCostNodes(std::vector<framework::CostNode>& nodes) {
    return nodes | ranges::views::transform([](auto& a) { return std::make_pair(a.GetName(), a.GetDevice()); })
           | ranges::to<std::map<std::string, std::string>>();
}

framework::Graph ConvertGraphDefToGraph(const GraphDef& graph_def) {
    framework::Graph graph;
    auto context = ConvertContext(graph_def);

    for (const auto& node_def : graph_def.node()) {
        framework::NodeBase node;
        node.Name(node_def.name());
        node.Op(node_def.op());
        node.Device(node_def.device());
        // input data
        for (int id = 0; id < node_def.input_size(); id++) {
            const std::string& inputbyid = node_def.input(id);
            auto view = inputbyid | ranges::views::split(':') | ranges::to<std::vector<std::string>>();
            std::string index = "0";
            if (view.size() == 2) {
                index = view[1];
            }
            std::string input_data = fmt::format("{}:{}", view[0], index);
            std::string input_node = view[0];
            node.InputsData().push_back(input_data);
            node.AddInput(input_node);
        }
        auto sorted_outputs = std::move(context.name_to_output[node_def.name()]) | ranges::actions::unique
                              | ranges::actions::sort([](const auto& a, auto& b) { return a.second < b.second; });
        for (const auto& i : sorted_outputs) {
            node.AddOutput(i.first->name());
            auto outputs_data = fmt::format("{}:{}", node_def.name(), i.second);
            node.OutputsData().push_back(outputs_data);
        }
        auto attrmap = node_def.attr();
        std::string shapevalue;
        shapevalue = attrmap["shape"].s();
        node.Attrs().insert(std::pair<std::string, std::string>("shape", shapevalue));
        graph.AddNode(node);
    }
    return graph;
}

Status PlacementPass::Run(const GraphOptimizationPassOptions& options) {
    VLOG(INFO) << "PlacementPass";
    VLOG(INFO) << "is_function_graph: " << options.is_function_graph;

    GraphDef graph_def;
    (*options.graph)->ToGraphDef(&graph_def);
    auto graph = ConvertGraphDefToGraph(graph_def);
    std::map<std::string, std::string> device_map;
    auto policy = GetPlacementPolicyVar();
    if (policy == PolicyType::None) {
        LOG(WARNING) << "TF_PLACEMENT_POLICY is not set. skip PlacementPass.";
        return Status::OK();
    }
    if (policy == PolicyType::Aware) {
        auto rpc_address = GetPlacementRpcAddressEnvVar();
        if (rpc_address.empty()) {
            LOG(WARNING) << "TF_PLACEMENT_RPC_ADDRESS is not set. skip!";
            return Status::OK();
        }
        auto rpc_graph = framework::ConvertGraphToMessage(graph);

        framework::RpcServiceClient client(grpc::CreateChannel(rpc_address, grpc::InsecureChannelCredentials()));
        auto r = client.Call(rpc_graph);
        if (r.has_error()) {
            VLOG(WARNING) << fmt::format("call rpc error. {}", r.error().text);
            return Status::OK();
        }
        device_map = std::move(r.value());
        VLOG(INFO) << fmt::to_string(device_map);
    } else if (policy == PolicyType::FdDps) {
        std::vector<framework::Device> devices;
        for (auto* i : options.device_set->devices()) {
            auto memory = i->attributes().memory_limit();
            devices.emplace_back(framework::DeviceStatus::Idle, framework::DeviceTypeFrom(i->device_type()), i->name(),
                                 memory, memory, 0);
        }
        framework::CostGraph cost_graph = ConvertGraphToCostGraph(graph);
        framework::FDDPSAlgorithm fddps_algorithm(cost_graph, devices);
        auto r = fddps_algorithm.Placement();
        if (r.has_error()) {
            VLOG(INFO) << fmt::format("call fddps error. {}", r.error().text);
            return Status::OK();
        }
        device_map = GetDeviceMapFromCostNodes(r.value());
    }

    SetDevice(**options.graph, device_map);
    return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1, PlacementPass);
}  // namespace tensorflow
