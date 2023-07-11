#include "adapters/tensorflow/pass/placement_pass.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "DistributedIR/graph.hpp"
#include "adapters/tensorflow/rpc/client.h"
#include "adapters/tensorflow/rpc/graph.pb.h"
#include "adapters/tensorflow/rpc/util.h"
#include "common/fmt.hpp"
#include "cost_graph/common.hpp"
#include "google/protobuf/text_format.h"
#include "jsoncpp/json/json.h"
#include "policy/fd-dps/fddps_algorithm.h"
#include "policy/sgp/graphPartition.h"
#include "range/v3/all.hpp"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
std::map<std::string, framework::DataType> type_map = {{"int32", framework::DataType::I32},
                                                       {"float32", framework::DataType::F32},
                                                       {"float32_ref", framework::DataType::F32}};
void CalculateMemory(std::shared_ptr<framework::NodeBase> node) {
    // 输入数据大小
    std::int64_t input_total_size = 1;
    std::int64_t output_total_size = 1;
    for (auto inputport : node->InputPorts()) {
        std::int64_t input_size = 1;
        framework::shape_t shape = inputport.entity.tensor.shape;
        for (const auto& dim : shape) {
            input_size *= dim;
        }
        framework::DataType dtype = inputport.entity.tensor.dtype;
        input_size *= 4;
        input_total_size += input_size;
    }
    node->InputMemory(input_total_size);
    // 输出数据大小
    for (auto outputport : node->OutputPorts()) {
        std::int64_t output_size = 1;
        framework::shape_t shape = outputport.entity.shape;
        for (const auto& dim : shape) {
            output_size *= dim;
        }
        framework::DataType dtype = outputport.entity.dtype;
        output_size *= 4;
        output_total_size += output_size;
    }
    node->OutputMemory(output_total_size);
    // 节点结构大小
    node->PersistentMemory(sizeof(*node));
}

void CalculateCost(std::shared_ptr<framework::NodeBase> node) {
    std::int64_t compute_cost = ceil(
        0.2
        * ceil((node->InputMemory() + node->OutputMemory())
               / (1 + exp(ceil(-(fabs(node->InputMemory() - node->OutputMemory())) / (1 + node->OutputMemory()))))));
    node->ComputeCost(compute_cost);
}

std::string GetUseCPU() {
    const char* fetch_node = getenv("USECPU");
    if (!fetch_node) {
        LOG(WARNING) << "USECPU is not set";
        return "use";
    }
    return std::string{fetch_node};
}

std::string GetGraphJson() {
    const char* fetch_node = getenv("GRAPH");
    if (!fetch_node) {
        LOG(WARNING) << "graph path is not set";
        return "";
    }
    return std::string{fetch_node};
}

int GetBatchSize() {
    const char* fetch_node = getenv("BATCH_SIZE");
    if (!fetch_node) {
        LOG(WARNING) << "batch size is not set";
        return 32;
    }
    return std::stoi(fetch_node);
}

int GetDeviceNum() {
    const char* fetch_node = getenv("DEVICE_NUM");
    if (!fetch_node) {
        LOG(WARNING) << "device num is not set";
        return 2;
    }
    return std::stoi(fetch_node);
}

std::string GetGtemFetchNode() {
    const char* fetch_node = getenv("TF_GRAPH_FETCH_NODE");
    if (!fetch_node) {
        LOG(WARNING) << "train_op is not set";
        return "";
    }
    return std::string{fetch_node};
}

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
    SGP,
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
    if (policy_str == "SGP") {
        return PolicyType::SGP;
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
            node->set_requested_device(find_iter->second);
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
    Json::Reader reader;
    Json::Value root;
    std::ifstream is;
    is.open(GetGraphJson(), std::ios::binary);
    reader.parse(is, root);
    for (const auto& node_def : graph_def.node()) {
        auto jsonnode = root[node_def.name()];
        framework::NodeBase node;
        node.Attrs().insert(
            std::pair<std::string, std::string>("colocation_group", jsonnode["colocation_group"].asString()));
        node.Name(node_def.name());
        node.Op(node_def.op());
        node.Device(node_def.device());
        for (int id = 0; id < node_def.input_size(); id++) {
            const std::string& input = node_def.input(id);
            auto view = input | ranges::views::split(':') | ranges::to<std::vector<std::string>>();
            std::string index = "0";
            if (view.size() == 2) {
                index = view[1];
            }
            std::string input_node = view[0];
            framework::shape_t shape;
            framework::DataType dtype;
            int indexi = std::stoi(index);
            if (input_node.rfind('^', 0) == 0) {
                input_node = input_node.substr(1);
            }
            for (Json::Value dim : jsonnode["inputs"][input_node]["shape"]) {
                shape.push_back(dim.asInt());
            }
            dtype = type_map[jsonnode["inputs"][input_node]["dtype"].asString()];
            auto result = node.AddInputPort(input_node, indexi, id, dtype, shape);
            node.AddInput(input);
        }
        auto sorted_outputs = std::move(context.name_to_output[node_def.name()]) | ranges::actions::unique
                              | ranges::actions::sort([](const auto& a, auto& b) { return a.second < b.second; });
        for (const auto& i : sorted_outputs) {
            node.AddOutput(i.first->name());
            framework::shape_t shape;
            framework::DataType dtype;
            for (Json::Value dim : jsonnode["outputs"][std::to_string(i.second)]["shape"]) {
                shape.push_back(dim.asInt());
            }
            dtype = type_map[jsonnode["outputs"][std::to_string(i.second)]["dtype"].asString()];
            auto result = node.AddOutputPort(dtype, shape, i.second);
        }
        node.OutputsNum(node.Outputs().size());
        graph.AddNode(node);
    }
    for (auto node : graph.Nodes()) {
        CalculateMemory(node);
        CalculateCost(node);
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
            devices.emplace_back(framework::DeviceTypeFrom(i->device_type()), i->name(), memory, memory, 0);
        }
        framework::CostGraph cost_graph = ConvertGraphToCostGraph(graph);
        framework::FDDPSAlgorithm fddps_algorithm(cost_graph, devices);
        auto r = fddps_algorithm.Placement();
        if (r.has_error()) {
            VLOG(INFO) << fmt::format("call fddps error. {}", r.error().text);
            return Status::OK();
        }
        device_map = GetDeviceMapFromCostNodes(r.value());
    } else if (policy == PolicyType::SGP) {
        std::vector<framework::Device> devices;
        for (auto* i : options.device_set->devices()) {
            auto memory = i->attributes().memory_limit();
            if (GetUseCPU() != "use") {
                if (i->device_type() != "CPU") {
                    devices.emplace_back(framework::DeviceTypeFrom(i->device_type()), i->name(), memory, memory, 0);
                }
            } else {
                devices.emplace_back(framework::DeviceTypeFrom(i->device_type()), i->name(), memory, memory, 0);
            }
        }
        framework::Partition Partition(graph, GetDeviceNum(), devices);
        device_map = Partition.op_group;
    }

    SetDevice(**options.graph, device_map);
    return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1, PlacementPass);
}  // namespace tensorflow
