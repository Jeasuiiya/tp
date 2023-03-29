#include "adapters/tensorflow/pass/placement_pass.h"

#include <map>
#include <string>
#include <utility>

#include "DistributedIR/graph.hpp"
#include "adapters/tensorflow/rpc/client.h"
#include "adapters/tensorflow/rpc/graph.pb.h"
#include "adapters/tensorflow/rpc/util.h"
#include "common/fmt.hpp"
#include "range/v3/all.hpp"
#include "tensorflow/tsl/platform/status.h"

namespace fw = framework;

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

static Status GetPlacementConf(std::string& result) {
    std::string prefix = GetPlacementConfPrefixEnvVar();
    std::string path = GetPlacementConfPathEnvVar();
    if (path.empty()) {
        return Status(error::Code::UNKNOWN, "config is not set.");
    }

    auto* env = tensorflow::Env::Default();

    prefix += path;
    TF_RETURN_IF_ERROR(ReadFileToString(env, prefix, &result));
    VLOG(0) << "read: " << result;
    return Status::OK();
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
    auto rpc_address = GetPlacementRpcAddressEnvVar();
    if (rpc_address.empty()) {
        LOG(WARNING) << "TF_PLACEMENT_RPC_ADDRESS is not set. skip!";
        return Status::OK();
    }
    GraphDef graph_def;
    (*options.graph)->ToGraphDef(&graph_def);
    auto graph = ConvertGraphDefToGraph(graph_def);
    auto rpc_graph = framework::ConvertGraphToMessage(graph);

    framework::RpcServiceClient client(grpc::CreateChannel(rpc_address, grpc::InsecureChannelCredentials()));
    auto reply = client.Call(rpc_graph);
    if (reply.has_error()) {
        VLOG(WARNING) << fmt::format("call rpc error. {}", reply.error().text);
        return Status::OK();
    }
    auto device_map = reply.value();
    VLOG(INFO) << fmt::to_string(device_map);

    SetDevice(**options.graph, device_map);
    return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1, PlacementPass);
}  // namespace tensorflow
