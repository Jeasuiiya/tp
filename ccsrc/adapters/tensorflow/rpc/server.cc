#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "adapters/tensorflow/rpc/graph.pb.h"
#include "adapters/tensorflow/rpc/service.grpc.pb.h"
#include "adapters/tensorflow/rpc/service.pb.h"
#include "adapters/tensorflow/rpc/util.h"
#include "common/log.h"
#include "cost_graph/common.hpp"
#include "fmt/format.h"
#include "fusion/aware_fusion.h"
#include "grpcpp/ext/proto_server_reflection_plugin.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/health_check_service_interface.h"
#include "policy/aware/aware_interface.h"
#include "policy/aware/networkx_generator.h"
#include "policy/trinity/trinity_interface.h"

using framework::rpc::CallRequest;
using framework::rpc::CallResponse;
using framework::rpc::RpcService;
using RpcGraph = framework::rpc::Graph;
using RpcNode = framework::rpc::Node;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
namespace py = pybind11;

namespace framework {

void PrepareParams(py::dict& config_params, py::dict& simulator_params) {
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
    config_params["output_save_path"] = "./TestAwareOutput";
    config_params["turn_based_baseline"] = true;
    config_params["print_freq"] = 1;
    config_params["save_freq"] = 1;
    config_params["eval_freq"] = 5;
    config_params["bl_n_rnds"] = 1000;
    config_params["mem_penalty"] = 3.0;
    config_params["max_mem"] = 10.0;
    config_params["max_runtime_mem_penalized"] = 10.0;
    config_params["node_traversal_order"] = "random";

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
}

/**
 * RpcServiceImpl
 * implement RpcService
 */
class RpcServiceImpl final : public RpcService::Service {
    Status Call(ServerContext* /*context*/, const CallRequest* request, CallResponse* reply) override {
        std::cerr << "received request" << std::endl;
        try {
            auto graph = ConvertMessageToGraph(request->graph());
            std::string policy = request->policy();
            py::scoped_interpreter python;
            py::dict config_params;
            py::dict simulator_params;
            py::int_ n_devs = 8;
            py::int_ num_cpus = 1;
            py::bool_ verbose = true;
            py::int_ step = 50;
            py::object hparams = py::module::import("framework.trinity.trinity_program").attr("trinity_mian_hparams")();
            py::object gcontroller =
                py::module::import("framework.trinity.cluster").attr("TrinityControllerTest")(n_devs, num_cpus);
            py::object gcluster = gcontroller.attr("getCluster")();
            PrepareParams(config_params, simulator_params);

            CostGraph cost_graph = ConvertGraphToCostGraph(graph);

            MergedCostGraph merged_cost_graph = InitMergedCostGraph(cost_graph);
            AwareFusion aware_fusion(cost_graph, merged_cost_graph);
            // MergedCostGraph new_merged_cost_graph = aware_fusion.GenerateFusedGraph();
            NetworkxGenerator networkx_generator(merged_cost_graph);
            networkx_generator.ConvertMergedCostGraph();
            py::object networkx_graph = networkx_generator.GetNetworkxGraph();
            std::map<std::string, std::string> best_placement;
            std::map<std::string, RpcNode*> name_to_node;
            auto* result_graph = new RpcGraph(request->graph());
            if (policy == "aware") {
                std::cerr << "start aware" << std::endl;
                AwareInterface aware_interface(config_params, simulator_params, networkx_graph);
                std::cerr << "run aware" << std::endl;
                bool success = aware_interface.StartReinLearningModule();
                std::cerr << "aware finished" << std::endl;
                if (!success) {
                    std::cerr << "aware failed" << std::endl;
                    reply->set_success(false);
                    return Status::OK;
                }
                for (auto& n : *result_graph->mutable_node()) {
                    name_to_node.insert({n.name(), &n});
                }
                aware_interface.GetReinLearningBestPlacement(&best_placement);
            } else if (policy == "trinity") {
                std::cerr << "start trinity" << std::endl;
                TrinityInterface trinity_interface(networkx_graph, gcluster, hparams, verbose, step);
                std::cerr << "run trinity" << std::endl;
                bool success = trinity_interface.StartReinLearningModule();
                std::cerr << "trinity finished" << std::endl;
                if (!success) {
                    std::cerr << "trinity failed" << std::endl;
                    reply->set_success(false);
                    return Status::OK;
                }
                for (auto& n : *result_graph->mutable_node()) {
                    name_to_node.insert({n.name(), &n});
                }
                trinity_interface.GetReinLearningBestPlacement(&best_placement);
            }
            std::vector<MergedCostNode>& merged_cost_nodes = merged_cost_graph.GetMergedCostNodes();
            for (auto& merged_cost_node : merged_cost_nodes) {
                const std::string& node_name = merged_cost_node.GetName();
                std::string device = best_placement[node_name];
                for (auto& cost_name : merged_cost_node.GetCostNodeNames()) {
                    auto it = name_to_node.find(cost_name);
                    if (it != name_to_node.end()) {
                        it->second->set_device(device);
                    }
                }
            }

            reply->set_success(true);
            reply->set_allocated_graph(result_graph);
        } catch (const std::exception& e) {
            reply->set_success(false);
            return Status::OK;
        }

        return Status::OK;
    }
};

void RunServer(std::string address) {
    RpcServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    //   grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    SPDLOG_INFO("Server listening on", address);

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

}  // namespace framework
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << fmt::format("usage: {} [TARGET]", argv[0]) << std::endl;
        exit(1);
    }
    std::string server_address{argv[1]};
    framework::RunServer(server_address);
    return 0;
}
