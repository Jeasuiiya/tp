#include "adapters/tensorflow/rpc/client.h"

#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

#include "fmt/format.h"
using framework::rpc::CallRequest;
using framework::rpc::CallResponse;
using framework::rpc::RpcService;
using RpcGraph = framework::rpc::Graph;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
namespace framework {

RpcServiceClient::RpcServiceClient(std::shared_ptr<Channel> channel) : stub_(RpcService::NewStub(channel)) {}

// Assembles the client's payload, sends it and presents the response back
// from the server.
cpp::result<std::map<std::string, std::string>, Error> RpcServiceClient::Call(const RpcGraph& graph) {
    try {
        // Data we are sending to the server.
        CallRequest request;
        request.set_allocated_graph(new RpcGraph(graph));

        // Container for the data we expect from the server.
        CallResponse reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->Call(&context, request, &reply);

        // Act upon its status.
        if (!status.ok() || !reply.success()) {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return cpp::fail(Error(Kind::Internal, fmt::format("{}:{}", status.error_code(), status.error_message())));
        }
        return GetDeviceMapFromMessage(reply.graph());

    } catch (const std::exception& e) {
        return cpp::fail(Error(Kind::Unknown, e.what()));
    }
}
}  // namespace framework