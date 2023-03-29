#include <map>
#include <string>

#include "DistributedIR/graph.hpp"
#include "adapters/tensorflow/rpc/graph.pb.h"
#ifndef ADAPTERS_TENSORFLOW_RPC_RPC_UTIL_H
#define ADAPTERS_TENSORFLOW_RPC_RPC_UTIL_H

namespace framework {

std::map<std::string, std::string> GetDeviceMapFromMessage(framework::rpc::Graph const& graph);

framework::rpc::Graph ConvertGraphToMessage(framework::Graph& graph);

framework::Graph ConvertMessageToGraph(const framework::rpc::Graph& graph);
}  // namespace framework
#endif
