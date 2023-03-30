#include "policy/aware/networkx_generator.h"

namespace framework {
void NetworkxGenerator::ConvertMergedCostGraph() {
    MergedCostGraph& merged_cost_graph = GetMergedCostGraph();
    for (auto& node : merged_cost_graph.GetMergedCostNodes()) {
        std::string& name = node.GetName();
        const int64_t cost = node.GetComputeCost();
        const int64_t mem = node.GetMemoryCost();
        std::vector<int64_t>& out_sizes = node.GetOutputCommCosts();
        std::string& device = node.GetDevice();
        std::vector<std::string>& outputs = node.GetOutputs();
        NetworkxGraphAddNode(name, cost, mem, device);
        for (unsigned int i = 0; i < outputs.size(); i++) {
            NetworkxGraphAddEdge(name, outputs[i], out_sizes[i]);
        }
    }
}
}  // namespace framework
