#include "cost_graph/common.hpp"
#include "cost_graph/cost_graph.hpp"

#ifndef FRAMEWORK_FUSION_AWARE_FUSION_H
#define FRAMEWORK_FUSION_AWARE_FUSION_H

namespace framework {
class AwareFusion {
  private:
    CostGraph cost_graph;
    MergedCostGraph merged_cost_graph;

  public:
    AwareFusion() = default;
    explicit AwareFusion(CostGraph& _cost_graph) : cost_graph(_cost_graph) {
        merged_cost_graph = InitMergedCostGraph(_cost_graph);
    }
    AwareFusion(CostGraph& _cost_graph, MergedCostGraph& _merged_cost_graph)
        : cost_graph(_cost_graph), merged_cost_graph(_merged_cost_graph) {}
    explicit AwareFusion(MergedCostGraph& _merged_cost_graph) : merged_cost_graph(_merged_cost_graph) {
        cost_graph = ConvertMergedCostGraphToCostGraph(_merged_cost_graph);
    }
    virtual ~AwareFusion() = default;

    DECL_ACCESSOR(GetCostGraph, SetCostGraph, cost_graph, M)
    DECL_ACCESSOR(GetMergedCostGraph, SetMergedCostGraph, merged_cost_graph, M)

    MergedCostGraph GenerateFusedGraph();
};
}  // namespace framework

#endif
