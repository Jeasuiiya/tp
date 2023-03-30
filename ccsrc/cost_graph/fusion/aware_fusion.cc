#include "fusion/aware_fusion.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

namespace framework {
MergedCostGraph AwareFusion::GenerateFusedGraph() {
    // 先创建一个包含 Edge 的 queue
    std::queue<CostEdge> edge_queue;
    // 把所有的边先装入 queue 中
    std::vector<MergedCostNode>& merged_cost_nodes = merged_cost_graph.GetMergedCostNodes();
    for (auto& merged_cost_node : merged_cost_nodes) {
        std::vector<std::string>& outputs = merged_cost_node.GetOutputs();
        std::vector<int64_t>& output_comm_costs = merged_cost_node.GetOutputCommCosts();
        const std::string left_node = merged_cost_node.GetName();
        for (unsigned int i = 0; i < outputs.size(); i++) {
            const std::string right_node = outputs[i];
            const int64_t comm_cost = output_comm_costs[i];
            CostEdge edge(left_node, right_node, comm_cost);
            edge_queue.push(edge);
        }
    }

    // 创建一个节点和所在组的映射表, 所在组用节点名称命名 (方便后续融合和修改信息使用)
    std::map<std::string, std::string> node_to_group;
    // 初始化
    // 分两个for循环, 结构看起来清晰点
    for (auto& merged_cost_node : merged_cost_nodes) {
        const std::string head_node_name = merged_cost_node.GetName();
        std::vector<std::string>& aggregrated_nodes = merged_cost_node.GetCostNodeNames();
        for (auto& aggregrated_node : aggregrated_nodes) {
            node_to_group.insert(std::make_pair(aggregrated_node, head_node_name));
        }
    }

    // 创建一个 map, 用于根据节点名快速进行 hash 查找
    std::map<std::string, MergedCostNode> merged_cost_node_map;
    for (auto merged_cost_node : merged_cost_nodes) {
        const std::string name = merged_cost_node.GetName();
        merged_cost_node_map.insert(std::make_pair(name, merged_cost_node));
    }

    // 开始进行算子融合
    while (!edge_queue.empty()) {
        CostEdge edge = edge_queue.front();
        edge_queue.pop();

        // 判断是否符合融合条件 条件一 : 出度或者入度为 1
        std::string& left_node_name = edge.GetLeftNode();
        std::string& right_node_name = edge.GetRightNode();

        // 直接将左节点和右节点替换成所在的组
        left_node_name = node_to_group[left_node_name];
        right_node_name = node_to_group[right_node_name];

        if (left_node_name == right_node_name) {
            continue;
        }

        MergedCostNode& left_node = merged_cost_node_map[left_node_name];
        MergedCostNode& right_node = merged_cost_node_map[right_node_name];
        int64_t left_node_out_degree = left_node.GetOutputs().size();
        int64_t right_node_in_degree = right_node.GetInputs().size();
        std::vector<std::string>& left_node_inputs = left_node.GetInputs();
        std::vector<std::string>& right_node_inputs = right_node.GetInputs();
        std::vector<std::string>& left_node_outputs = left_node.GetOutputs();
        std::vector<std::string>& right_node_outputs = right_node.GetOutputs();
        std::vector<int64_t>& left_node_inputs_comm_costs = left_node.GetInputCommCosts();
        std::vector<int64_t>& right_node_inputs_comm_costs = right_node.GetInputCommCosts();
        std::vector<int64_t>& left_node_outputs_comm_costs = left_node.GetOutputCommCosts();
        std::vector<int64_t>& right_node_outputs_comm_costs = right_node.GetOutputCommCosts();
        std::vector<std::string>& left_node_merged_node_names = left_node.GetCostNodeNames();
        std::vector<std::string>& right_node_merged_node_names = right_node.GetCostNodeNames();

        if (left_node_out_degree != 1 && right_node_in_degree != 1) {
            continue;
        }

        // 符合算子融合条件, 开始算子融合 (右节点融合到左节点里面去)
        // 融合 compute_cost, memory_cost, input 和 output
        left_node.SetComputeCost(right_node.GetComputeCost() + left_node.GetComputeCost());
        left_node.SetMemoryCost(right_node.GetMemoryCost() + left_node.GetMemoryCost());
        left_node.SetOutputMemory(right_node.GetOutputMemory() + left_node.GetOutputMemory());

        for (auto& input : left_node_inputs) {
            input = node_to_group[input];
        }
        for (auto& output : left_node_outputs) {
            output = node_to_group[output];
        }
        for (auto& input : right_node_inputs) {
            input = node_to_group[input];
        }
        for (auto& output : right_node_outputs) {
            output = node_to_group[output];
        }

        // 融合边集
        for (unsigned int i = 0; i < right_node_inputs.size(); i++) {
            auto it = find(left_node_inputs.begin(), left_node_inputs.end(), right_node_inputs[i]);
            if (it != left_node_inputs.end()) {
                left_node_inputs_comm_costs[std::distance(left_node_inputs.begin(), it)] +=
                    right_node_inputs_comm_costs[i];
            } else {
                left_node_inputs.push_back(right_node_inputs[i]);
                left_node_inputs_comm_costs.push_back(right_node_inputs_comm_costs[i]);
            }
        }

        for (unsigned int i = 0; i < left_node_inputs.size(); i++) {
            if (left_node_inputs[i] == left_node_name) {
                left_node_inputs.erase(left_node_inputs.begin() + i);
                left_node_inputs_comm_costs.erase(left_node_inputs_comm_costs.begin() + i);
            }
        }

        for (unsigned int i = 0; i < right_node_outputs.size(); i++) {
            auto it = find(left_node_outputs.begin(), left_node_outputs.end(), right_node_outputs[i]);
            if (it != left_node_outputs.end()) {
                left_node_outputs_comm_costs[std::distance(left_node_outputs.begin(), it)] +=
                    right_node_outputs_comm_costs[i];
            } else {
                left_node_outputs.push_back(right_node_outputs[i]);
                left_node_outputs_comm_costs.push_back(right_node_outputs_comm_costs[i]);
            }
        }

        // 删除右节点
        for (unsigned int i = 0; i < left_node_outputs.size(); i++) {
            if (left_node_outputs[i] == right_node_name) {
                left_node_outputs.erase(left_node_outputs.begin() + i);
                left_node_outputs_comm_costs.erase(left_node_outputs_comm_costs.begin() + i);
            }
        }

        // 合并group
        for (auto& right_node_merged_node_name : right_node_merged_node_names) {
            left_node_merged_node_names.push_back(right_node_merged_node_name);
            // 修改完之后, 需要修改 node_to_group
            node_to_group[right_node_merged_node_name] = left_node_name;
        }

        merged_cost_node_map.erase(right_node_name);
    }

    std::vector<MergedCostNode> new_merged_cost_nodes;
    for (auto& merged_cost_node_pair : merged_cost_node_map) {
        MergedCostNode& merged_cost_node = merged_cost_node_pair.second;
        std::vector<std::string>& outputs = merged_cost_node.GetOutputs();
        std::vector<std::string>& inputs = merged_cost_node.GetInputs();
        // 由于之前算子融合时没考虑到前后链接的节点值也要变化, 因此在这里遍历一遍
        for (auto& output : outputs) {
            output = node_to_group[output];
        }
        for (auto& input : inputs) {
            input = node_to_group[input];
        }
        merged_cost_node.SetName(merged_cost_node_pair.first);
        new_merged_cost_nodes.push_back(merged_cost_node);
    }

    MergedCostGraph new_merged_cost_graph;
    new_merged_cost_graph.SetMergedCostNodes(new_merged_cost_nodes);

    return new_merged_cost_graph;
}
}  // namespace framework
