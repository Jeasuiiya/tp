#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "cmath"
#include "cstdio"
#include "graphGroup.h"
#include "iostream"
#include "map"
#include "string"
#include "vector"

#define MATH_DEFINES_DEFINED

namespace framework {

class Builder {
  public:
    Graph graph = Graph();
    std::map<std::string, std::int64_t> op_index;
    std::map<std::string, std::int64_t> dtype_bytes;
    std::map<std::string, std::string> colocation_group;
    std::map<std::string, std::map<std::string, std::int64_t>> in_edge;
    std::map<std::string, std::map<std::string, std::int64_t>> out_edge;

    Builder() = default;

    void ConstructBuilder(Graph& graph_) {
        this->SetDtypeBytes();
        int64_t i = 0;
        std::vector<std::shared_ptr<NodeBase>>& graph_nodes = graph_.Nodes();
        for (auto& node : graph_nodes) {
            this->op_index.insert(std::pair<std::string, std::int64_t>((*node).Name(), i++));
            std::map<std::string, std::string> attr = (*node).Attrs();
            if (attr.count("colocation_group") == 0) {
                attr["colocation_group"] = (*node).Name();
            }
            this->SetColocationGroup((*node).Name(), attr["colocation_group"]);
            this->AddEdges(*node);
            this->graph.AddNode(*node);
        }
        this->DefineOpCost();
    }

    void SetDtypeBytes() {
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("int64_t", 8));
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("int32_t", 4));
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("int16_t", 2));
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("int8_t", 1));
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("float32", 4));
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("float32_ref", 4));
        this->dtype_bytes.insert(std::pair<std::string, std::int64_t>("resource", 0));
    }

    void SetColocationGroup(const std::string& op, const std::string& colo) {
        this->colocation_group.insert(std::pair<std::string, std::string>(op, colo));
    }

    std::string GetOpColocationGroup(const std::string& op) {
        return this->colocation_group[op];
    }

    void AddEdges(NodeBase node) {
        for (auto input : node.Inputs()) {
            // "input" is input tensor's name
            auto view = input | ranges::views::split(':') | ranges::to<std::vector<std::string>>();
            std::string index = "0";
            if (view.size() == 2) {
                index = view[1];
            }
            auto result = node.InputPort(std::stoi(index));
            if (result.has_value()) {
                std::int64_t comm_cost = 1;
                auto inputport = result.value();
                // framework::DataType dtype = inputport.entity.tensor.dtype;
                framework::shape_t shape = inputport.entity.tensor.shape;
                std::string input_node = inputport.entity.source;
                for (auto dimi : shape) {
                    comm_cost *= dimi;
                }
                comm_cost *= 4;
                if (shape.empty()) {
                    comm_cost = 0;
                }
                this->in_edge[node.Name()][input] = comm_cost;
                this->out_edge[input][node.Name()] = comm_cost;
            } else {
                this->in_edge[node.Name()][input] = 0;
                this->out_edge[input][node.Name()] = 0;
            }
        }
    }

    void DefineOpCost(float R = 0.2) {
        std::vector<std::shared_ptr<NodeBase>>& graph_nodes = this->graph.Nodes();
        for (auto& node : graph_nodes) {
            std::string op = (*node).Name();

            std::int64_t input_tensor = 0;
            std::map<std::string, std::int64_t> in_edges = this->in_edge[op];
            for (auto& in_edge : in_edges) {
                input_tensor += in_edge.second;
            }

            std::int64_t output_tensor = 0;
            std::map<std::string, std::int64_t> out_edges = this->out_edge[op];
            for (auto& out_edge : out_edges) {
                output_tensor += out_edge.second;
            }

            (*node).InputMemory(input_tensor);
            (*node).OutputMemory(output_tensor);

            std::int64_t compute_cost = ceil(R
                                             * ceil((input_tensor + output_tensor)
                                                    / (1
                                                       + exp(ceil(-(std::abs(input_tensor - output_tensor))
                                                                  / static_cast<double>(1 + output_tensor))))));
            (*node).ComputeCost(compute_cost);
        }

        for (auto& node : graph_nodes) {
            std::string op = (*node).Name();
            if ((*node).ComputeCost() == 0) {
                std::int64_t compute_cost = 0;
                std::int64_t related_op_num = 0;

                std::map<std::string, std::int64_t> in_edges = this->in_edge[op];
                for (auto& in_edge : in_edges) {
                    compute_cost += in_edge.second;
                    related_op_num += 1;
                }
                std::map<std::string, std::int64_t> out_edges = this->out_edge[op];
                for (auto& out_edge : out_edges) {
                    compute_cost += out_edge.second;
                    related_op_num += 1;
                }

                if (related_op_num != 0) {
                    (*node).ComputeCost(ceil(static_cast<double>(compute_cost) / related_op_num));
                }
            }
        }
    }
};
}  // namespace framework
