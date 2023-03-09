#ifndef FRAMEWORK_GRAPH_GRAPH_H
#define FRAMEWORK_GRAPH_GRAPH_H

#include <memory>
#include <vector>

#include "common/util.hpp"
#include "node.hpp"
namespace framework {

class Graph {
  private:
    std::vector<std::shared_ptr<NodeBase>> nodes;
    std::map<std::string, std::shared_ptr<NodeBase>> node_map;

  public:
    Graph() = default;
    virtual ~Graph() = default;
    Graph(const Graph& g) {
        for (const auto& i : g.nodes) {
            auto ptr = std::make_shared<NodeBase>(*i);
            nodes.push_back(ptr);
            node_map.insert({ptr->Name(), ptr});
        }
    }
    Graph(Graph&& g) noexcept : nodes(std::move(g.nodes)), node_map(std::move(g.node_map)) {}
    DECL_ACCESSOR(Nodes, Nodes, std::vector<std::shared_ptr<NodeBase>>, nodes, M)
    DECL_ACCESSOR(NodeMap, NodeMap, ALL(std::map<std::string, std::shared_ptr<NodeBase>>), node_map, M)
    void AddNode(NodeBase node) {
        auto n = std::make_shared<NodeBase>(node);
        node_map.insert({node.Name(), n});
        nodes.push_back(n);
    }
    void AddNode(int at, NodeBase node) {
        auto n = std::make_shared<NodeBase>(node);
        node_map.insert({node.Name(), n});
        nodes.insert(nodes.begin() + at, n);
    }

    void AddNode(const std::shared_ptr<NodeBase>& node) {
        node_map.insert({node->Name(), node});
        nodes.push_back(node);
    }
    void AddNode(int at, const std::shared_ptr<NodeBase>& node) {
        node_map.insert({node->Name(), node});
        nodes.insert(nodes.begin() + at, node);
    }

    std::shared_ptr<NodeBase>& GetNode(int at) {
        return nodes.at(at);
    }
    std::shared_ptr<NodeBase>& GetNode(const std::string& name) {
        return node_map.find(name)->second;
    }

    std::string ToString() {
        // return "";
        return std::accumulate(
            nodes.begin(), nodes.end(), std::string(),
            [](const std::string& s, std::shared_ptr<NodeBase>& p) { return s + "\n" + p->ToString(); });
    }
};

class SubGraph : public Graph {
    std::vector<std::shared_ptr<SubGraph>> input_graphs;           // 输入图
    std::vector<std::multimap<std::string, std::string>> inputs;   // 各图输入
    std::vector<std::shared_ptr<SubGraph>> output_graphs;          // 输出图
    std::vector<std::multimap<std::string, std::string>> outputs;  // 输出

  public:
    void AddInputGraph(const std::shared_ptr<SubGraph>& g) {
        input_graphs.push_back(g);
    }
    void AddInputGraph(const SubGraph& g) {
        input_graphs.push_back(std::make_shared<SubGraph>(g));
    }
    void AddOutputGraph(const std::shared_ptr<SubGraph>& g) {
        output_graphs.push_back(g);
    }
    void AddOutputGraph(const SubGraph& g) {
        output_graphs.push_back(std::make_shared<SubGraph>(g));
    }

    void AddInput(const std::multimap<std::string, std::string>& op_op) {
        inputs.push_back(op_op);
    }
    void AddOutput(const std::multimap<std::string, std::string>& op_op) {
        outputs.push_back(op_op);
    }
    DECL_GETTER(GetInputs, ALL(std::vector<std::multimap<std::string, std::string>>), inputs)
    DECL_GETTER(GetOutputs, ALL(std::vector<std::multimap<std::string, std::string>>), outputs)
};

}  // namespace framework

#endif
