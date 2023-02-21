#ifndef FRAMEWORK_GRAPH_GRAPH_H
#define FRAMEWORK_GRAPH_GRAPH_H

#include <memory>
#include <numeric>
#include <utility>
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
    DECL_ACCESSOR(Nodes, Nodes, std::vector<std::shared_ptr<NodeBase>>, nodes, true)
    DECL_ACCESSOR(NodeMap, NodeMap, ALL(std::map<std::string, std::shared_ptr<NodeBase>>), node_map, true)
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
    std::vector<std::shared_ptr<SubGraph>> input_graphs;      // 输入图
    std::vector<std::map<std::string, std::string>> inputs;   // 各图输入
    std::vector<std::shared_ptr<SubGraph>> output_graphs;     // 输出图
    std::vector<std::map<std::string, std::string>> outputs;  // 输出

    void AddInputGraph(const std::shared_ptr<SubGraph>& g) {
        input_graphs.push_back(g);
    }
    void AddInputGraph(SubGraph g) {
        input_graphs.push_back(std::make_shared<SubGraph>(g));
    }
    void AddOutputGraph(const std::shared_ptr<SubGraph>& g) {
        output_graphs.push_back(g);
    }
    void AddOutputGraph(SubGraph g) {
        output_graphs.push_back(std::make_shared<SubGraph>(g));
    }
};

}  // namespace framework

#endif
