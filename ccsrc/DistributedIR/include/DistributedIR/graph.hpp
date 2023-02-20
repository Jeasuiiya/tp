#ifndef FRAMEWORK_GRAPH_GRAPH_H
#define FRAMEWORK_GRAPH_GRAPH_H

#include <numeric>
#include <utility>
#include <vector>

#include "common/util.hpp"
#include "node.hpp"
namespace framework {

class Graph {
   private:
    std::vector<NodeBase> nodes;
    std::map<std::string, NodeBase&> node_map;

   public:
    Graph() = default;
    virtual ~Graph() = default;
    DECL_ACCESSOR(Nodes, Nodes, std::vector<NodeBase>, nodes, true)
    DECL_ACCESSOR(NodeMap, NodeMap, ALL(std::map<std::string, NodeBase&>), node_map, true)
    void AddNode(NodeBase node) {
        node_map.insert(std::pair<std::string, NodeBase&>(node.Name(), node));
        nodes.push_back(node);
    }
    void AddNode(int at, NodeBase node) {
        node_map.insert(std::pair<std::string, NodeBase&>(node.Name(), node));
        nodes.insert(nodes.begin() + at, node);
    }

    NodeBase& GetNode(int at) { return nodes.at(at); }
    NodeBase& GetNode(const std::string& name) { return node_map.find(name)->second; }
    std::string ToString() {
        // return "";
        return std::accumulate(nodes.begin(), nodes.end(), std::string(),
                               [](const std::string& s, NodeBase& p) { return s + "\n" + p.ToString(); });
    }
};

class SubGraph : public Graph {
    std::vector<SubGraph*> input_graphs;                      // 输入图
    std::vector<std::map<std::string, std::string>> inputs;   // 各图输入
    std::vector<SubGraph*> output_graphs;                     // 输出图
    std::vector<std::map<std::string, std::string>> outputs;  // 输出
};

}  // namespace framework

#endif
