#ifndef _FRAMEWORK_GRAPH_GRAPH_H
#define _FRAMEWORK_GRAPH_GRAPH_H

#include <numeric>
#include <utility>
#include <vector>

#include "node.hpp"
#include "util.hpp"
namespace framework {

class Graph {
   private:
    std::vector<NodeBase> nodes;
    std::map<std::string, NodeBase&> node_map;

   public:
    Graph() {}
    virtual ~Graph() {}
    GEN_ACCESSOR_IN_DEC(std::vector<NodeBase>, nodes)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, NodeBase&>), node_map)
    void add_node(NodeBase node) {
        node_map.insert(
            std::pair<std::string, NodeBase&>(node.get_name(), node));
        nodes.push_back(node);
    }
    void add_node(int at, NodeBase node) {
        node_map.insert(
            std::pair<std::string, NodeBase&>(node.get_name(), node));
        nodes.insert(nodes.begin() + at, node);
    }

    NodeBase& get_node(int at) { return nodes.at(at); }
    NodeBase& get_node(std::string name) { return node_map.find(name)->second; }
    std::string to_string() {
        // return "";
        return std::accumulate(nodes.begin(), nodes.end(), std::string(),
                               [](std::string s, NodeBase& p) {
                                   return s + "\n" + p.to_string();
                               });
    }
};

class SubGraph : public Graph {
    std::vector<Graph> input_graphs;                          //输入图
    std::vector<std::map<std::string, std::string>> inputs;   //各图输入
    std::vector<Graph> output_graphs;                         //输出图
    std::vector<std::map<std::string, std::string>> outputs;  //输出
};

}  // namespace framework

#endif /* ifndef _GRAPH_GRAPH_H \*/
