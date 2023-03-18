#ifndef FRAMEWORK_GRAPH_GRAPH_H
#define FRAMEWORK_GRAPH_GRAPH_H

#include <memory>
#include <vector>

#include "common/fmt.hpp"
#include "common/util.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "node.hpp"
namespace framework {
class SubGraph;
class Graph {
    friend struct fmt::formatter<Graph>;
    friend struct fmt::formatter<SubGraph>;
    friend struct fmt::formatter<std::shared_ptr<SubGraph>>;

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
};

class SubGraph : public Graph {
    friend struct fmt::formatter<SubGraph>;
    friend struct fmt::formatter<std::shared_ptr<SubGraph>>;

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

// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<framework::Graph> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::Graph& g, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Graph(nodes={})", g.nodes);
    }
};

template <>
struct fmt::formatter<framework::SubGraph> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::SubGraph& g, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Graph(nodes={}, input_graphs={}, inputs={}, output_graphs={}, outputs={})",
                              g.nodes, g.input_graphs, g.inputs, g.output_graphs, g.outputs);
    }
};
// NOLINTEND(readability-identifier-naming)

#endif
