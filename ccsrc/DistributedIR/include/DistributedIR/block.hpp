#pragma once

#ifndef FRAMEWORK_IR_BLOCK_H
#define FRAMEWORK_IR_BLOCK_H
#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DistributedIR/graph.hpp"
#include "common/fmt.hpp"
#include "common/util.hpp"
#include "edge.hpp"
namespace framework {

class Block {
    friend struct fmt::formatter<Block>;

  public:
    Block(std::string id, std::string device, SubGraph graph)
        : id(std::move(id)), device(std::move(device)), graph(std::move(graph)) {}
    virtual ~Block() = default;

    void AddInputPort(const EdgePort<std::string>& port) {
        inputs.push_back(port);
    }
    void AddOutputPort(const EdgePort<std::string>& port) {
        outputs.push_back(port);
    }
    DECL_ACCESSOR(Inputs, Inputs, ALL(std::vector<EdgePort<std::string>>), inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, ALL(std::vector<EdgePort<std::string>>), outputs, M)

    bool operator==(const Block& block) const {
        return id == block.id;
    }

  private:
    std::string id;
    std::string device;
    SubGraph graph;
    std::vector<EdgePort<std::string>> inputs;
    std::vector<EdgePort<std::string>> outputs;
};

class DeviceGraph : public HasInternalEdge, public HasEdgePort<Block> {
    friend struct fmt::formatter<DeviceGraph>;

  public:
    explicit DeviceGraph(std::string id) : id(std::move(id)){};
    virtual ~DeviceGraph() = default;
    DECL_ACCESSOR(Inputs, Inputs, std::vector<EdgePort<Block>>, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, std::vector<EdgePort<Block>>, outputs, M)
    void AddBlock(const Block& block) {
        blocks.emplace_back(block);
    }
    void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) override;

    void BuildInputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Inputs().size();
            },
            [](auto e) {
                // input edge port must be a edge end
                return e.end;
            },
            blocks, edges, inputs);
    }
    void BuildOutputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Outputs().size();
            },
            [](auto e) {
                // output edge port must be a edge start
                return e.start;
            },
            blocks, edges, outputs);
    }
    bool operator==(const DeviceGraph& graph) const {
        return id == graph.id;
    }

  private:
    std::string id;
    std::string device;
    std::vector<Block> blocks;
    // internal edge
    std::vector<Edge<Block>> edges;

    // port
    std::vector<EdgePort<Block>> inputs;
    std::vector<EdgePort<Block>> outputs;
};

class ServerGraph : public HasInternalEdge, public HasEdgePort<DeviceGraph> {
    friend struct fmt::formatter<ServerGraph>;

  public:
    explicit ServerGraph(std::string id) : id(std::move(id)){};
    virtual ~ServerGraph() = default;
    DECL_ACCESSOR(Inputs, Inputs, std::vector<EdgePort<DeviceGraph>>, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, std::vector<EdgePort<DeviceGraph>>, outputs, M)

    void AddDeviceGraph(const DeviceGraph& graph) {
        device_graphs.emplace_back(graph);
    }
    void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) override;
    void BuildInputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Inputs().size();
            },
            [](auto e) {
                // input edge port must be a edge end
                return e.end;
            },
            device_graphs, edges, inputs);
    }
    void BuildOutputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Outputs().size();
            },
            [](auto e) {
                return
                    // output edge port must be a edge start
                    e.start;
            },
            device_graphs, edges, outputs);
    }

    bool operator==(const ServerGraph& graph) const {
        return id == graph.id;
    }

  private:
    std::string id;
    std::string server;
    std::vector<DeviceGraph> device_graphs;
    std::vector<Edge<DeviceGraph>> edges;
    // port
    std::vector<EdgePort<DeviceGraph>> inputs;
    std::vector<EdgePort<DeviceGraph>> outputs;
};

class ClusterGraph : public HasInternalEdge, public HasEdgePort<ServerGraph> {
    friend struct fmt::formatter<ClusterGraph>;

  public:
    explicit ClusterGraph(std::string id) : id(std::move(id)){};
    virtual ~ClusterGraph() = default;
    void AddServerGraph(const ServerGraph& graph) {
        server_graphs.emplace_back(graph);
    }
    void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) override;
    void BuildInputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Inputs().size();
            },
            [](auto e) {
                return
                    // input edge port must be a edge end
                    e.end;
            },
            server_graphs, edges, inputs);
    }
    void BuildOutputPorts() override {
        BuildSeclectedPorts(
            [](auto b) {
                // search outputs
                return b.Outputs().size();
            },
            [](auto e) {
                // output edge port must be a edge start
                return e.start;
            },
            server_graphs, edges, outputs);
    }

    bool operator==(const ClusterGraph& graph) const {
        return id == graph.id;
    }

  private:
    std::string id;
    std::vector<ServerGraph> server_graphs;
    std::vector<Edge<ServerGraph>> edges;
    // port
    std::vector<EdgePort<ServerGraph>> inputs;
    std::vector<EdgePort<ServerGraph>> outputs;
};

}  // namespace framework
// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<framework::Block> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const framework::Block& b, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "Block(id={})", b.id);
        }
        return fmt::format_to(ctx.out(), "Block(id={}, device={}, graph={}, inputs={}, outputs={})", b.id, b.device,
                              b.graph, b.inputs, b.outputs);
    }
};

template <>
struct fmt::formatter<framework::DeviceGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const framework::DeviceGraph& dg, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "DeviceGraph(id={})", dg.id);
        }
        return fmt::format_to(ctx.out(), "DeviceGraph(id={}, device={}, blocks={}, edges={}, inputs={}, outputs={})",
                              dg.id, dg.device, dg.blocks, dg.edges, dg.inputs, dg.outputs);
    }
};

template <>
struct fmt::formatter<framework::ServerGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const framework::ServerGraph& sg, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "ServerGraph(id={})", sg.id);
        }
        return fmt::format_to(ctx.out(), "ServerGraph(id={}, device={}, graphs={}, edges={}, inputs={}, outputs={})",
                              sg.id, sg.server, sg.device_graphs, sg.edges, sg.inputs, sg.outputs);
    }
};
template <>
struct fmt::formatter<framework::ClusterGraph> : public fmt::formatter<ShortFormat> {
    template <typename FormatContext>
    auto format(const framework::ClusterGraph& cg, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (presentation == 's') {
            return fmt::format_to(ctx.out(), "ClusterGraph(id={})", cg.id);
        }
        return fmt::format_to(ctx.out(), "ClusterGraph(id={}, graphs={}, edges={}, inputs={}, outputs={})", cg.id,
                              cg.server_graphs, cg.edges, cg.inputs, cg.outputs);
    }
};
// NOLINTEND(readability-identifier-naming)
#endif /* end of include guard: FRAMEWORK_IR_BLOCK_H */
