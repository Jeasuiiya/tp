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
#include "common/util.hpp"
namespace framework {

inline namespace edge {
template <typename T>
struct Edge;

template <typename T>
struct EdgePort {
    using TT = typename std::remove_reference<T>::type;
    TT* entity;
    int index;
    Edge<TT>* edge = nullptr;
    EdgePort(TT* entity, int index) : entity(entity), index(index) {}
    bool operator==(const EdgePort<T> port) const {
        return *entity == *port.entity && this->index == port.index;
    }

    EdgePort<TT> operator|(const Edge<TT>& edge) {
        this->edge = &edge;
        return *this;
    }

    EdgePort<TT> operator|(Edge<TT>* edge) {
        this->edge = edge;
        return *this;
    }

    EdgePort<TT> operator>>(Edge<TT>& edge) {
        edge.start = *this;
        return *this;
    }

    EdgePort<TT> operator<<(Edge<TT>& edge) {
        edge.end = *this;
        return *this;
    }

    EdgePort<TT> operator>>(Edge<TT>* edge) {
        edge->start = *this;
        return *this;
    }

    EdgePort<TT> operator<<(Edge<TT>* edge) {
        edge->end = *this;
        return *this;
    }
};

template <>
struct EdgePort<std::string> {
    std::string entity;
    explicit EdgePort(std::string entity) : entity(std::move(entity)) {}
    bool operator==(const EdgePort<std::string>& port) const {
        return this->entity == port.entity;
    }
};

template <typename T>
struct Edge {
    using TT = typename std::remove_reference<T>::type;
    Edge(TT* start, int start_index, TT* end, int end_index)
        : start(EdgePort<TT>(start, start_index)), end(EdgePort<TT>(end, end_index)) {
        this->start | this;
        this->end | this;
    }
    Edge(EdgePort<TT> start, EdgePort<TT> end) : start(start), end(end) {
        this->start | this;
        this->end | this;
    }
    EdgePort<TT> start;
    EdgePort<TT> end;
};

struct HasInternalEdge {
    virtual void Connect(int start_item, int start_out_index, int end_item, int end_arg_index) = 0;
};

template <typename T>
struct HasEdgePort {
    virtual void BuildInputPorts() = 0;
    virtual void BuildOutputPorts() = 0;
    void BuildSeclectedPorts(const std::function<uint64_t(const T&)>& searchPortSize,
                             const std::function<EdgePort<T>(Edge<T>&)>& getEdgePort, std::vector<T>& internelEles,
                             std::vector<Edge<T>>& edges, std::vector<EdgePort<T>>& out) {
        std::vector<EdgePort<T>> result;
        for (auto ele : internelEles) {
            for (size_t i = 0; i < searchPortSize(ele); i++) {
                EdgePort<T> ep(&ele, i);
                bool internal_connected = false;
                for (auto e : edges) {
                    // output edge port must be a edge start
                    if (getEdgePort(e) == ep) {
                        internal_connected = true;
                        break;
                    }
                }
                if (!internal_connected) {
                    result.push_back(ep);
                }
            }
        }
        out = result;
    }
    virtual void BuildPorts() {
        BuildInputPorts();
        BuildOutputPorts();
    }
};

}  // namespace edge

class Block {
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
    std::vector<Block> blocks;
    // internal edge
    std::vector<Edge<Block>> edges;

    // port
    std::vector<EdgePort<Block>> inputs;
    std::vector<EdgePort<Block>> outputs;
};

class ServerGraph : public HasInternalEdge, public HasEdgePort<DeviceGraph> {
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
    std::vector<DeviceGraph> device_graphs;
    std::vector<Edge<DeviceGraph>> edges;
    // port
    std::vector<EdgePort<DeviceGraph>> inputs;
    std::vector<EdgePort<DeviceGraph>> outputs;
};

class ClusterGraph : public HasInternalEdge, public HasEdgePort<ServerGraph> {
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
#endif /* end of include guard: FRAMEWORK_IR_BLOCK_H */
