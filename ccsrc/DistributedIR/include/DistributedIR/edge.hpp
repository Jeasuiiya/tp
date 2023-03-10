#pragma once

#ifndef FRAMEWORK_IR_EDGE_H
#define FRAMEWORK_IR_EDGE_H
#include <functional>
#include <string>

#include "fmt/format.h"
namespace framework {
template <typename T>
struct Edge;
template <typename T>
struct EdgePort {
    friend struct fmt::formatter<EdgePort<T>>;
    using TT = std::remove_reference_t<T>;
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
    friend struct fmt::formatter<EdgePort<std::string>>;
    std::string entity;
    explicit EdgePort(std::string entity) : entity(std::move(entity)) {}
    bool operator==(const EdgePort<std::string>& port) const {
        return this->entity == port.entity;
    }
};

template <typename T>
struct Edge {
    friend struct fmt::formatter<Edge<T>>;
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
        for (auto& ele : internelEles) {
            for (size_t i = 0; i < searchPortSize(ele); i++) {
                EdgePort<T> ep(&ele, i);
                bool internal_connected = false;
                for (auto& e : edges) {
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
}  // namespace framework

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
struct fmt::formatter<framework::EdgePort<T>> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::EdgePort<T>& ep, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (ep.entity == nullptr) {
            return fmt::format_to(ctx.out(), "EdgePort(entity=null, index={})", ep.index);
        }
        return fmt::format_to(ctx.out(), "EdgePort(entity={:s}, index={})", *ep.entity, ep.index);
    }
};

template <>
struct fmt::formatter<framework::EdgePort<std::string>> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::EdgePort<std::string>& ep, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "EdgePort(entity={})", ep.entity);
    }
};

template <typename T>
struct fmt::formatter<framework::Edge<T>> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::Edge<T>& e, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Edge(start={}, end={})", e.start, e.end);
    }
};
// NOLINTEND(readability-identifier-naming)
#endif
