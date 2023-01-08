#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <DistributedIR/graph.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace framework::py {
class Node {
   private:
    framework::NodeBase* node_ptr;

   public:
    Node(framework::NodeBase* node_ptr) { node_ptr = node_ptr; }
    Node(Node&& node) { node_ptr = node.node_ptr; }
    Node(Node& node) { node_ptr = node.node_ptr; }
    Node() { node_ptr = new framework::NodeBase(); }
    ~Node() { delete node_ptr; }
    framework::NodeBase* NodePtr() { return this->node_ptr; }
    GEN_PROXY_ACCESSOR(std::string, node_ptr, name)
    GEN_PROXY_ACCESSOR(std::string, node_ptr, op)
    GEN_PROXY_ACCESSOR(std::vector<std::string>, node_ptr, inputs)
    GEN_PROXY_ACCESSOR(std::vector<std::string>, node_ptr, outputs)
    GEN_PROXY_ACCESSOR(ALL(std::map<std::string, std::string>), node_ptr, attrs)
    GEN_PROXY_ACCESSOR(long, node_ptr, start_time)
    GEN_PROXY_ACCESSOR(long, node_ptr, end_time)
    GEN_PROXY_ACCESSOR(long, node_ptr, compute_cost)
    GEN_PROXY_ACCESSOR(long, node_ptr, temporary_memory)
    GEN_PROXY_ACCESSOR(long, node_ptr, persistent_memory)
    GEN_PROXY_ACCESSOR(long, node_ptr, input_memory)
    GEN_PROXY_ACCESSOR(long, node_ptr, output_memory)
    void add_input(std::string input) { node_ptr->add_input(input); }
    void add_output(std::string output) { node_ptr->add_output(output); }
    std::string to_string() { return node_ptr->to_string(); }
};
class Graph {
   private:
    framework::Graph* graph_ptr;

   public:
    Graph(framework::Graph* graph) { graph_ptr = graph; }
    Graph(Graph&& graph) { graph_ptr = graph.graph_ptr; }
    Graph() { graph_ptr = new framework::Graph(); }
    ~Graph() { delete graph_ptr; }
    framework::Graph* GraphPtr() { return graph_ptr; }
    void add_node(Node& node) {
        framework::NodeBase node_base(node.NodePtr());
        graph_ptr->get_node_map().insert(
            std::pair<std::string, NodeBase&>(node_base.get_name(), node_base));
        graph_ptr->get_nodes().push_back(node_base);
    }
    void add_node(int at, Node& node) {
        framework::NodeBase node_base(node.NodePtr());
        graph_ptr->get_node_map().insert(
            std::pair<std::string, NodeBase&>(node_base.get_name(), node_base));
        graph_ptr->get_nodes().insert(graph_ptr->get_nodes().begin() + at,
                                      node_base);
    }

    Node get_node(int at) { return Node(&graph_ptr->get_nodes().at(at)); }
    Node get_node(std::string name) {
        return Node(&graph_ptr->get_node_map().find(name)->second);
    }
    std::string to_string() { return graph_ptr->to_string(); }
};
};  // namespace framework::py

namespace py = pybind11;
using PyNode = framework::py::Node;
using PyGraph = framework::py::Graph;
PYBIND11_MODULE(PYBIND11_CURRENT_MODULE_NAME, m) {
    m.doc() = R"pbdoc(
        python graph
        -----------------------
        .. currentmodule:: _graph
    )pbdoc";

    py::class_<PyNode>(m, "Node")
        .def(py::init())
        .def(py::init([](std::string name, std::string op) {
            PyNode* n = new PyNode;
            n->set_name(name);
            n->set_op(op);
            return n;
        }))
        .def_property("name", &PyNode::get_name, &PyNode::set_name)
        .def_property("op", &PyNode::get_op, &PyNode::set_op)
        .def_property("inputs", &PyNode::get_inputs, &PyNode::set_inputs)
        .def_property("outputs", &PyNode::get_outputs, &PyNode::set_outputs)
        .def_property("attrs", &PyNode::get_attrs, &PyNode::set_attrs)
        .def_property("start_time", &PyNode::get_start_time,
                      &PyNode::set_start_time)
        .def_property("end_time", &PyNode::get_end_time, &PyNode::set_end_time)
        .def_property("compute_cost", &PyNode::get_compute_cost,
                      &PyNode::set_compute_cost)
        .def_property("temporary_memory", &PyNode::get_temporary_memory,
                      &PyNode::set_temporary_memory)
        .def_property("persistent_memory", &PyNode::get_persistent_memory,
                      &PyNode::set_persistent_memory)
        .def_property("input_memory", &PyNode::get_input_memory,
                      &PyNode::set_input_memory)
        .def_property("output_memory", &PyNode::get_output_memory,
                      &PyNode::set_output_memory)
        .def("add_input", &PyNode::add_input)
        .def("add_output", &PyNode::add_output)
        .def("__repr__", &PyNode::to_string)
        .def("__str__", &PyNode::to_string);
    py::class_<PyGraph>(m, "Graph")
        .def(py::init())
        .def("add_node", py::overload_cast<PyNode&>(&PyGraph::add_node))
        .def("add_node", py::overload_cast<int, PyNode&>(&PyGraph::add_node))
        .def("get_node", py::overload_cast<int>(&PyGraph::get_node))
        .def("get_node", py::overload_cast<std::string>(&PyGraph::get_node))
        .def("__repr__", &PyGraph::to_string)
        .def("__str__", &PyGraph::to_string);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}