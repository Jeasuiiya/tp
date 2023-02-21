#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <DistributedIR/graph.hpp>
#include <memory>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace framework::py {
class Node {
   private:
    std::shared_ptr<framework::NodeBase> node_ptr;

   public:
    explicit Node(framework::NodeBase node) { node_ptr = std::make_shared<framework::NodeBase>(node); }
    explicit Node(std::shared_ptr<framework::NodeBase> node) { node_ptr = std::move(node); }
    Node(Node&& node) noexcept { node_ptr = node.node_ptr; }
    Node(Node& node) { node_ptr = node.node_ptr; }
    Node() { node_ptr = std::make_shared<framework::NodeBase>(); }
    ~Node() = default;
    std::shared_ptr<framework::NodeBase>& NodePtr() { return this->node_ptr; }
    DECL_ACCESSOR_PROXY_S(SetName, GetName, std::string, node_ptr, Name)
    DECL_ACCESSOR_PROXY_S(SetOp, GetOp, std::string, node_ptr, Op)
    DECL_ACCESSOR_PROXY_S(SetInputs, GetInputs, std::vector<std::string>, node_ptr, Inputs)
    DECL_ACCESSOR_PROXY_S(SetOutputs, GetOutputs, std::vector<std::string>, node_ptr, Outputs)
    DECL_ACCESSOR_PROXY_S(SetAttrs, GetAttrs, ALL(std::map<std::string, std::string>), node_ptr, Attrs)
    DECL_ACCESSOR_PROXY_S(SetStartTime, GetStartTime, int64_t, node_ptr, StartTime)
    DECL_ACCESSOR_PROXY_S(SetEndTime, GetEndTime, int64_t, node_ptr, EndTime)
    DECL_ACCESSOR_PROXY_S(SetComputeCost, GetComputeCost, int64_t, node_ptr, ComputeCost)
    DECL_ACCESSOR_PROXY_S(SetTemporaryMemory, GetTemporaryMemory, int64_t, node_ptr, TemporaryMemory)
    DECL_ACCESSOR_PROXY_S(SetPersistentMemory, GetPersistentMemory, int64_t, node_ptr, PersistentMemory)
    DECL_ACCESSOR_PROXY_S(SetInputMemory, GetInputMemory, int64_t, node_ptr, InputMemory)
    DECL_ACCESSOR_PROXY_S(SetOutputMemory, GetOutputMemory, int64_t, node_ptr, OutputMemory)
    void AddInput(const std::string& input) { node_ptr->AddInput(std::move(input)); }
    void AddOutput(const std::string& output) { node_ptr->AddOutput(std::move(output)); }
    std::string ToString() { return node_ptr->ToString(); }
};
class Graph {
   private:
    framework::Graph* graph_ptr;

   public:
    explicit Graph(framework::Graph* graph) { graph_ptr = graph; }
    Graph(Graph&& graph) noexcept { graph_ptr = graph.graph_ptr; }
    Graph() { graph_ptr = new framework::Graph(); }
    ~Graph() { delete graph_ptr; }
    framework::Graph* GraphPtr() { return graph_ptr; }
    void AddNode(Node& node) { graph_ptr->AddNode(node.NodePtr()); }
    void AddNode(int at, Node& node) { graph_ptr->AddNode(at, node.NodePtr()); }

    Node GetNode(int at) { return Node(graph_ptr->GetNode(at)); }
    Node GetNode(const std::string& name) { return Node(graph_ptr->GetNode(name)); }
    std::string ToString() { return graph_ptr->ToString(); }
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
            auto n = std::make_unique<PyNode>();
            n->SetName(std::move(name));
            n->SetOp(std::move(op));
            return n;
        }))
        .def_property("name", &PyNode::GetName, &PyNode::SetName)
        // .def_property("name", &PyNode::Name, &PyNode::Name)
        .def_property("op", &PyNode::GetOp, &PyNode::SetOp)
        .def_property("inputs", &PyNode::GetInputs, &PyNode::SetInputs)
        .def_property("outputs", &PyNode::GetOutputs, &PyNode::SetOutputs)
        .def_property("attrs", &PyNode::GetAttrs, &PyNode::SetAttrs)
        .def_property("start_time", &PyNode::GetStartTime, &PyNode::SetStartTime)
        .def_property("end_time", &PyNode::GetEndTime, &PyNode::SetEndTime)
        .def_property("compute_cost", &PyNode::GetComputeCost, &PyNode::SetComputeCost)
        .def_property("temporary_memory", &PyNode::GetTemporaryMemory, &PyNode::SetTemporaryMemory)
        .def_property("persistent_memory", &PyNode::GetPersistentMemory, &PyNode::SetPersistentMemory)
        .def_property("input_memory", &PyNode::GetInputMemory, &PyNode::SetInputMemory)
        .def_property("output_memory", &PyNode::GetOutputMemory, &PyNode::SetOutputMemory)
        .def("add_input", &PyNode::AddInput)
        .def("add_output", &PyNode::AddOutput)
        .def("__repr__", &PyNode::ToString)
        .def("__str__", &PyNode::ToString);
    py::class_<PyGraph>(m, "Graph")
        .def(py::init())
        .def("add_node", py::overload_cast<PyNode&>(&PyGraph::AddNode))
        .def("add_node", py::overload_cast<int, PyNode&>(&PyGraph::AddNode))
        .def("get_node", py::overload_cast<int>(&PyGraph::GetNode))
        .def("get_node", py::overload_cast<const std::string&>(&PyGraph::GetNode))
        .def("__repr__", &PyGraph::ToString)
        .def("__str__", &PyGraph::ToString);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
