#ifndef _FRAMEWORK_GRAPH_NODE_H
#define _FRAMEWORK_GRAPH_NODE_H

#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/util.hpp"
namespace framework {

class NodeBase {
   private:
    std::string name;                 // 节点名
    std::string op;                   // 算子名
    std::vector<std::string> inputs;  // 节点输入
    std::vector<std::string> outputs;
    std::string device;                        // 该节点的计算设备
    std::map<std::string, std::string> attrs;  // 节点属性
    long start_time;                           // 开始时间
    long end_time;                             // 结束时间
    long compute_cost;                         // 计算代价
    long temporary_memory;                     // 临时内存
    long persistent_memory;                    // 持久内存
    long input_memory;                         // 输入内存
    long output_memory;                        // 输出内存

    // T data;
   public:
    NodeBase() {}
    NodeBase(NodeBase* node)
        : name(node->name),
          op(node->op),
          inputs(node->inputs),
          outputs(node->outputs),
          device(node->device),
          attrs(node->attrs),
          start_time(node->start_time),
          end_time(node->end_time),
          compute_cost(node->compute_cost),
          temporary_memory(node->temporary_memory),
          persistent_memory(node->persistent_memory),
          input_memory(node->input_memory),
          output_memory(node->output_memory) {}
    virtual ~NodeBase() {}
    GEN_ACCESSOR_IN_DEC(std::string, name)
    GEN_ACCESSOR_IN_DEC(std::string, op)
    GEN_ACCESSOR_IN_DEC(std::string, device)
    GEN_ACCESSOR_IN_DEC(std::vector<std::string>, inputs)
    GEN_ACCESSOR_IN_DEC(std::vector<std::string>, outputs)
    GEN_ACCESSOR_IN_DEC(ALL(std::map<std::string, std::string>), attrs)
    GEN_ACCESSOR_IN_DEC(long, start_time)
    GEN_ACCESSOR_IN_DEC(long, end_time)
    GEN_ACCESSOR_IN_DEC(long, compute_cost)
    GEN_ACCESSOR_IN_DEC(long, temporary_memory)
    GEN_ACCESSOR_IN_DEC(long, persistent_memory)
    GEN_ACCESSOR_IN_DEC(long, input_memory)
    GEN_ACCESSOR_IN_DEC(long, output_memory)
    // // GEN_ACCESSOR_IN_DEC(T, data)
    void add_input(std::string input) { inputs.push_back(input); }
    void add_output(std::string output) { outputs.push_back(output); }
    std::string to_string() {
        std::stringstream ss;
        ss << "name:" << name << std::endl;
        ss << "op:" << op << std::endl;
        ss << "inputs: "
           << std::accumulate(inputs.begin(), inputs.end(), std::string(),
                              [](const std::string& s, const std::string& p) {
                                  return s +
                                         (s.empty() ? std::string() : ", ") + p;
                              })
           << std::endl;
        ss << "device: " << device << std::endl;
        ss << "attrs: "
           << std::accumulate(
                  attrs.begin(), attrs.end(), std::string(),
                  [](const std::string& s,
                     const std::pair<const std::string, std::string>& p) {
                      return s + (s.empty() ? std::string() : "\n") + p.first +
                             ": " + p.second;
                  })
           << std::endl;
        return ss.str();
    }
};

class MergedNode : public NodeBase {
    std::vector<NodeBase> merged_nodes;  // 已合并节点
};

// template <typename T>
// class Node : public NodeBase {
//   using NodeBase::NodeBase;

//  public:
//   T data;
//   GEN_ACCESSOR_IN_DEC(T, data)
// };
// template <>
// class Node<void> : public NodeBase {
//   using NodeBase::NodeBase;
// };
// GEN_SETTER(Node, std::string, name)
}  // namespace framework

#endif /* ifndef _GRAPH_NODE_H */