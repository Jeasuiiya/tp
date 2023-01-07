#ifndef _FRAMEWORK_GRAPH_NODE_H
#define _FRAMEWORK_GRAPH_NODE_H

#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "util.hpp"
namespace framework {

class NodeBase {
   private:
    std::string name;                          //节点名
    std::string op;                            //算子名
    std::vector<std::string> inputs;           // 节点输入
    std::string device;                        //该节点的计算设备
    std::map<std::string, std::string> attrs;  //节点属性
    long start_time;                           //开始时间
    long end_time;                             //结束时间
    long compute_cost;                         //计算代价
    long temporary_memory;                     //临时内存
    long persistent_memory;                    //持久内存
    long input_memory;                         //输入内存
    long output_memory;                        //输出内存

    // T data;
   public:
    NodeBase() {}
    virtual ~NodeBase() {}
    // GEN_ACCESSOR_IN_DEC(std::string, name)
    // GEN_ACCESSOR_IN_DEC(std::string, op)
    // GEN_ACCESSOR_IN_DEC(std::string, device)
    // GEN_ACCESSOR_IN_DEC(std::vector<std::string>, inputs)
    // GEN_ACCESSOR_IN_DEC(TYPE(std::map<std::string, std::string>), attrs)
    // // GEN_ACCESSOR_IN_DEC(T, data)
    // void add_input(std::string input) { inputs.push_back(input); }
    // std::string to_string() {
    //   std::stringstream ss;
    //   ss << "name:" << name << std::endl;
    //   ss << "op:" << op << std::endl;
    //   ss << "inputs: "
    //      << std::accumulate(inputs.begin(), inputs.end(), std::string(),
    //                         [](const std::string& s, const std::string& p) {
    //                           return s + (s.empty() ? std::string() : ", ") +
    //                           p;
    //                         })
    //      << std::endl;
    //   ss << "device: " << device << std::endl;
    //   ss << "attrs: "
    //      << std::accumulate(
    //             attrs.begin(), attrs.end(), std::string(),
    //             [](const std::string& s,
    //                const std::pair<const std::string, std::string>& p) {
    //               return s + (s.empty() ? std::string() : "\n") + p.first +
    //               ": " +
    //                      p.second;
    //             })
    //      << std::endl;
    //   return ss.str();
    // }
};

class MergedNode : public NodeBase {
    std::vector<NodeBase> merged_nodes;  //已合并节点
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