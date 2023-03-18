#ifndef FRAMEWORK_GRAPH_NODE_H
#define FRAMEWORK_GRAPH_NODE_H

#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/util.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
namespace framework {

class NodeBase {
    friend struct fmt::formatter<NodeBase>;
    friend struct fmt::formatter<std::shared_ptr<NodeBase>>;

  private:
    std::string name;                 // 节点名
    std::string op;                   // 算子名
    std::vector<std::string> inputs;  // 节点输入
    std::vector<std::string> outputs;
    std::vector<std::string> inputs_data;      // 输入节点名:输出index
    std::vector<std::string> outputs_data;     // 当前节点的输出   当前节点名:输出index
    std::string device;                        // 该节点的计算设备
    std::map<std::string, std::string> attrs;  // 节点属性
    int64_t outputs_num;                       // 输出个数
    int64_t start_time;                        // 开始时间
    int64_t end_time;                          // 结束时间
    int64_t compute_cost;                      // 计算代价
    int64_t temporary_memory;                  // 临时内存
    int64_t persistent_memory;                 // 持久内存
    int64_t input_memory;                      // 输入内存
    int64_t output_memory;                     // 输出内存

    // T data;
  public:
    NodeBase() = default;
    NodeBase(const NodeBase& n) = default;
    explicit NodeBase(NodeBase* node)
        : name(node->name),
          op(node->op),
          inputs(node->inputs),
          outputs(node->outputs),
          inputs_data(node->inputs_data),
          outputs_data(node->outputs_data),
          device(node->device),
          attrs(node->attrs),
          outputs_num(node->outputs_num),
          start_time(node->start_time),
          end_time(node->end_time),
          compute_cost(node->compute_cost),
          temporary_memory(node->temporary_memory),
          persistent_memory(node->persistent_memory),
          input_memory(node->input_memory),
          output_memory(node->output_memory) {}
    virtual ~NodeBase() = default;
    DECL_ACCESSOR(Name, Name, std::string, name, M)
    DECL_ACCESSOR(Op, Op, std::string, op, M)
    DECL_ACCESSOR(Device, Device, std::string, device, M)
    DECL_ACCESSOR(Inputs, Inputs, std::vector<std::string>, inputs, M)
    DECL_ACCESSOR(Outputs, Outputs, std::vector<std::string>, outputs, M)
    DECL_ACCESSOR(InputsData, InputsData, std::vector<std::string>, inputs_data, M)
    DECL_ACCESSOR(OutputsData, OutputsData, std::vector<std::string>, outputs_data, M)
    DECL_ACCESSOR(OutputsNum, OutputsNum, int64_t, outputs_num, M)
    DECL_ACCESSOR(Attrs, Attrs, ALL(std::map<std::string, std::string>), attrs, M)
    DECL_ACCESSOR(StartTime, StartTime, int64_t, start_time, M)
    DECL_ACCESSOR(EndTime, EndTime, int64_t, end_time, M)
    DECL_ACCESSOR(ComputeCost, ComputeCost, int64_t, compute_cost, M)
    DECL_ACCESSOR(TemporaryMemory, TemporaryMemory, int64_t, temporary_memory, M)
    DECL_ACCESSOR(PersistentMemory, PersistentMemory, int64_t, persistent_memory, M)
    DECL_ACCESSOR(InputMemory, InputMemory, int64_t, input_memory, M)
    DECL_ACCESSOR(OutputMemory, OutputMemory, int64_t, output_memory, M)
    // // DECL_ACCESSOR(T, data)
    void AddInput(const std::string& input) {
        inputs.push_back(input);
    }
    void AddOutput(const std::string& output) {
        outputs.push_back(output);
    }
};

class MergedNode : public NodeBase {
    std::vector<NodeBase> merged_nodes;  // 已合并节点
};

}  // namespace framework

// NOLINTBEGIN(readability-identifier-naming)
template <>
struct fmt::formatter<framework::NodeBase> {
    static constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const framework::NodeBase& n, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "NodeBase(name={}, op={}, device={}, inputs={}, attrs={})", n.name, n.op,
                              n.device, n.inputs, n.attrs);
    }
};
// NOLINTEND(readability-identifier-naming)

#endif /* ifndef _GRAPH_NODE_H */
