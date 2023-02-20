#include "DistributedIR/op.hpp"

#include <memory>

namespace framework {

OpRegistry* OpRegistry::Global() {
    static auto* global_op_registry = new OpRegistry();
    return global_op_registry;
}
void OpRegistry::Register(Op const& op) { this->ops.push_back(op); }

}  // namespace framework
