#pragma once

#ifndef FRAMEWORK_IR_OP_H
#define FRAMEWORK_IR_OP_H

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "DistributedIR/tensor.hpp"
#include "common/util.hpp"
namespace framework {

class Op {
  public:
    Op(uint in, uint out) {
        this->in = in;
        this->out = out;
        in_args = new AbstractTensor[in];
        out_args = new AbstractTensor[out];
    }
    ~Op() {
        delete[] in_args;
        delete[] out_args;
    }

  private:
    /* data */
    uint in;
    uint out;
    std::string name;
    AbstractTensor* in_args;
    AbstractTensor* out_args;
};

class OpRegistry {
  public:
    OpRegistry() = default;
    ~OpRegistry() = default;
    void Register(Op const& op);
    static OpRegistry* Global();

  private:
    std::vector<Op> ops;
};

}  // namespace framework
#endif /* end of include guard: FRAMEWORK_IR_OP_H */
