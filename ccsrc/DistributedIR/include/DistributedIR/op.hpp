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

#include "common/util.hpp"
namespace framework {

/*! \enum ElementType
 *
 *  Detailed description
 */
enum class ElementType { U8, I8, U16, I16, U32, I32, U64, I64, F32, F64 };

class Arg {
   public:
    Arg() = default;
    ~Arg() = default;

   private:
    /* data */
    std::set<ElementType> element_type;
    std::vector<uint> shape;
};

class Op {
   public:
    Op(uint in, uint out) {
        this->in = in;
        this->out = out;
        in_args = new Arg[in];
        out_args = new Arg[out];
    }
    ~Op() {
        delete[] in_args;
        delete[] out_args;
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << "Op("
           << "in=" << in << ",out=" << out << ",name=" << name << ")";
        return ss.str();
    }
    GEN_ACCESSOR_IN_DEC(std::string, name)

   private:
    /* data */
    uint in;
    uint out;
    std::string name;
    Arg* in_args;
    Arg* out_args;
};

class OpRegistry {
   public:
    OpRegistry() = default;
    ~OpRegistry() = default;
    void Register(Op op);
    static OpRegistry* Global();
    std::string to_string() {
        std::stringstream ss;
        ss << "OpRegistry["
           << std::accumulate(ops.begin(), ops.end(), std::string(),
                              [](const std::string& s, const Op& p) {
                                  return s +
                                         (s.empty() ? std::string() : ", ") +
                                         p.to_string();
                              })
           << "]";
        return ss.str();
    }

   private:
    std::vector<Op> ops;
};

}  // namespace framework
#endif /* end of include guard: FRAMEWORK_IR_OP_H */
