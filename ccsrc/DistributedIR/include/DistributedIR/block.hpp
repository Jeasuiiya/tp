#pragma once

#ifndef FRAMEWORK_IR_BLOCK_H
#define FRAMEWORK_IR_BLOCK_H
#include <string>
#include <vector>

#include "DistributedIR/graph.hpp"
namespace framework {

class Block {
   public:
    Block();
    virtual ~Block();

   private:
    std::string device;
    SubGraph graph;
};

class BlockSchedule {
   public:
    BlockSchedule();
    virtual ~BlockSchedule();

   private:
    /* data */
    std::vector<Block> blocks;
};

}  // namespace framework
#endif /* end of include guard: FRAMEWORK_IR_BLOCK_H */
