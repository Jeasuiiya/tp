#include <string>
#include <vector>

class Var {
    int type;          // float64
    vector<int> size;  // 2,3,4
};

class Op {
    std::string name;
    int input_num;
    vector<Var> inputs;
    int output_num;
    vector<Var> outputs;
};

class StageOp : public Op {
    std::vector<ModuleOp> region;  // ModuleOp: MLIR op
    std::string device;
}
