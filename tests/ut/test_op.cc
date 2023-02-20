#include <iostream>

#include "DistributedIR/op.hpp"

int main(int argc, char *argv[]) {
    std::cout << "1" << std::endl;
    framework::Op op(1, 1);
    std::cout << "2" << std::endl;
    framework::OpRegistry::Global()->Register(op);
    std::cout << "3" << framework::OpRegistry::Global()->ToString()
              << std::endl;
    delete framework::OpRegistry::Global();
    std::cout << "4" << std::endl;
    return 0;
}
