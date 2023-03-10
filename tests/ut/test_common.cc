#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "common/fmt.hpp"
#include "DistributedIR/node.hpp"
namespace framework {
// Demonstrate some basic assertions.
TEST(TestCommon, FmtRaiiPtr) {
    EXPECT_NO_THROW(std::cout << std::make_shared<std::string>("") << std::endl);
    EXPECT_NO_THROW(std::cout << fmt_unique(std::make_unique<std::string>("")) << std::endl);
    EXPECT_NO_THROW(std::cout << fmt_unique(std::make_unique<framework::NodeBase>()) << std::endl);
}
}  // namespace framework
