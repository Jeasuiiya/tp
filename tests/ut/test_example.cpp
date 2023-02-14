#include <gtest/gtest.h>
// #include "add.h"
#include "DistributedIR/op.hpp"

// Demonstrate some basic assertions.
TEST(TestExample, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
  // EXPECT_EQ(add(1, 2), 3);
}
