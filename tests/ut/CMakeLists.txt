add_executable(
  test_example
  test_example.cpp
)
target_link_libraries(
  test_example
  GTest::gtest_main
  DistributedIR
)

add_executable(
    test_DistributedIR
    test_DistributedIR.cc
)
target_link_libraries(
  test_DistributedIR
  GTest::gtest_main
  DistributedIR
)
add_executable(
    test_Common
    test_common.cc
)
target_link_libraries(
  test_Common
  GTest::gtest_main
  CommonHeaders
  DistributedIR
)

add_executable(test_op test_op.cc)
target_link_libraries(
  test_op
  DistributedIR
)

include(GoogleTest)
gtest_discover_tests(test_example)
gtest_discover_tests(test_DistributedIR)
gtest_discover_tests(test_Common)
