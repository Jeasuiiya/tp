

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

include(GoogleTest)
gtest_discover_tests(test_DistributedIR)
gtest_discover_tests(test_Common)
