add_executable(
  test_example
  test_example.cpp
)
target_link_libraries(
  test_example
  GTest::gtest_main
  Op
)

add_executable(test_op test_op.cc)
target_link_libraries(
  test_op
  Op
)

include(GoogleTest)
gtest_discover_tests(test_example)
