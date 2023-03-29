

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

add_executable(test_cost_graph test_cost_graph.cc)
target_link_libraries(
  test_cost_graph
  GTest::gtest_main
  CostGraph
)

add_executable(test_aware_fusion test_aware_fusion.cc)
target_link_libraries(
  test_aware_fusion
  GTest::gtest_main
  CostGraph
)

add_executable(test_networkx_generator test_networkx_generator.cc)
target_link_libraries(
  test_networkx_generator
  GTest::gtest_main
  Policy
  CostGraph
)

add_executable(test_aware test_aware.cc)
target_link_libraries(
  test_aware
  GTest::gtest_main
  Policy
  CostGraph
)

add_executable(test_aware_main test_aware_main.cc)
target_link_libraries(
  test_aware_main
  GTest::gtest_main
  Policy
  CostGraph
)


target_compile_definitions(test_networkx_generator PRIVATE SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)
target_compile_definitions(test_aware PRIVATE SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)
target_compile_definitions(test_aware_main PRIVATE SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)

include(GoogleTest)
gtest_discover_tests(test_DistributedIR)
gtest_discover_tests(test_Common)
gtest_discover_tests(test_cost_graph)
gtest_discover_tests(test_aware_fusion)
gtest_discover_tests(test_networkx_generator)
gtest_discover_tests(test_aware)