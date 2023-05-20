add_cc_gtest(test_DistributedIR SRCS test_DistributedIR.cc DEPENDS DistributedIR)

add_cc_gtest(test_Common SRCS test_common.cc DEPENDS CommonHeaders DistributedIR)

add_cc_gtest(test_cost_graph SRCS test_cost_graph.cc DEPENDS CostGraph)

add_cc_gtest(
  test_aware_fusion
  SRCS test_aware_fusion.cc
  DEPENDS CostGraph
  DEFINITIONS SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)

add_cc_gtest(
  test_networkx_generator
  SRCS test_networkx_generator.cc
  DEPENDS AwarePolicy CostGraph
  DEFINITIONS SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)

add_cc_gtest(
  test_aware
  SRCS test_aware.cc
  DEPENDS AwarePolicy CostGraph
  DEFINITIONS SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)

add_cc_gtest(
  test_trinity 
  SRCS test_trinity.cc 
  DEPENDS TrinityPolicy AwarePolicy CostGraph
  DEFINITIONS SOURCE_PYTHONPATH=${CMAKE_SOURCE_DIR}/python)

add_cc_gtest(test_fddps SRCS test_fddps.cc DEPENDS FDDPSPolicy CostGraph)
