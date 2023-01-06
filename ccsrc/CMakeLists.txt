add_subdirectory(cluster)
add_subdirectory(common)
add_subdirectory(cost_graph)
add_subdirectory(DistributedIR)
add_subdirectory(policy)
add_subdirectory(simulator)

add_executable(test_log test_log.cc)
target_link_libraries(test_log PRIVATE spdlog::spdlog)
