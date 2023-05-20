add_subdirectory(include)
add_library(CostGraph fusion/aware_fusion.cc)
target_link_libraries(CostGraph CostGraphHearders)
