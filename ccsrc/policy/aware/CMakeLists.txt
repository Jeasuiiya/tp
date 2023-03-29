add_library(Policy networkx_generator.cc aware_interface.cc)
target_link_libraries(Policy PUBLIC PolicyHeaders)
