add_library(AwarePolicy networkx_generator.cc aware_interface.cc)
target_link_libraries(AwarePolicy PUBLIC PolicyHeaders)

