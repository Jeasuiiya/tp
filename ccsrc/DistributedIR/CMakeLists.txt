add_subdirectory(include)
# include_directories(include)


add_library(a STATIC a.cc)
target_link_libraries(a DistributedIRHeaders ClusterHeaders)
# add_dependencies(a DistributedIRHeaders)