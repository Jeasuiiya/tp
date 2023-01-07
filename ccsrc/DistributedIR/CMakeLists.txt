add_subdirectory(include)
if(FRAMEWORK_ENABLE_BINDINGS_PYTHON)
    add_python_subdirectory(python)
endif()


add_library(a STATIC a.cc)
target_link_libraries(a DistributedIRHeaders ClusterHeaders)
