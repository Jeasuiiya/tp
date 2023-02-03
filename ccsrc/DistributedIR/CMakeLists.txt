add_subdirectory(include)
if(FRAMEWORK_ENABLE_BINDINGS_PYTHON)
    add_python_subdirectory(python ${CMAKE_PROJECT_NAME}/core)
endif()

add_library(Op op.cc)
target_link_libraries(Op DistributedIRHeaders)
