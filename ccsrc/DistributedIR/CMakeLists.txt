add_subdirectory(include)
if(FRAMEWORK_ENABLE_BINDINGS_PYTHON)
    add_python_subdirectory(python ${CMAKE_PROJECT_NAME}/core)
endif()

add_library(DistributedIR op.cc block.cc)
target_link_libraries(DistributedIR DistributedIRHeaders)

