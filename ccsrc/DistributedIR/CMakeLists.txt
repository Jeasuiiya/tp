add_header_only_library(DistributedIRHeaders ${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CMAKE_INSTALL_INCLUDEDIR}/DistributedIR DEPENDS CommonHeaders)
if(FRAMEWORK_ENABLE_BINDINGS_PYTHON)
  add_python_subdirectory(python ${CMAKE_PROJECT_NAME}/core)
endif()

add_library(DistributedIR op.cc block.cc dividegraph.cpp)
target_link_libraries(DistributedIR DistributedIRHeaders)
