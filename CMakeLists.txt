cmake_minimum_required(VERSION 3.22)
project(framework CXX)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(FRAMEWORK_BUILD_TEST "build test" ON)
option(FRAMEWORK_ENABLE_BINDINGS_PYTHON "python bindings" ON)
# set(CMAKE_MESSAGE_LOG_LEVEL TRACE)
include(HeaderOnlyLibrary)
include(FetchSpdlog)

fetch_spdlog()
if(FRAMEWORK_ENABLE_BINDINGS_PYTHON)
    include(FetchPybind11)
    fetch_pybind11()
endif()

add_subdirectory(ccsrc)

if(FRAMEWORK_BUILD_TEST)
    include(FetchGTest)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    fetch_gtest()
    add_subdirectory(tests)
endif(FRAMEWORK_BUILD_TEST)





# add_subdirectory(ExternalProject)