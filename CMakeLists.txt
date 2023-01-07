cmake_minimum_required(VERSION 3.22)
project(distrbuted_framework CXX)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(BUILD_TEST "build test" ON)
# set(CMAKE_MESSAGE_LOG_LEVEL TRACE)
include(HeaderOnlyLibrary)
include(FetchSpdlog)
fetch_spdlog()


add_subdirectory(ccsrc)


if(BUILD_TEST)
    include(FetchGTest)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    fetch_gtest()
    add_subdirectory(tests)
endif(BUILD_TEST)





# add_subdirectory(ExternalProject)