cmake_minimum_required(VERSION 3.22)
project(distrbuted_framework CXX)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(BUILD_TEST "build test" ON)

include(FetchSpdlog)
fetch_spdlog()
add_subdirectory(ccsrc)



if(BUILD_TEST)
    include(FetchGTest)
    fetch_gtest()
    add_subdirectory(tests)
endif(BUILD_TEST)





# add_subdirectory(ExternalProject)