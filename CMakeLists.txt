cmake_minimum_required(VERSION 3.22)

# provide args:
# FRAMEWORK_BUILD_TEST
# FRAMEWORK_ENABLE_BINDINGS_PYTHON
# PYTHON_EXECUTABLE
# CPM_SOURCE_CACHE
project(framework CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/third_party)

option(FRAMEWORK_BUILD_TEST "build test" ON)

# cmake -DPYBIND11_PYTHON_VERSION=3.6 ..

# # Another method: cmake -DPYTHON_EXECUTABLE=/path/to/python ..

# # This often is a good way to get the current Python, works in environments: cmake -DPYTHON_EXECUTABLE=$(python3 -c
# "import sys; print(sys.executable)") ..
option(FRAMEWORK_ENABLE_BINDINGS_PYTHON "python bindings" ON)

# begin config set(CMAKE_MESSAGE_LOG_LEVEL TRACE)
include(CPM)
include(HeaderOnlyLibrary)
include(FetchSpdlog)
include(FetchResult)
include(FetchFmt)
include(FetchLLVM)
include(FetchBoost)

# set(FETCHCONTENT_QUIET FALSE)

# fetch_llvm()
fetch_result()
fetch_fmt()
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
  enable_testing()
  add_subdirectory(tests)
endif(FRAMEWORK_BUILD_TEST)
