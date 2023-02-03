# require FetchContent
include(FetchContent)

function(fetch_pybind11)
    FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip
    )
    FetchContent_MakeAvailable(pybind11)
endfunction(fetch_pybind11)

function(add_python_subdirectory subdirectory)
    message(STATUS "${ARGV0} in ${CMAKE_CURRENT_SOURCE_DIR} -> $\{CMAKE_BINARY_DIR\}/python/${ARGV1}")
    add_subdirectory(${subdirectory} ${CMAKE_BINARY_DIR}/python/${ARGV1})
endfunction(add_python_subdirectory)

function(add_pybind11_module)
    pybind11_add_module(${ARGN})
    target_compile_definitions(${ARGV0}
                            PRIVATE
                            VERSION_INFO=${PROJECT_VERSION}
                            PYBIND11_CURRENT_MODULE_NAME=${ARGV0})
    add_dependencies(pymodule ${ARGV0})
endfunction(add_pybind11_module)

add_custom_target(pymodule)