# require CPM
include(CPM)
# cmake-lint: disable=C0103
# fetch_pybind11
function(fetch_pybind11)
  if(NOT TARGET pybind11::pybind11)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
    CPMAddPackage(
      NAME pybind11
      GITHUB_REPOSITORY pybind/pybind11
      GIT_TAG v2.10.3
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_pybind11)

# cmake-lint: disable=W0106
# add_python_subdirectory subdirectory [BINARY_DIR]
function(add_python_subdirectory subdirectory)
  message(STATUS "${ARGV0} in ${CMAKE_CURRENT_SOURCE_DIR} -> $\{CMAKE_BINARY_DIR\}/python/${ARGV1}")
  add_subdirectory(${subdirectory} ${CMAKE_BINARY_DIR}/python/${ARGV1})
endfunction(add_python_subdirectory)
# add_pybind11_module target [SOURCE ...]
function(add_pybind11_module)
  pybind11_add_module(${ARGN})
  target_compile_definitions(${ARGV0} PRIVATE VERSION_INFO=${PROJECT_VERSION} PYBIND11_CURRENT_MODULE_NAME=${ARGV0})
  add_dependencies(pymodule ${ARGV0})
endfunction(add_pybind11_module)

add_custom_target(pymodule COMMENT "pymodule")
