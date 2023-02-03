# require FetchContent
include(FetchContent)
# macro()
    
# endmacro()

macro(populate_llvm_source)
    FetchContent_GetProperties(llvm)
    if(NOT llvm_POPULATED)
        message(STATUS "Download llvm project source code")
        FetchContent_Populate(llvm)
        if (WIN32)
            if (EXISTS ${llvm_SOURCE_DIR}/llvm/resources/windows_version_resource.rc)
                file(RENAME
                    ${llvm_SOURCE_DIR}/llvm/resources/windows_version_resource.rc
                    ${llvm_SOURCE_DIR}/llvm/resources/windows_version_resource.rc.bak)
            endif()
        endif()
    endif()
endmacro(populate_llvm_source)

function(fetch_llvm)
    FetchContent_Declare(
    llvm
    URL https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-15.0.7.zip
    )
    set(LLVM_ENABLE_PROJECTS mlir CACHE STRING "")
    set(LLVM_TARGETS_TO_BUILD host CACHE STRING "")
    set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
    set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
    set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
    set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
    set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
    set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE STRING "")
    set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
    set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
    populate_llvm_source()
    add_subdirectory(${llvm_SOURCE_DIR}/llvm ${llvm_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction(fetch_llvm)


