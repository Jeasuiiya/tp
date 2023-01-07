include(CMakeParseArguments)
function(add_header_only_library target_name build_interface install_interface)
    cmake_parse_arguments(_ARG "" "" "DEPENDS" ${ARGN})
    add_library(${target_name} INTERFACE)
    target_include_directories(
        ${target_name}
        INTERFACE
        $<BUILD_INTERFACE:${build_interface}>
        $<INSTALL_INTERFACE:${install_interface}>
    )
    target_link_libraries(${target_name} INTERFACE ${_ARG_DEPENDS})
endfunction(add_header_only_library)
