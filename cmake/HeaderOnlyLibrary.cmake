include(CMakeParseArguments)
# add_header_only_library(target build_include_dir install_include_dir [DEPENDS <targets>] [ALIAS <alias_name>])
function(add_header_only_library target_name build_interface install_interface)
    cmake_parse_arguments(_ARG "" "ALIAS" "DEPENDS" ${ARGN})
    if(_ARG_ALIAS)
        add_library(${target_name} INTERFACE ALISA ${_ARG_ALIAS})
    else()
        add_library(${target_name} INTERFACE)
    endif()
    target_include_directories(
        ${target_name}
        INTERFACE
        $<BUILD_INTERFACE:${build_interface}>
        $<INSTALL_INTERFACE:${install_interface}>
    )
    target_link_libraries(${target_name} INTERFACE ${_ARG_DEPENDS})
endfunction(add_header_only_library)
