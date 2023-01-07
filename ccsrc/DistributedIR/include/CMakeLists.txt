
add_header_only_library(DistributedIRHeaders
    ${CMAKE_CURRENT_SOURCE_DIR}/DistributedIR
    ${CMAKE_INSTALL_INCLUDEDIR}/DistributedIR
    DEPENDS CommonHeaders)