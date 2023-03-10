# require CPM
include(CPM)

# fetch_result
function(fetch_fmt)
  CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 9.1.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON)
endfunction(fetch_fmt)
