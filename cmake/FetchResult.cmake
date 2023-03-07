# require CPM
include(CPM)

# fetch_result
function(fetch_result)
  CPMAddPackage(
    NAME result
    GITHUB_REPOSITORY bitwizeshift/result
    GIT_TAG v1.0.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON)
endfunction(fetch_result)
