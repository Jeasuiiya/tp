# require CPM
include(CPM)

# fetch_result
function(fetch_range)
  CPMAddPackage(
    NAME range
    GITHUB_REPOSITORY ericniebler/range-v3
    GIT_TAG d04bfc # Commits on Mar 3, 2023
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON)
endfunction(fetch_range)
