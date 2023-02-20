# require CPM
include(CPM)

# fetch_spdlog
function(fetch_spdlog)
  CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    GIT_TAG v1.11.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON)
endfunction(fetch_spdlog)
