# require CPM
include(CPM)

# fetch_result
function(fetch_range)
  CPMAddPackage(
    NAME range
    URL "https://github.com/ericniebler/range-v3/archive/541b06320b89c16787cc6f785f749f8847cf2cd1.zip" # Commits on Mar 14, 2023
    EXCLUDE_FROM_ALL ON)
endfunction(fetch_range)