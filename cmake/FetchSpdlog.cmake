
# require FetchContent
include(FetchContent)

function(fetch_spdlog)
    FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    FetchContent_MakeAvailable(spdlog)
endfunction(fetch_spdlog)

