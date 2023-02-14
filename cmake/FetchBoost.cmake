
# require FetchContent
include(FetchContent)

function(fetch_boost)
    FetchContent_Declare(
        boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.7z
    )
    FetchContent_MakeAvailable(boost)
    if(IS_DIRECTORY "${boost_SOURCE_DIR}")
        set_property(DIRECTORY ${boost_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
    endif()

endfunction(fetch_boost)

