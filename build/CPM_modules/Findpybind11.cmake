include("/home/ai/cy/temp/GeeSibling/cmake/third_party/CPM.cmake")
CPMAddPackage("NAME;pybind11;GITHUB_REPOSITORY;pybind/pybind11;GIT_TAG;v2.10.3;EXCLUDE_FROM_ALL;ON")
set(pybind11_FOUND TRUE)