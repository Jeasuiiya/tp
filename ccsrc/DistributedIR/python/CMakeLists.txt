add_pybind11_module(_graph graph.cc)
target_link_libraries(_graph PUBLIC DistributedIRHeaders)

