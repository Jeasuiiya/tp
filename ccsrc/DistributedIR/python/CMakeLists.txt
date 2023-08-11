add_pybind11_module(_graph framework/core/lib SRCS graph.cc DEPENDS DistributedIR FDDPSPolicy ClusterHeaders SGPPolicy)
