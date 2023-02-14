#pragma once
#ifndef _FRAMEWORK_CLUSTER_CLUSTER_H
#define _FRAMEWORK_CLUSTER_CLUSTER_H

#include <vector>

#include "server.hpp"
namespace framework {

class Cluster {
    std::vector<Server> servers;  // 服务器
    std::vector<Link> links;      // 服务器链接
    long totalMemory;             // 集群总内存
    long usedMemory;              // 集群已使用内存
};

};  // namespace framework

#endif