#pragma once
#ifndef _FRAMEWORK_CLUSTER_SERVER_H
#define _FRAMEWORK_CLUSTER_SERVER_H

#include <vector>

namespace framework {
enum class DeviceStatus {
    Using,  //使用
    Idle    //闲置
};
enum class DeviceType { Cpu, NVGpu, AMDGpu, Ascend };
class Device {
    DeviceStatus status;  //设备使用状态
    DeviceType type;      //设备类型
    std::string name;     //设备逻辑名称
    long memory;          //设备总内存
    long usedMemory;      //已使用内存
};

enum class LinkType { PCIE, NVLink, Eth };
class Link {
    LinkType type;                 //链接类型
    std::vector<Device*> devices;  //可用设备
    long totalBandwidth;           //共享总带宽
};

class Server {
    DeviceStatus status;          //设备使用状态
    std::string name;             //服务器逻辑名称
    std::vector<Device> devices;  //计算设备
    long usedMemory;              //已使用内存
    long totalMemory;             //总内存
    std::vector<Link> links;      //链接
};

};  // namespace framework

#endif