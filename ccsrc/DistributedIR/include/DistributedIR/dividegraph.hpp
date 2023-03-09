#ifndef FRAMEWORK_IR_DivideGraph_H
#define FRAMEWORK_IR_DivideGraph_H

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>

#include "common/error.hpp"
#include "common/result_macro.hpp"
#include "graph.hpp"
#include "node.hpp"

namespace framework {

class Nodevalue {
  public:
    NodeBase node;
    bool device_in = false;                    // 默认不是device_in算子
    bool device_out = false;                   // 默认不是device_out算子
    std::vector<std::string> device_in_node;   // 同设备的前驱算子
    std::vector<std::string> device_out_node;  // 同设备的后继算子
    int inputs_num = 0;
    int outputs_num = 0;
    int subgraph_num = -1;  // 默认为-1,若为-1则表示该算子没有被搜索过；
};
// 搜索方式1
bool SearchBeforeNode(std::map<std::string, Nodevalue>& sub_graph_info, std::string& before_node);
// 搜索方式2
bool SearchNextNode(std::map<std::string, Nodevalue>& sub_graph_info, std::string& next_node);
// 切断符合条件的节点
cpp::result<void, Error> SubgraphSearch(std::map<std::string, Nodevalue>& sub_graph_info,
                                        std::vector<std::string>& device_in, std::vector<std::string>& device_out);
// 广度优先搜索图，并且给对应算子标记子图编号
void NodeBfs(std::map<std::string, Nodevalue>& sub_graph_info, int subgraph_num, std::vector<std::string>& joint_nodes,
             std::multimap<int, NodeBase>& nodes_to_subgraph);
// 获取子图编号
int CreateSubgraphNum(std::map<std::string, Nodevalue>& sub_graph_info,
                      std::multimap<int, NodeBase>& nodes_to_subgraph);
// 生成子图并返回
std::map<int, SubGraph> CreateSubgraph(std::map<std::string, Nodevalue>& sub_graph_info,
                                       std::multimap<int, NodeBase>& nodes_to_subgraph, int subgraph_num);
// 入口函数
cpp::result<std::map<int, SubGraph>, Error> DivideGraph(Graph graph);

}  // namespace framework

#endif
