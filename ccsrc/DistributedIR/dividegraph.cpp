#include "DistributedIR/dividegraph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>

#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"

namespace framework {

// 深度优先搜索
bool SearchBeforeNode(std::map<std::string, Nodevalue>& sub_graph_info, std::string& before_node) {
    std::queue<std::string> before_node_queue;
    before_node_queue.push(before_node);
    bool flag = false;
    while (!before_node_queue.empty()) {
        std::string the_node = before_node_queue.front();
        Nodevalue the_node_value = sub_graph_info.find(the_node)->second;
        if (the_node_value.device_out) {
            flag = true;
            break;
        }
        if (the_node_value.inputs_num != 0) {
            for (auto& before_node : the_node_value.device_in_node) {
                before_node_queue.push(before_node);
            }
        }
        before_node_queue.pop();
    }
    return flag;
}
bool SearchNextNode(std::map<std::string, Nodevalue>& sub_graph_info, std::string& next_node) {
    std::queue<std::string> next_node_queue;
    next_node_queue.push(next_node);
    bool flag = false;
    while (!next_node_queue.empty()) {
        std::string the_node = next_node_queue.front();
        Nodevalue the_node_value = sub_graph_info.find(the_node)->second;
        if (the_node_value.device_in) {
            flag = true;
            break;
        }
        if (the_node_value.inputs_num != 0) {
            for (auto& next_node : the_node_value.device_out_node) {
                next_node_queue.push(next_node);
            }
        }
        next_node_queue.pop();
    }
    return flag;
}

cpp::result<void, Error> SubgraphSearch(std::map<std::string, Nodevalue>& sub_graph_info,
                                        std::vector<std::string>& device_in_group,
                                        std::vector<std::string>& device_out_group) {
    // 先写一个最简单的dfs 获得完整的图为一个子图
    // 遍历device_in中的节点，查询同设备中是否有device_out的节点在该节点之前
    int node_num = device_in_group.size();
    std::vector<std::pair<std::string, std::string>> cut_edges;  // 为了不重复切分
    for (int i = 0; i < node_num; i++) {
        // 同设备中是否切断算子之间的边 1为需要切断 0为保留
        std::string current_node = device_in_group.at(i);
        Nodevalue& current_node_value = sub_graph_info.find(current_node)->second;
        if (current_node_value.device_in)  // 再次确认
        {
            // 找寻该算子放在同设备的前驱算子中是否有device_out
            for (auto& before_node : current_node_value.device_in_node) {
                bool result = SearchBeforeNode(sub_graph_info, before_node);
                if (result) {
                    cut_edges.emplace_back(before_node, current_node);
                    std::cout << before_node << "--" << current_node << ":切断" << std::endl;
                }
            }
        } else {
            return cpp::fail(Error(Kind::Invalid, "The device_in_flag of node is fault"));
        }
    }

    // 遍历device_out中的节点，查询同设备中是否有device_in的节点在该节点之后
    int out_node_num = device_out_group.size();
    for (int i = 0; i < out_node_num; i++) {
        std::string current_node = device_out_group.at(i);
        Nodevalue& current_node_value = sub_graph_info.find(current_node)->second;
        if (current_node_value.device_out) {
            for (auto& next_node : current_node_value.device_out_node) {
                bool result = SearchNextNode(sub_graph_info, next_node);
                if (result) {
                    cut_edges.emplace_back(current_node, next_node);
                    std::cout << current_node << "--" << next_node << ":切断" << std::endl;
                }
            }

        } else
            return cpp::fail(Error(Kind::Invalid, "The device_out_flag of node is fault"));
        // 在current_node的device_in_node和before_node的device_out_node中擦除需要切断的算子，方便组成子图
        for (auto& cut_edge : cut_edges) {
            std::string before_node = cut_edge.first;    // 在同设备后继中擦除current_node
            std::string current_node = cut_edge.second;  // 在同设备前驱中擦除before_node
            Nodevalue& before_node_value = sub_graph_info.find(before_node)->second;
            std::vector<std::string>& device_out_node = before_node_value.device_out_node;
            for (auto iter = device_out_node.begin(); iter != device_out_node.end();) {
                if (*iter == current_node)
                    iter = device_out_node.erase(iter);
                else
                    iter++;
            }
            Nodevalue& current_node_value = sub_graph_info.find(current_node)->second;
            std::vector<std::string>& device_in_node = current_node_value.device_in_node;
            for (auto iter = device_in_node.begin(); iter != device_in_node.end();) {
                if (*iter == before_node)
                    iter = device_in_node.erase(iter);
                else
                    iter++;
            }
        }
    }
    return {};
}

int CreateSubgraphNum(std::map<std::string, Nodevalue>& sub_graph_info,
                      std::multimap<int, NodeBase>& nodes_to_subgraph) {
    int subgraph_num = -1;
    // 给算子子图编号
    for (auto& iter : sub_graph_info) {
        std::string current_node_name = iter.first;
        auto& current_node_value = iter.second;
        std::queue<std::string> joint_nodes;
        if (current_node_value.subgraph_num == -1) {
            joint_nodes.push(current_node_name);
            subgraph_num++;
        }
        while (!joint_nodes.empty()) {
            std::string the_node = joint_nodes.front();
            Nodevalue& the_node_value = sub_graph_info.find(the_node)->second;
            the_node_value.subgraph_num = subgraph_num;
            nodes_to_subgraph.insert(std::pair<int, NodeBase>(subgraph_num, the_node_value.node));
            std::vector<std::string> device_in_node = the_node_value.device_in_node;  // 当前算子还相连的前驱算子
            std::vector<std::string> device_out_node = the_node_value.device_out_node;  // 当前算子还相连的后继算子
            for (auto& in_op : device_in_node) {
                Nodevalue& in_op_value = sub_graph_info.find(in_op)->second;
                if (in_op_value.subgraph_num == -1) joint_nodes.push(in_op);
            }
            for (auto& out_op : device_out_node) {
                Nodevalue& out_op_value = sub_graph_info.find(out_op)->second;
                if (out_op_value.subgraph_num == -1) joint_nodes.push(out_op);
            }
            joint_nodes.pop();
        }
    }
    return subgraph_num + 1;
}

std::map<int, SubGraph> CreateSubgraph(std::map<std::string, Nodevalue>& sub_graph_info,
                                       std::multimap<int, NodeBase>& nodes_to_subgraph, int subgraph_num) {
    std::map<int, SubGraph> sub_graphs;  // 所有的子图合集
    // 将算子放进对应的子图
    for (int i = 0; i < subgraph_num; i++) {
        SubGraph sub_graph;
        std::vector<NodeBase> nodes_list;
        std::map<std::string, NodeBase&> node_map;
        auto node_to_place = nodes_to_subgraph.find(i);
        while (node_to_place != nodes_to_subgraph.end()) {
            std::cout << "subgraph" << i << "add:" << node_to_place->second.Name() << std::endl;
            sub_graph.AddNode(node_to_place->second);
            nodes_to_subgraph.erase(node_to_place);
            node_to_place = nodes_to_subgraph.find(i);
        }
        sub_graphs.insert(std::pair<int, SubGraph>(i, sub_graph));
    }
    if (subgraph_num != sub_graphs.size()) {
        std::cout << "subgraph_num is " << subgraph_num << std::endl;
        std::cout << "subgraphs'size is " << sub_graphs.size() << std::endl;
        std::cout << "error:they are different!!!" << std::endl;
    }

    // //获取子图连接信息
    for (auto& iter : sub_graphs) {
        SubGraph& current_sub_graph = iter.second;
        auto current_sub_graph_num = iter.first;
        auto current_nodes = current_sub_graph.Nodes();
        // map<前驱的子图信息int 对应的节点输入map<string string>>
        std::map<int, std::multimap<std::string, std::string>> subgraph_op_input;  // before_node 前驱节点中第几个输出
        std::map<int, std::multimap<std::string, std::string>> subgraph_op_output;  // next_node 后继节点中第几个输入
        // 获得前子图，获得后子图
        // 获得图input   格式：  before节点名:输出index
        // 获得图output  格式：  current节点名:输入index  input_data排序得出
        for (auto& i : current_nodes)  // 遍历子图中的所有节点
        {
            NodeBase& current_node = *i;
            Nodevalue& current_node_value = sub_graph_info.find(current_node.Name())->second;
            auto device_in_node = current_node_value.device_in_node;
            auto device_out_node = current_node_value.device_out_node;
            auto inputs = current_node.Inputs();
            auto outputs = current_node.Outputs();
            std::vector<std::string> inputs_diff;
            std::vector<std::string> outputs_diff;  // 差集及为前驱后继不同子图的算子
            std::set_difference(inputs.begin(), inputs.end(), device_in_node.begin(), device_in_node.end(),
                                inserter(inputs_diff, inputs_diff.begin()));  // old-->new需要删除的
            std::set_difference(outputs.begin(), outputs.end(), device_out_node.begin(), device_out_node.end(),
                                inserter(outputs_diff, outputs_diff.begin()));  // old-->new需要删除的

            if (!inputs_diff.empty()) {
                // 该节点是图边缘节点
                // 找寻非同设备前驱的节点，前驱的output_data和该节点的input_data对上的部分放进第一个string
                int input_index = 0;
                std::multimap<std::string, std::string> data2data;
                auto current_inputs_data = current_node.InputsData();
                for (auto& input_data : current_inputs_data) {
                    for (auto& before_node : inputs_diff) {
                        auto before_node_value = sub_graph_info.find(before_node)->second;
                        NodeBase& before_node_real = before_node_value.node;
                        auto before_outputs_data = before_node_real.OutputsData();
                        auto iter = std::find(before_outputs_data.begin(), before_outputs_data.end(), input_data);
                        if (iter != before_outputs_data.end()) {
                            // 放进map1<input_data, currentnode.name:input_index>
                            //  上个节点的第几个输出  当前节点的第几个输入
                            data2data.insert(std::pair<std::string, std::string>(
                                input_data, current_node.Name() + ":" + std::to_string(input_index)));
                        }
                    }
                    input_index++;
                }
                // 放进subgraph_op_input<当前node所在map的序号，map1>
                subgraph_op_input.insert(std::pair<int, std::multimap<std::string, std::string>>(
                    current_node_value.subgraph_num, data2data));
            }
            if (!outputs_diff.empty()) {
                int input_index = 0;
                std::multimap<std::string, std::string> data2data;
                auto current_outputs_data = current_node.OutputsData();
                for (auto& output_data : current_outputs_data) {
                    for (auto& next_node : outputs_diff) {
                        auto next_node_value = sub_graph_info.find(next_node)->second;
                        NodeBase& next_node_real = next_node_value.node;
                        auto next_inputs_data = next_node_real.InputsData();
                        auto iter = std::find(next_inputs_data.begin(), next_inputs_data.end(), output_data);
                        input_index = std::distance(next_inputs_data.begin(), iter);
                        if (iter != next_inputs_data.end()) {
                            // 这个节点的哪个输出 下个节点的第几个输入
                            data2data.insert(std::pair<std::string, std::string>(
                                output_data, next_node_real.Name() + ":" + std::to_string(input_index)));
                        }
                    }
                }
                subgraph_op_output.insert(std::pair<int, std::multimap<std::string, std::string>>(
                    current_node_value.subgraph_num, data2data));
            }
        }
        for (auto& before : subgraph_op_input) {
            int before_subgraph_num = before.first;
            std::multimap<std::string, std::string> op_op = before.second;
            SubGraph before_subgraph = sub_graphs.find(before_subgraph_num)->second;
            current_sub_graph.AddInputGraph(before_subgraph);
            current_sub_graph.AddInput(op_op);
        }

        for (auto& next : subgraph_op_output) {
            int next_subgraph_num = next.first;
            std::multimap<std::string, std::string> op_op = next.second;
            SubGraph next_subgraph = sub_graphs.find(next_subgraph_num)->second;
            current_sub_graph.AddOutputGraph(next_subgraph);
            current_sub_graph.AddOutput(op_op);
        }
    }
    // 输出子图信息
    for (auto& iter : sub_graphs) {
        std::cout << "======================================" << std::endl;
        int subgraph_num = iter.first;
        SubGraph current_subgraph = iter.second;
        auto nodes_list = current_subgraph.Nodes();
        std::cout << "The subgraph num is " << subgraph_num << ". This subgraph contains " << nodes_list.size()
                  << " nodes.They are :";
        for (auto& i : nodes_list) {
            NodeBase& the_node = *i;
            std::cout << the_node.Name() << " ";
        }
        std::cout << std::endl;

        // std::cout<<"The number of output subgraphs is :"<<current_subgraph.output_graphs.size()<<std::cout;
        std::cout << "before_op_output--current_subgraph_input are :" << std::endl;
        for (auto& op_op : current_subgraph.GetInputs()) {
            for (auto& iter : op_op) {
                std::cout << iter.first << "--" << iter.second << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "current_subgraph_output--next_op_input are :" << std::endl;
        for (auto& op_op : current_subgraph.GetOutputs()) {
            for (auto& iter : op_op) {
                std::cout << iter.first << "--" << iter.second << " ";
            }
            std::cout << std::endl;
        }
    }
    return sub_graphs;
}

cpp::result<std::map<int, SubGraph>, Error> DivideGraph(Graph graph) {
    // map获取节点信息 device_in、device_out、前驱后继的节点以及节点数量 当前节点放置的位置
    std::map<std::string, Nodevalue> sub_graph_info;
    int node_num = graph.NodeMap().size();
    std::cout << "node_num" << node_num << std::endl;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> device_in_group;   // 记录所有标记device_in的算子，方便后期遍历
    std::vector<std::string> device_out_group;  // 记录所有标记device_out的算子，方便后期遍历
    for (int i = 0; i < node_num; i++) {
        NodeBase current_node = *graph.GetNode(i);
        Nodevalue current_node_value;
        current_node_value.node = current_node;
        inputs = current_node.Inputs();
        int inputs_num = inputs.size();
        outputs = current_node.Outputs();
        int outputs_num = outputs.size();
        current_node_value.inputs_num = inputs_num;
        current_node_value.outputs_num = outputs_num;
        // std::cout<<"inputs_num"<<inputs_num<<"outputs_num"<<outputs_num<<std::endl;

        std::string device = current_node.Device();
        // device_in
        for (int j = 0; j < inputs_num; j++) {
            // std::cout<<"device_in"<<std::endl;
            std::string input = inputs.at(j);
            NodeBase& before_node = *graph.GetNode(input);

            // std::cout<<j<<" before_node_name:"<<before_node.Name()<<std::endl;
            std::string before_device = before_node.Device();
            if (device == before_device)  // 同设备的算子记录一下
            {
                current_node_value.device_in_node.push_back(input);
            } else {
                current_node_value.device_in = true;
                device_out_group.push_back(input);
                std::cout << "device_out:" << input << std::endl;
            }
        }
        // device_out
        for (int j = 0; j < outputs_num; j++) {
            // std::cout<<"device_out"<<std::endl;
            std::string output = outputs.at(j);
            NodeBase& next_node = *graph.GetNode(output);
            std::string next_device = next_node.Device();
            if (device == next_device) {
                current_node_value.device_out_node.push_back(output);
            } else {
                current_node_value.device_out = true;
                device_in_group.push_back(output);
                std::cout << "device_in:" << output << std::endl;
            }
        }
        sub_graph_info.insert(std::pair<std::string, Nodevalue&>(current_node.Name(), current_node_value));
        inputs.clear();
        outputs.clear();
    }

    // 深度优先搜索切断同设备不满足条件的边
    TRY(SubgraphSearch(sub_graph_info, device_in_group, device_out_group));

    // 形成子图编号
    std::multimap<int, NodeBase> nodes_to_subgraph;
    int subgraph_num = CreateSubgraphNum(sub_graph_info, nodes_to_subgraph);

    std::map<int, SubGraph> sub_graph = CreateSubgraph(sub_graph_info, nodes_to_subgraph, subgraph_num);
    return sub_graph;
}
}  // namespace framework
