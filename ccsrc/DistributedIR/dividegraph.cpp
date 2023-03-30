
#include "DistributedIR/dividegraph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>

#include "DistributedIR/block.hpp"
#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "common/log.h"
#include "range/v3/view/filter.hpp"
#include "range/v3/view/transform.hpp"

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
                    spdlog::debug("{} -- {} : cut", before_node, current_node);
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
                    spdlog::debug("{} -- {} : cut", current_node, next_node);
                }
            }

        } else {
            return cpp::fail(Error(Kind::Invalid, "The device_out_flag of node is fault"));
        }
        // 在current_node的device_in_node和before_node的device_out_node中擦除需要切断的算子，方便组成子图
        for (auto& cut_edge : cut_edges) {
            std::string before_node = cut_edge.first;    // 在同设备后继中擦除current_node
            std::string current_node = cut_edge.second;  // 在同设备前驱中擦除before_node
            Nodevalue& before_node_value = sub_graph_info.find(before_node)->second;
            std::vector<std::string>& device_out_node = before_node_value.device_out_node;
            for (auto iter = device_out_node.begin(); iter != device_out_node.end();) {
                if (*iter == current_node) {
                    iter = device_out_node.erase(iter);
                } else {
                    iter++;
                }
            }
            Nodevalue& current_node_value = sub_graph_info.find(current_node)->second;
            std::vector<std::string>& device_in_node = current_node_value.device_in_node;
            for (auto iter = device_in_node.begin(); iter != device_in_node.end();) {
                if (*iter == before_node) {
                    iter = device_in_node.erase(iter);
                } else {
                    iter++;
                }
            }
        }
    }
    return {};
}

int CreateSubgraphNum(std::map<std::string, Nodevalue>& sub_graph_info,
                      std::map<int, std::set<NodePtr>>& nodes_to_subgraph) {
    int subgraph_num = -1;
    // 给算子子图编号
    for (auto& iter : sub_graph_info) {
        std::string current_node_name = iter.first;
        auto& current_node_value = iter.second;
        std::queue<std::string> joint_nodes;
        if (current_node_value.subgraph_num == -1) {
            joint_nodes.push(current_node_name);
            subgraph_num++;
            nodes_to_subgraph.insert({subgraph_num, {}});
        }
        while (!joint_nodes.empty()) {
            std::string the_node = joint_nodes.front();
            Nodevalue& the_node_value = sub_graph_info.find(the_node)->second;
            the_node_value.subgraph_num = subgraph_num;
            nodes_to_subgraph[subgraph_num].insert(the_node_value.node);
            // nodes_to_subgraph.insert({subgraph_num, the_node_value.node});
            std::vector<std::string> device_in_node = the_node_value.device_in_node;  // 当前算子还相连的前驱算子
            std::vector<std::string> device_out_node = the_node_value.device_out_node;  // 当前算子还相连的后继算子
            for (auto& in_op : device_in_node) {
                Nodevalue& in_op_value = sub_graph_info.find(in_op)->second;
                if (in_op_value.subgraph_num == -1) {
                    joint_nodes.push(in_op);
                }
            }
            for (auto& out_op : device_out_node) {
                Nodevalue& out_op_value = sub_graph_info.find(out_op)->second;
                if (out_op_value.subgraph_num == -1) {
                    joint_nodes.push(out_op);
                }
            }
            joint_nodes.pop();
        }
    }
    return subgraph_num + 1;
}

std::map<int, SubGraphPtr> CreateSubgraph(std::map<std::string, Nodevalue>& sub_graph_info,
                                          std::map<int, std::set<NodePtr>>& nodes_to_subgraph) {
    std::map<int, SubGraphPtr> sub_graphs;  // 所有的子图合集
    // 将算子放进对应的子图
    spdlog::debug("subgraph size: {} {}", nodes_to_subgraph.size());
    for (auto& index_nodes : nodes_to_subgraph) {
        spdlog::debug("subgraph {} node size: {}", index_nodes.first, index_nodes.second.size());

        SubGraph sub_graph;
        // std::map<std::string, NodeBase&> node_map;
        // auto node_to_place = nodes_to_subgraph.find(i);
        for (const auto& i : index_nodes.second) {
            spdlog::debug("subgraph {} add: {} {}", index_nodes.first, i->Name(), i);
            sub_graph.AddNode(i);
            // nodes_to_subgraph.erase(node_to_place);
            // node_to_place = nodes_to_subgraph.find(i);
        }
        // while (node_to_place != nodes_to_subgraph.end()) {
        //     std::cout << "subgraph" << i << "add:" << node_to_place->second->Name() << std::endl;
        //     sub_graph.AddNode(node_to_place->second);
        //     nodes_to_subgraph.erase(node_to_place);
        //     node_to_place = nodes_to_subgraph.find(i);
        // }
        sub_graphs.insert({index_nodes.first, std::make_shared<SubGraph>(std::move(sub_graph))});
    }
    if (nodes_to_subgraph.size() != sub_graphs.size()) {
        spdlog::debug("subgraph_num is {}", nodes_to_subgraph.size());
        spdlog::debug("subgraph's size is {}", sub_graphs.size());
        spdlog::debug("error:they are different!!!");
    }

    // //获取子图连接信息
    for (auto& iter : sub_graphs) {
        auto& current_sub_graph = iter.second;
        // auto current_sub_graph_num = iter.first;
        auto current_nodes = current_sub_graph->Nodes();
        // map<前驱的子图信息int 对应的节点输入map<string string>>
        std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>
            subgraph_op_input;  // before_node 前驱节点中第几个输出
        std::map<int, std::vector<std::pair<StrAndInt, StrAndInt>>>
            subgraph_op_output;  // next_node 后继节点中第几个输入
        // 获得前子图，获得后子图
        // 获得图input   格式：  before节点名:输出index
        // 获得图output  格式：  current节点名:输入index  input_data排序得出
        for (auto& i : current_nodes)  // 遍历子图中的所有节点
        {
            auto& current_node = i;
            Nodevalue& current_node_value = sub_graph_info.find(current_node->Name())->second;
            auto device_in_node = current_node_value.device_in_node;
            auto device_out_node = current_node_value.device_out_node;
            auto inputs = current_node->Inputs();
            auto outputs = current_node->Outputs();
            std::vector<std::string> inputs_diff;
            std::vector<std::string> outputs_diff;  // 差集及为前驱后继不同子图的算子
            std::set_difference(inputs.begin(), inputs.end(), device_in_node.begin(), device_in_node.end(),
                                inserter(inputs_diff, inputs_diff.begin()),
                                [](auto& a, auto& b) { return a != b; });  // old-->new需要删除的

            std::set_difference(outputs.begin(), outputs.end(), device_out_node.begin(), device_out_node.end(),
                                inserter(outputs_diff, outputs_diff.begin()),
                                [](auto& a, auto& b) { return a != b; });  // old-->new需要删除的
            spdlog::debug(
                "current node: {} device_in_node: {} inputs:{} inputs_diff: {} device_out_node: {} outputs: {} "
                "outputs_diff:{} subgraph_num: {}",
                current_node->Name(), device_in_node, inputs, inputs_diff, device_out_node, outputs, outputs_diff,
                current_node_value.subgraph_num);

            if (!inputs_diff.empty()) {
                // 该节点是图边缘节点
                // 找寻非同设备前驱的节点，前驱的output_data和该节点的input_data对上的部分放进第一个string
                std::multimap<std::string, std::string> data2data;
                auto& current_inputs_data = current_node->InputPorts();
                for (auto& input_data : current_inputs_data) {
                    for (auto& before_node : inputs_diff) {
                        auto before_node_value = sub_graph_info.find(before_node)->second;
                        auto& before_node_real = before_node_value.node;
                        auto& before_outputs_data = before_node_real->OutputPorts();
                        auto iter = std::find_if(
                            before_outputs_data.begin(), before_outputs_data.end(), [&](EdgePort<AbstractTensor>& i) {
                                return before_node_real->OutputName(i.index) == input_data.entity.Ref();
                            });
                        // detect tensor connect
                        if (iter != before_outputs_data.end()) {
                            // 放进map1<input_data, currentnode.name:input_index>
                            //  上个节点的第几个输出  当前节点的第几个输入
                            auto r = current_node->InputName(input_data.index);
                            assert(r.has_value());
                            // data2data.insert({input_data.entity.Ref(), r.value()});
                            auto data2data = subgraph_op_input.find(before_node_value.subgraph_num);
                            // 放进subgraph_op_input<前序node所在子图序号，tensor connect>
                            if (data2data == subgraph_op_input.end()) {
                                std::vector<std::pair<StrAndInt, StrAndInt>> m;
                                m.emplace_back(input_data.entity.Ref(), r.value());
                                subgraph_op_input.insert({before_node_value.subgraph_num, m});
                            } else {
                                data2data->second.emplace_back(input_data.entity.Ref(), r.value());
                            }
                        }
                    }
                }
                // 放进subgraph_op_input<当前node所在map的序号，map1>
                // subgraph_op_input.insert({current_node_value.subgraph_num, data2data});
            }
            if (!outputs_diff.empty()) {
                std::multimap<std::string, std::string> data2data;
                auto& current_outputs_data = current_node->OutputPorts();
                for (auto& output_data : current_outputs_data) {
                    for (auto& next_node : outputs_diff) {
                        auto next_node_value = sub_graph_info.find(next_node)->second;
                        auto& next_node_real = next_node_value.node;
                        auto next_inputs_data = next_node_real->InputPorts();
                        auto iter =
                            std::find_if(next_inputs_data.begin(), next_inputs_data.end(), [&](EdgePort<InputStr>& i) {
                                return current_node->OutputName(output_data.index) == i.entity.Ref();
                            });
                        // input_index = std::distance(next_inputs_data.begin(), iter);
                        // detect tensor connect
                        if (iter != next_inputs_data.end()) {
                            // 这个节点的哪个输出 下个节点的第几个输入
                            auto r = next_node_real->InputName(iter->index);
                            assert(r.has_value());
                            // data2data.insert({iter->entity.Ref(), r.value()});
                            auto data2data = subgraph_op_output.find(next_node_value.subgraph_num);
                            // 放进subgraph_op_input<前序node所在子图序号，tensor connect>
                            if (data2data == subgraph_op_output.end()) {
                                std::vector<std::pair<StrAndInt, StrAndInt>> m;
                                m.emplace_back(iter->entity.Ref(), r.value());
                                subgraph_op_output.insert({next_node_value.subgraph_num, m});
                            } else {
                                data2data->second.emplace_back(iter->entity.Ref(), r.value());
                            }
                        }
                    }
                }
                // subgraph_op_output.insert({current_node_value.subgraph_num, data2data});
            }
        }
        for (auto& before : subgraph_op_input) {
            int before_subgraph_num = before.first;
            auto& op_op = before.second;
            auto& before_subgraph = sub_graphs.find(before_subgraph_num)->second;
            current_sub_graph->AddInputGraph(before_subgraph);
            current_sub_graph->AddInput(op_op);
        }

        for (auto& next : subgraph_op_output) {
            int next_subgraph_num = next.first;
            auto& op_op = next.second;
            auto& next_subgraph = sub_graphs.find(next_subgraph_num)->second;
            current_sub_graph->AddOutputGraph(next_subgraph);
            current_sub_graph->AddOutput(op_op);
        }
    }
    // 输出子图信息
    for (auto& iter : sub_graphs) {
        std::stringstream ss;
        ss << "======================================" << std::endl;
        int subgraph_num = iter.first;
        auto& current_subgraph = iter.second;
        auto nodes_list = current_subgraph->Nodes();
        ss << "The subgraph num is " << subgraph_num << ". This subgraph contains " << nodes_list.size()
           << " nodes.They are :";
        for (auto& i : nodes_list) {
            NodeBase& the_node = *i;
            ss << the_node.Name() << " ";
        }
        ss << std::endl;

        // std::cout<<"The number of output subgraphs is :"<<current_subgraph.output_graphs.size()<<std::cout;
        ss << "before_op_output--current_subgraph_input are :" << std::endl;
        for (auto& op_op : current_subgraph->GetInputs()) {
            for (auto& iter : op_op) {
                ss << iter.first << "--" << iter.second << " ";
            }
            ss << std::endl;
        }
        ss << "current_subgraph_output--next_op_input are :" << std::endl;
        for (auto& op_op : current_subgraph->GetOutputs()) {
            for (auto& iter : op_op) {
                ss << iter.first << "--" << iter.second << " ";
            }
            ss << std::endl;
        }
        spdlog::debug(ss.str());
    }
    return sub_graphs;
}

std::set<NodePtr> CheckCircleAndSplit(std::map<std::string, Nodevalue>& sub_graph_info, std::set<NodePtr>& nodes) {
    spdlog::debug("====================Start Check Circle===========================");
    // std::set<NodePtr> nodes_set = std::set<NodePtr>(nodes.begin(), nodes.end());
    std::set<NodePtr> ret;
    auto in_filter = [&](auto& n) {
        return sub_graph_info[n->Name()].device_in;
    };
    auto out_filter = [&](auto& n) {
        return sub_graph_info[n->Name()].device_out;
    };
    auto device_in_nodes = nodes | ranges::views::filter(in_filter) | ranges::to_vector;
    auto device_out_nodes = nodes | ranges::views::filter(out_filter) | ranges::to_vector;
    auto device_out_nodes_size = device_out_nodes.size();
    std::queue<NodePtr> q;
    for (auto& i : device_in_nodes) {
        q.push(i);
    }
    // std::vector<NodePtr> device_in_successor;
    std::map<NodePtr, bool> visited;
    while (!q.empty()) {
        auto& node = q.front();
        visited[node] = true;
        ret.insert(node);
        device_out_nodes =
            device_out_nodes | ranges::views::remove_if([&](auto& i) { return i == node; }) | ranges::to_vector;
        auto successor = node->Outputs();
        for (auto& i : successor) {
            auto n = sub_graph_info[i].node;
            auto it = visited.find(n);
            auto same_graph = nodes.find(n) != nodes.end();
            // auto same_graph = true;
            if (same_graph && (it == visited.end() || !it->second)) {
                q.push(n);
            }
        }
        q.pop();
    }
    // exists device_out node which is not device_in's successor
    // may cause cycle
    // split these node to another graph
    if (!device_out_nodes.empty() && device_out_nodes_size > 0) {
        spdlog::debug("DetectCycle: {}", device_out_nodes);
        spdlog::debug("nodes size: {}", ret.size());
        spdlog::debug("====================End Check Circle===========================");
        return ret;
    }
    spdlog::debug("====================End Check Circle===========================");
    return {};
}

void FixCircle(std::map<std::string, Nodevalue>& sub_graph_info, std::map<int, std::set<NodePtr>>& nodes_to_subgraph) {
    std::vector<std::set<NodePtr>> splited_graph;
    for (auto& i : nodes_to_subgraph) {
        // auto subgraph_order = i.first;
        auto ret = CheckCircleAndSplit(sub_graph_info, i.second);
        for (const auto& j : ret) {
            i.second.erase(j);
        }
        if (!ret.empty()) {
            splited_graph.push_back(ret);
        }
        // i.second.erase()
        // if (!ret.empty()) {
        //     i.second = i.second | ranges::views::remove_if([&](auto& i) { return ranges::find(ret, i) != ret.end();
        //     })
        //                | ranges::to_vector;
        // }
    }
    for (auto& i : splited_graph) {
        std::set<NodePtr> split_node_set = std::set<NodePtr>(i.begin(), i.end());
        auto size = nodes_to_subgraph.size();
        nodes_to_subgraph[size] = i;
        for (const auto& node : nodes_to_subgraph[size]) {
            auto& node_value = sub_graph_info[node->Name()];
            node_value.subgraph_num = size;
            // previous node not in split graph
            spdlog::debug(
                "PROCESS SPLITED NODE -- name:{} device_in:{} device_in_node:{} device_out:{} device_out_node:{} {}",
                node->Name(), node_value.device_in, node_value.device_in_node, node_value.device_out,
                node_value.device_out_node, node);
            for (auto& input : node->Inputs()) {
                auto& previous_value = sub_graph_info[input];
                if (split_node_set.find(sub_graph_info[input].node) == split_node_set.end()) {
                    spdlog::debug("PROCESS SPLITED NODE -- FIND DIFFERENT GRAPH node:{}", input);
                    node_value.device_in = true;
                    node_value.device_in_node = node_value.device_in_node
                                                | ranges::views::remove_if([&](auto& i) { return i == input; })
                                                | ranges::to_vector;
                    previous_value.device_out = true;
                    previous_value.device_out_node =
                        previous_value.device_out_node
                        | ranges::views::remove_if([&](auto& i) { return i == node->Name(); }) | ranges::to_vector;
                } else {
                    node_value.device_in_node.push_back(input);
                    previous_value.device_out_node.push_back(node->Name());
                }
            }
            // // successor node not in split grpah
            // for (auto& output : node->Outputs()) {
            //     if (split_node_set.find(sub_graph_info[output].node) == split_node_set.end()) {
            //         node_value.device_out = true;
            //         node_value.device_out_node = node_value.device_out_node
            //                                      | ranges::views::remove_if([&](auto& i) { return i == output; })
            //                                      | ranges::to_vector;
            //         auto& next_value = sub_graph_info[output];
            //         next_value.device_in = true;
            //         next_value.device_in_node = next_value.device_in_node
            //                                     | ranges::views::remove_if([&](auto& i) { return i == node->Name();
            //                                     }) | ranges::to_vector;
            //     }
            // }
        }
        spdlog::debug("add graph: id:{} node number {} | nodes: {}", size, i.size(), i);
    }
}

cpp::result<std::map<int, SubGraphPtr>, Error> DivideGraph(Graph graph) {
    // map获取节点信息 device_in、device_out、前驱后继的节点以及节点数量 当前节点放置的位置
    std::map<std::string, Nodevalue> sub_graph_info;
    int node_num = graph.NodeMap().size();
    spdlog::debug("node_num: {}", node_num);
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> device_in_group;   // 记录所有标记device_in的算子，方便后期遍历
    std::vector<std::string> device_out_group;  // 记录所有标记device_out的算子，方便后期遍历
    for (int i = 0; i < node_num; i++) {
        auto current_node = graph.GetNode(i).value();
        Nodevalue current_node_value;
        current_node_value.node = current_node;
        inputs = current_node->Inputs();
        int inputs_num = inputs.size();
        outputs = current_node->Outputs();
        int outputs_num = outputs.size();
        current_node_value.inputs_num = inputs_num;
        current_node_value.outputs_num = outputs_num;
        // std::cout<<"inputs_num"<<inputs_num<<"outputs_num"<<outputs_num<<std::endl;

        std::string device = current_node->Device();
        // device_in
        for (int j = 0; j < inputs_num; j++) {
            // std::cout<<"device_in"<<std::endl;
            std::string input = inputs.at(j);
            auto& before_node = graph.GetNode(input).value();

            // std::cout<<j<<" before_node_name:"<<before_node.Name()<<std::endl;
            std::string before_device = before_node->Device();
            if (device == before_device)  // 同设备的算子记录一下
            {
                current_node_value.device_in_node.push_back(input);
            } else {
                current_node_value.device_in = true;
                device_out_group.push_back(input);
                spdlog::debug("device_out: {}", input);
            }
        }
        // device_out
        for (int j = 0; j < outputs_num; j++) {
            // std::cout<<"device_out"<<std::endl;
            std::string output = outputs.at(j);
            auto& next_node = graph.GetNode(output).value();
            std::string next_device = next_node->Device();
            if (device == next_device) {
                current_node_value.device_out_node.push_back(output);
            } else {
                current_node_value.device_out = true;
                device_in_group.push_back(output);
                spdlog::debug("device_in: {}", output);
            }
        }
        sub_graph_info.insert({current_node->Name(), current_node_value});
        inputs.clear();
        outputs.clear();
    }

    // 深度优先搜索切断同设备不满足条件的边
    TRY(SubgraphSearch(sub_graph_info, device_in_group, device_out_group));

    // 形成子图编号
    // std::multimap<int, NodePtr> nodes_to_subgraph;
    std::map<int, std::set<NodePtr>> nodes_to_subgraph;
    CreateSubgraphNum(sub_graph_info, nodes_to_subgraph);
    FixCircle(sub_graph_info, nodes_to_subgraph);
    std::map<int, SubGraphPtr> sub_graph = CreateSubgraph(sub_graph_info, nodes_to_subgraph);
    return sub_graph;
}

}  // namespace framework
