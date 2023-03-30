#include <iostream>
#include <map>
#include <string>

#include "DistributedIR/block.hpp"
#include "DistributedIR/dividegraph.hpp"
#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "gtest/gtest.h"
namespace framework {
// Demonstrate some basic assertions.
TEST(TestDistributedIR, DeviceGraph) {
    DeviceGraph graph("graph");
    Block block1("dev_1");
    block1.AddInputPort(0, 0, framework::DataType::U8, {});
    block1.AddInputPort(0, 1, framework::DataType::U8, {});
    block1.AddInputPort(0, 2, framework::DataType::U8, {});
    EXPECT_EQ(block1.inputs.size(), 3);
    block1.AddOutputPort(DataType::U8, {});
    EXPECT_EQ(block1.outputs.size(), 1);

    Block block2("dev_2");
    block2.AddInputPort(block1.Id(), 3, framework::DataType::U8, {});
    block2.AddInputPort(0, 4, framework::DataType::U8, {});
    EXPECT_EQ(block2.inputs.size(), 2);
    block2.AddOutputPort(DataType::U8, {});
    EXPECT_EQ(block2.outputs.size(), 1);

    graph.AddBlock(block1);
    graph.AddBlock(block2);
    EXPECT_EQ(graph.blocks.size(), 2);
    graph.Connect(0, 0, 1, 0);
    graph.BuildPorts();
    EXPECT_EQ(graph.inputs.size(), 4);
    EXPECT_EQ(graph.outputs.size(), 1);

    graph.blocks[0].AddOutputPort(DataType::U8, {});
    graph.Connect(0, 1, 1, 1);
    graph.BuildPorts();

    EXPECT_EQ(graph.inputs.size(), 3);
    EXPECT_EQ(graph.outputs.size(), 1);

    graph.blocks[1].AddOutputPort(DataType::U8, {});
    graph.BuildPorts();

    EXPECT_EQ(graph.inputs.size(), 3);
    EXPECT_EQ(graph.outputs.size(), 2);

    //================================================

    DeviceGraph graph1("graph1");
    Block graph1_block1("dev_1");
    graph1_block1.AddInputPort(0, 0, framework::DataType::U8, {});
    graph1_block1.AddInputPort(0, 1, framework::DataType::U8, {});
    graph1_block1.AddInputPort(0, 2, framework::DataType::U8, {});
    EXPECT_EQ(graph1_block1.inputs.size(), 3);
    graph1_block1.AddOutputPort(DataType::U8, {});
    EXPECT_EQ(graph1_block1.outputs.size(), 1);

    Block graph1_block2("dev_2");
    graph1_block2.AddInputPort(0, 0, framework::DataType::U8, {});
    graph1_block2.AddInputPort(0, 1, framework::DataType::U8, {});
    EXPECT_EQ(graph1_block2.inputs.size(), 2);
    graph1_block2.AddOutputPort(DataType::U8, {});
    EXPECT_EQ(graph1_block2.outputs.size(), 1);

    graph1.AddBlock(graph1_block1);
    graph1.AddBlock(graph1_block2);
    EXPECT_EQ(graph1.blocks.size(), 2);
    graph1.Connect(0, 0, 1, 0);
    graph1.BuildPorts();
    EXPECT_EQ(graph1.inputs.size(), 4);
    EXPECT_EQ(graph1.outputs.size(), 1);

    ServerGraph server_graph("server");
    server_graph.AddDeviceGraph(graph);
    server_graph.AddDeviceGraph(graph1);

    server_graph.Connect(0, 0, 1, 0);
    server_graph.Connect(0, 1, 1, 1);

    server_graph.BuildPorts();

    EXPECT_EQ(server_graph.inputs.size(), 5);
    EXPECT_EQ(server_graph.outputs.size(), 1);

    //=============================================

    DeviceGraph graph2("graph2");
    Block graph2_block1("dev_1");
    graph2_block1.AddInputPort(0, 0, framework::DataType::U8, {});
    graph2_block1.AddInputPort(0, 1, framework::DataType::U8, {});
    graph2_block1.AddInputPort(0, 2, framework::DataType::U8, {});
    EXPECT_EQ(graph2_block1.inputs.size(), 3);
    graph2_block1.AddOutputPort(DataType::U8, {});
    EXPECT_EQ(graph2_block1.outputs.size(), 1);

    Block graph2_block2("dev_2");
    graph2_block2.AddInputPort(0, 0, framework::DataType::U8, {});
    graph2_block2.AddInputPort(0, 1, framework::DataType::U8, {});
    EXPECT_EQ(graph2_block2.inputs.size(), 2);
    graph2_block2.AddOutputPort(DataType::U8, {});
    EXPECT_EQ(graph2_block2.outputs.size(), 1);

    graph2.AddBlock(graph2_block1);
    graph2.AddBlock(graph2_block2);
    EXPECT_EQ(graph2.blocks.size(), 2);
    graph2.Connect(0, 0, 1, 0);
    graph2.BuildPorts();
    EXPECT_EQ(graph2.inputs.size(), 4);
    EXPECT_EQ(graph2.outputs.size(), 1);

    ServerGraph server_graph1("server2");
    server_graph1.AddDeviceGraph(graph2);

    server_graph1.BuildPorts();

    EXPECT_EQ(server_graph1.inputs.size(), 4);
    EXPECT_EQ(server_graph1.outputs.size(), 1);

    ClusterGraph cluster_graph("cluster");
    cluster_graph.AddServerGraph(server_graph);
    cluster_graph.AddServerGraph(server_graph1);

    cluster_graph.Connect(0, 0, 1, 0);
    cluster_graph.BuildPorts();

    EXPECT_EQ(cluster_graph.inputs.size(), 8);
    EXPECT_EQ(cluster_graph.outputs.size(), 1);
}
TEST(TestDistributedIR, DivideGraph2SubGraph) {
    //================================================
    Graph graph3;

    NodeBase node_1;
    node_1.Name("input");
    node_1.Device("dev0");
    node_1.ComputeCost(2);
    node_1.Inputs({});
    node_1.PersistentMemory(1);
    node_1.Outputs({"conv1", "conv3"});
    node_1.AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(DataType::U8, {}), 0)).expect("error");
    node_1.OutputsNum(1);

    NodeBase node_2;
    node_2.Name("conv1");
    node_2.Device("dev1");
    node_2.ComputeCost(3);
    node_2.Inputs({"input"});
    node_2.PersistentMemory(3);
    node_2.Outputs({"conv2"});
    node_2.AddInputPort("input_0", 0, 0, DataType::U8, {}).expect("error");
    node_2.AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(DataType::U8, {}), 0)).expect("error");
    node_2.OutputsNum(1);

    NodeBase node_3;
    node_3.Name("conv2");
    node_3.Device("dev1");
    node_3.ComputeCost(3);
    node_3.Inputs({"conv1"});
    node_3.PersistentMemory(3);
    node_3.Outputs({"output"});
    node_3.AddInputPort("conv1", 0, 0, DataType::U8, {}).expect("error");
    node_3.AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(DataType::U8, {}), 0)).expect("error");
    node_3.OutputsNum(1);

    NodeBase node_4;
    node_4.Name("conv3");
    node_4.Device("dev0");
    node_4.ComputeCost(3);
    node_4.Inputs({"input"});
    node_4.PersistentMemory(3);
    node_4.Outputs({"output"});
    node_4.AddInputPort("input_0", 0, 0, DataType::U8, {}).expect("error");
    node_4.AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(DataType::U8, {}), 0)).expect("error");
    node_4.OutputsNum(1);

    NodeBase node_5;
    node_5.Name("conv5");
    node_5.Device("dev0");
    node_5.ComputeCost(3);
    node_5.Inputs({});
    node_5.PersistentMemory(2);
    node_5.Outputs({"conv6"});
    node_5.AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(DataType::U8, {}), 0)).expect("error");
    node_5.OutputsNum(1);

    NodeBase node_6;
    node_6.Name("conv6");
    node_6.Device("dev0");
    node_6.ComputeCost(3);
    node_6.Inputs({"conv5"});
    node_6.PersistentMemory(2);
    node_6.Outputs({"output"});
    node_6.AddInputPort("conv5", 0, 0, DataType::U8, {}).expect("error");
    node_6.AddOutputPort(EdgePort<AbstractTensor>(AbstractTensor(DataType::U8, {}), 0)).expect("error");
    node_6.OutputsNum(1);

    NodeBase node_7;
    node_7.Name("output");
    node_7.Device("dev0");
    node_7.ComputeCost(3);
    node_7.Inputs({"conv2", "conv3", "conv6"});
    node_7.PersistentMemory(2);
    node_7.Outputs({});
    node_7.AddInputPort("conv2", 0, 0, DataType::U8, {}).expect("error");
    node_7.AddInputPort("conv3", 0, 1, DataType::U8, {}).expect("error");
    node_7.AddInputPort("conv6", 0, 2, DataType::U8, {}).expect("error");
    node_7.OutputsNum(0);

    graph3.AddNode(node_1);
    graph3.AddNode(node_2);
    graph3.AddNode(node_3);
    graph3.AddNode(node_4);
    graph3.AddNode(node_5);
    graph3.AddNode(node_6);
    graph3.AddNode(node_7);
    EXPECT_NO_THROW(std::cout << graph3 << std::endl);
    EXPECT_NO_THROW(std::cout << std::make_shared<Graph>(graph3) << std::endl);
    auto r = DivideGraph(graph3);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value().size(), 4);
}
}  // namespace framework
