#include <gtest/gtest.h>
// #include "add.h"
#define private public
#include <iostream>

#include "DistributedIR/block.hpp"
#include "DistributedIR/op.hpp"
namespace framework {
// Demonstrate some basic assertions.
TEST(TestDistributedIR, DeviceGraph) {
    DeviceGraph graph("graph");
    Block block1("graph/b1", "dev_1", SubGraph());
    block1.AddInputPort(EdgePort<std::string>("input_1"));
    block1.AddInputPort(EdgePort<std::string>("input_2"));
    block1.AddInputPort(EdgePort<std::string>("input_3"));
    EXPECT_EQ(block1.inputs.size(), 3);
    block1.AddOutputPort(EdgePort<std::string>("output_1"));
    EXPECT_EQ(block1.outputs.size(), 1);

    Block block2("graph/b2", "dev_2", SubGraph());
    block2.AddInputPort(EdgePort<std::string>("input_1"));
    block2.AddInputPort(EdgePort<std::string>("input_2"));
    EXPECT_EQ(block2.inputs.size(), 2);
    block2.AddOutputPort(EdgePort<std::string>("output_1"));
    EXPECT_EQ(block2.outputs.size(), 1);

    graph.AddBlock(block1);
    graph.AddBlock(block2);
    EXPECT_EQ(graph.blocks.size(), 2);
    graph.Connect(0, 0, 1, 0);
    graph.BuildPorts();

    EXPECT_EQ(graph.inputs.size(), 4);
    EXPECT_EQ(graph.outputs.size(), 1);

    graph.blocks[0].AddOutputPort(EdgePort<std::string>("output_2"));
    graph.Connect(0, 1, 1, 1);
    graph.BuildPorts();

    EXPECT_EQ(graph.inputs.size(), 3);
    EXPECT_EQ(graph.outputs.size(), 1);

    graph.blocks[1].AddOutputPort(EdgePort<std::string>("output_2"));
    graph.BuildPorts();

    EXPECT_EQ(graph.inputs.size(), 3);
    EXPECT_EQ(graph.outputs.size(), 2);

    //================================================

    DeviceGraph graph1("graph1");
    Block graph1_block1("graph1/b1", "dev_1", SubGraph());
    graph1_block1.AddInputPort(EdgePort<std::string>("input_1"));
    graph1_block1.AddInputPort(EdgePort<std::string>("input_2"));
    graph1_block1.AddInputPort(EdgePort<std::string>("input_3"));
    EXPECT_EQ(graph1_block1.inputs.size(), 3);
    graph1_block1.AddOutputPort(EdgePort<std::string>("output_1"));
    EXPECT_EQ(graph1_block1.outputs.size(), 1);

    Block graph1_block2("graph1/b2", "dev_2", SubGraph());
    graph1_block2.AddInputPort(EdgePort<std::string>("input_1"));
    graph1_block2.AddInputPort(EdgePort<std::string>("input_2"));
    EXPECT_EQ(graph1_block2.inputs.size(), 2);
    graph1_block2.AddOutputPort(EdgePort<std::string>("output_1"));
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
    Block graph2_block1("graph2/b1", "dev_1", SubGraph());
    graph2_block1.AddInputPort(EdgePort<std::string>("input_1"));
    graph2_block1.AddInputPort(EdgePort<std::string>("input_2"));
    graph2_block1.AddInputPort(EdgePort<std::string>("input_3"));
    EXPECT_EQ(graph2_block1.inputs.size(), 3);
    graph2_block1.AddOutputPort(EdgePort<std::string>("output_1"));
    EXPECT_EQ(graph2_block1.outputs.size(), 1);

    Block graph2_block2("graph2/b2", "dev_2", SubGraph());
    graph2_block2.AddInputPort(EdgePort<std::string>("input_1"));
    graph2_block2.AddInputPort(EdgePort<std::string>("input_2"));
    EXPECT_EQ(graph2_block2.inputs.size(), 2);
    graph2_block2.AddOutputPort(EdgePort<std::string>("output_1"));
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
}  // namespace framework
