#include "fusion/aware_fusion.h"
#include "gtest/gtest.h"

namespace framework {
TEST(TestFusion, AwareFusion) {
    NodeBase node_1;
    node_1.Name("input");
    node_1.Device("dev0");
    node_1.ComputeCost(2);
    node_1.PersistentMemory(1);
    node_1.Outputs({"conv1", "conv2"});
    node_1.OutputMemory(10);

    NodeBase node_2;
    node_2.Name("conv1");
    node_2.Device("dev1");
    node_2.ComputeCost(3);
    node_2.Inputs({"input"});
    node_2.InputMemory(2);
    node_2.PersistentMemory(3);
    node_2.Outputs({"conv2"});
    node_2.OutputMemory(3);

    NodeBase node_3;
    node_3.Name("conv2");
    node_3.Device("dev2");
    node_3.ComputeCost(3);
    node_3.Inputs({"input", "conv1"});
    node_3.InputMemory(8);
    node_3.PersistentMemory(4);
    node_3.Outputs({"conv3"});
    node_3.OutputMemory(3);

    NodeBase node_4;
    node_4.Name("conv3");
    node_4.Device("dev0");
    node_4.ComputeCost(3);
    node_4.Inputs({"conv2"});
    node_4.InputMemory(3);
    node_4.PersistentMemory(4);
    node_4.Outputs({"relu"});
    node_4.OutputMemory(3);

    NodeBase node_5;
    node_5.Name("relu");
    node_5.Device("dev1");
    node_5.ComputeCost(3);
    node_5.Inputs({"conv3"});
    node_5.InputMemory(3);
    node_5.PersistentMemory(4);
    node_5.Outputs({"output"});
    node_5.OutputMemory(3);

    NodeBase node_6;
    node_6.Name("output");
    node_6.Device("dev3");
    node_6.ComputeCost(3);
    node_6.Inputs({"relu"});
    node_6.InputMemory(3);
    node_6.PersistentMemory(6);

    Graph graph;
    graph.AddNode(node_1);
    graph.AddNode(node_2);
    graph.AddNode(node_3);
    graph.AddNode(node_4);
    graph.AddNode(node_5);
    graph.AddNode(node_6);

    CostGraph cost_graph = ConvertGraphToCostGraph(graph);
    MergedCostGraph merged_cost_graph = InitMergedCostGraph(cost_graph);

    AwareFusion aware_fusion(cost_graph, merged_cost_graph);
    MergedCostGraph new_merged_cost_graph = aware_fusion.GenerateFusedGraph();

    EXPECT_EQ("input", merged_cost_graph.GetMergedCostNodes()[0].GetName());
    EXPECT_EQ("conv1", merged_cost_graph.GetMergedCostNodes()[1].GetName());
    EXPECT_EQ("conv2", merged_cost_graph.GetMergedCostNodes()[2].GetName());
    EXPECT_EQ("conv3", merged_cost_graph.GetMergedCostNodes()[3].GetName());
    EXPECT_EQ("relu", merged_cost_graph.GetMergedCostNodes()[4].GetName());
    EXPECT_EQ("output", merged_cost_graph.GetMergedCostNodes()[5].GetName());

    EXPECT_EQ(1, new_merged_cost_graph.GetMergedCostNodes().size());
    EXPECT_EQ(6, new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames().size());

    EXPECT_EQ("input", new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[0]);
    EXPECT_EQ("conv1", new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[1]);
    EXPECT_EQ("conv2", new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[2]);
    EXPECT_EQ("conv3", new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[3]);
    EXPECT_EQ("relu", new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[4]);
    EXPECT_EQ("output", new_merged_cost_graph.GetMergedCostNodes()[0].GetCostNodeNames()[5]);
}
}  // namespace framework