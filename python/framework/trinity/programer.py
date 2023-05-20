# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Controller Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict


class Controller:
    """Controller class."""

    def __init__(self, nx_graph, cluster):
        """Controller class初始化.

        Args:
          nx_graph: 需要被放置的nx_graph
          cluster: 要优化的硬件设备集群
        """
        self.nx_graph = nx_graph
        self._fanout = defaultdict(lambda: [])
        for node in self.nx_graph.nodes():
            self._fanout[node].append(self.nx_graph.nodes[node])
        # 将重要的操作采用拓扑排序的方式挑选出来放到列表中，等待放置
        # 顺序. 这个集合的顺序是确定的.
        self.important_ops = []
        self.important_op_names = []
        for node in self.nx_graph.nodes():
            self.important_ops.append(self.nx_graph.nodes[node])
            self.important_op_names.append(node)

        for node in self.nx_graph.nodes():
            outedges = self.nx_graph.out_edges(node)
            output_memorys = []
            for edge in outedges:
                output_memorys.append(self.nx_graph[edge[0]][edge[1]]["weight"])
            self.node_properties = output_memorys

        self.cluster = cluster
        self.devices = cluster.ListDevices()

        """返回一个硬托管约束的列表

        要使模型工作，一个元组中的所有节点必须放在同一个设备上

        Returns:
          一系列约束元祖
        """

        # 重要op名称的列表，以不确定的顺序
        self.important_op_names = frozenset(self.important_op_names)

    @property
    def num_devices(self):
        return len(self.devices)

    def get_node_fanout(self, node):
        return self._fanout[node.name]

    # Get the nodes in the immediate fanin of node.
    # Beware: this doesn't take into account the nodes that may be skipped
    # since placement constraints force their placement.
