"""
Filename: tf_rew.py
"""
from collections import defaultdict
from dataclasses import dataclass
import heapq


def get_op_costs(nx_graph):
    """
    获取op_costs
    """
    cost_d = {}

    for node in nx_graph.nodes():
        cost_d[node] = nx_graph.nodes[node]["compute_cost"]

    return cost_d


@dataclass
class Node:
    """
    Node类
    """

    op_name: str
    device: str
    children: list
    compute_cost: int
    parents: list
    output_memory: list


class SimQueue:
    """
    SimQueue类
    """

    def __init__(self):
        self.queue = []

    def put(self, x):
        """
        put方法
        """
        heapq.heappush(self.queue, x)

    def empty(self):
        """
        empty方法
        """
        return len(self.queue) == 0

    def get(self):
        """
        get方法
        """
        return heapq.heappop(self.queue)


class Simulator:
    """Simulator class"""

    default_params = {
        # wait for delta1 after an op is done
        "delta1": 5.7,  # us 5.7
        # constant time overhead for transfer
        "delta2": 25,  # us 25
        "init_offset": 0,  # us
        "transfer_speed": 9200,  # bytes/us 7600
    }

    def __init__(self, nx_graph, cost_dict, output_dict, devices):
        """
        Init for Simulator class
        Args:
          nx_graph: nx_graph to simulate
          cost_dict: contains run_time of nodes in microseconds节点的运行时间
          output_dict: contains output sizes of nodes节点输出的数据大小
          devices: list of device names一系列设备的名称
          params: dictionary of parameters to use in simulator需要在模拟器中使用的参数字典
            some parameters can be missing
        """
        self.nx_graph = nx_graph
        self.cost_d = self.cost_dict = defaultdict(int, cost_dict)
        self.out_d = self.output_dict = defaultdict(list, output_dict)
        self.bus = "/bus"
        self.devices = devices
        self.params = self.default_params
        self.node_dict = self.get_attributes()

        # Make a parent_map : node_name -> bool map of parents 当前节点有父节点，则变成true
        self.parent_map = defaultdict(dict)
        for k, v in self.node_dict.items():
            for p in v.parents:
                self.parent_map[k][p] = True

        self.child_map = defaultdict(dict)
        for k, v in self.node_dict.items():
            for p in v.children:
                self.child_map[k][p] = True

        # 此处尝试处理通信量

    def get_attributes(self):
        """
        # 相当于是在对当前节点所有属性，以便后续进行编码
        Creates the node_dict. Node contains the following
        Attributes
          op_name: name of op 操作的名称
          device: device of the node 节点的设备
          compute_cost: run time in ns 计算运行时间
          output_memory: list of output sizes 输出尺寸
          parents: set of parents 父节点
          children: dict from node_name to output_index 子节点
        """
        # Create a dict from node_name -> Node
        node_dicts = {}
        for node in self.nx_graph.nodes():
            outedges = self.nx_graph.out_edges(node)
            inedges = self.nx_graph.in_edges(node)
            output_memorys = []
            parents = []
            children = []
            for edge in outedges:
                output_memorys.append(self.nx_graph[edge[0]][edge[1]]["weight"])
                children.append(edge[1])
            for edge in inedges:
                parents.append(edge[0])
            node_dicts[node] = Node(
                op_name=node,
                device=self.nx_graph.nodes[node]["device"],
                compute_cost=self.nx_graph.nodes[node]["compute_cost"],
                children=children,
                output_memory=output_memorys,
                parents=parents,
            )
        return node_dicts

    def is_scheduleable(self, k):
        """
        判断是否可调度
        """
        for v in self.parent_map[k].values():
            if v:
                return False
        return True

    def simulate(self, pl):
        """
        Run the simulation
        Args:
          pl: Contains mapping from device_name to device
            May be incomplete. Default mapping used if incomplete
        Return:
          tuple of (run_time, node_dict)
        """

        i, run_time, q, f = 0, 0, SimQueue(), self.node_dict
        # q has event (op/tranfer/remove_dependency) done events
        all_dev = self.devices + [self.bus + dev for dev in self.devices]
        # device_in_queue is basically is_device_busy map right now map
        device_in_queue, device_queue = dict((dev, False) for dev in all_dev), dict(
            (dev, SimQueue()) for dev in all_dev
        )
        # device_queue holds the currently runnable nodes waiting for device to get free

        # Reset parent_map
        for _, v in self.parent_map.items():
            for p in v.keys():
                v[p] = True

        # get the device for node k
        def get_dev(k):
            """
            获取设备数
            """
            if k in pl:
                return pl[k]
            return self.node_dict[k].device

        def add_to_dev_queue(t, op, dev, element):
            """
            添加设备到队列
            """
            nonlocal i
            i += 1
            device_queue[dev].put(((t, i), element))
            if not device_in_queue[dev]:
                q.put((t, op, dev))
                device_in_queue[dev] = True

        # Runs the next job on device
        def run_dev(dev):
            """
            rundev
            """
            p, node_name = device_queue[dev].get()
            assert p[0] <= t, f"Priority after exec time, p={p[0]}, t={t}"
            node = self.node_dict[node_name]
            # Compute start and end times

            # f[node_name] = self.Node()
            f[node_name].start_time = t
            f[node_name].end_time = t + node.compute_cost
            f[node_name].device = dev

            # Schedule when device is free again
            delta = 0 if node.compute_cost == 0 else self.params["delta1"]
            q.put((f[node_name].end_time + delta, "run_dev", dev))

            # Find which all output indices require bus
            require_bus = defaultdict(list)  # output_index to list of children
            for o, c in enumerate(node.children):
                if dev == get_dev(c):
                    q.put((f[node_name].end_time, "remove_dependency", (node_name, c)))
                else:
                    require_bus[o].append(c)

            # Schedule transfer on bus
            for o, c_list in require_bus.items():
                delay = node.output_memory[o] / self.params["transfer_speed"]
                add_to_dev_queue(f[node_name].end_time, "run_bus", self.bus + dev, (node_name, delay, c_list))

        # Run bus
        def run_bus(t, dev):
            """
            runbus
            """
            p, (node_name, delay, child_list) = device_queue[dev].get()
            # If bus is scheduled to run later, run later
            if p[0] > t:
                device_queue[dev].put((p, (node_name, delay, child_list)))
                q.put((p[0], "run_bus", dev))
                return
            for c in child_list:
                q.put((t + delay, "remove_dependency", (node_name, c)))
            q.put((t + delay + self.params["delta2"], "run_bus", dev))

        # Removes dependency of parent from child
        def remove_dependency(t, parent_name, child_name):
            """
            去除依赖
            """
            self.parent_map[child_name][parent_name] = False
            # Schedule child if no more dependencies
            if self.is_scheduleable(child_name):
                add_to_dev_queue(t, "run_dev", get_dev(child_name), child_name)

        # Insert all runnable ops to device_queue
        for name, node in self.node_dict.items():
            if not node.parents:
                add_to_dev_queue(self.params["init_offset"], "run_dev", get_dev(name), name)

        # Main loop
        while not q.empty():
            t, op, dev = q.get()
            run_time = max(run_time, t)
            if op in ["run_bus", "run_dev"] and device_queue[dev].empty():
                device_in_queue[dev] = False
                continue
            if op == "run_bus":
                run_bus(t, dev)
            if op == "remove_dependency":
                p_name, c_name = dev
                remove_dependency(t, p_name, c_name)
            if op == "run_dev":
                run_dev(dev)

        return run_time, f
