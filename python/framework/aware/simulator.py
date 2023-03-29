import heapq
from collections import defaultdict
from dataclasses import dataclass

__doc__ = "aware reinforcement simulator"


@dataclass
class Node:
    op_name: str
    device: str
    compute_cost: int
    children: list
    output_memory: list
    parents: list


class SimQueue:
    """
    Simulator queue, it is used to store tasks in the simulator.
    """

    def __init__(self):
        self.queue = []

    def put(self, x):
        heapq.heappush(self.queue, x)

    def get(self):
        return heapq.heappop(self.queue)

    def empty(self):
        return len(self.queue) == 0


class Simulator:
    """
    The second level simulator.
    """

    def __init__(self, nx_graph, dev_names):
        self.nx_graph = nx_graph
        self.node_dict = self.get_attributes()
        self.parent_map = defaultdict(dict)
        self.bus = "/bus"
        self.device_in_queue = {}
        self.device_queue = {}
        self.placement = {}
        self.sim_queue = SimQueue()
        self.count = 0
        self.dev_names = dev_names
        self.params = {"delta1": 5.7, "delta2": 25, "init_offset": 0, "transfer_speed": 7600}

        for name, node in self.node_dict.items():
            for parent in node.parents:
                self.parent_map[name][parent] = True

    def get_attributes(self):
        """
        Get the node dict used for simulator.
        """
        node_dict = {}
        for node in self.nx_graph.nodes():
            outedges = self.nx_graph.out_edges(node)
            inedges = self.nx_graph.in_edges(node)
            output_memory = []
            parents = []
            children = []
            for edge in outedges:
                output_memory.append(self.nx_graph[edge[0]][edge[1]]["weight"])
                children.append(edge[1])
            for edge in inedges:
                parents.append(edge[0])
            node_dict[node] = Node(
                op_name=node,
                device=self.nx_graph.nodes[node]["device"],
                compute_cost=self.nx_graph.nodes[node]["compute_cost"],
                children=children,
                output_memory=output_memory,
                parents=parents,
            )
        return node_dict

    def simulate(self, placement: dict):
        """
        The main body of simulator, start to simulate.
        """
        all_devs = self.dev_names + [self.bus + dev for dev in self.dev_names]
        self.device_in_queue = dict((dev, False) for dev in all_devs)
        self.device_queue = dict((dev, SimQueue()) for dev in all_devs)
        self.placement = placement

        # Reset parent_map
        for name, node in self.node_dict.items():
            for parent in node.parents:
                self.parent_map[name][parent] = True

        # Insert all runnable ops to device_queue
        for name, node in self.node_dict.items():
            # 初始节点
            if not node.parents:
                self.add_to_dev_queue(self.params["init_offset"], "run_dev", self.placement[name], name)

        # 模拟器核心, 模拟节点在设备上的执行
        # Main loop
        run_time = 0
        while not self.sim_queue.empty():
            time, op, dev = self.sim_queue.get()
            run_time = max(run_time, time)
            if (op in ("run_bus", "run_dev")) and self.device_queue[dev].empty():
                self.device_in_queue[dev] = False
            elif op == "run_bus":
                self.run_bus(time, dev)
            elif op == "remove_dependency":
                parent_name, child_name = dev
                self.remove_dependency(time, parent_name, child_name)
            elif op == "run_dev":
                self.run_dev(time, dev)

        # 修改到这里
        return run_time

    def add_to_dev_queue(self, time, op, dev, name):
        self.count += 1
        self.device_queue[dev].put(((time, self.count), name))
        if not self.device_in_queue[dev]:
            self.sim_queue.put((time, op, dev))
            self.device_in_queue[dev] = True

    # Runs the next job on device
    def run_dev(self, time, dev):
        """
        Runs the next job on device.
        """
        _, node_name = self.device_queue[dev].get()

        # Compute start and end times
        node = self.node_dict[node_name]
        self.node_dict[node_name].start_time = time
        self.node_dict[node_name].end_time = time + node.compute_cost
        self.node_dict[node_name].device = dev

        # Schedule when device is free again
        delta = 0 if node.compute_cost == 0 else self.params["delta1"]
        self.sim_queue.put((self.node_dict[node_name].end_time + delta, "run_dev", dev))

        # Find which all output indices require bus
        require_bus = defaultdict(list)  # output_index to list of children
        for idx, child in enumerate(node.children):
            # 如果父亲节点和儿子节点在同一设备上, 解除依赖
            if dev == self.placement[child]:
                self.sim_queue.put((self.node_dict[node_name].end_time, "remove_dependency", (node_name, child)))
            # 不同设备上, run bus跑一个通信开销
            else:
                require_bus[idx].append(child)

        # Schedule transfer on bus
        for idx, children in require_bus.items():
            delay = node.output_memory[idx] / self.params["transfer_speed"]
            self.add_to_dev_queue(
                self.node_dict[node_name].end_time, "run_bus", self.bus + dev, (node_name, delay, children)
            )

    def run_bus(self, time, dev):
        p, (node_name, delay, children) = self.device_queue[dev].get()
        # If bus is scheduled to run later, run later
        if p[0] > time:
            self.device_queue[dev].put((p, (node_name, delay, children)))
            self.sim_queue.put((p[0], "run_bus", dev))
            return
        for child in children:
            self.sim_queue.put((time + delay, "remove_dependency", (node_name, child)))
        self.sim_queue.put((time + delay + self.params["delta2"], "run_bus", dev))

    def remove_dependency(self, time, parent_name, child_name):
        self.parent_map[child_name][parent_name] = False
        # Schedule child if no more dependencies
        if self.is_scheduleable(child_name):
            self.add_to_dev_queue(time, "run_dev", self.placement[child_name], child_name)

    def is_scheduleable(self, node):
        for v in self.parent_map[node].values():
            if v:
                return False
        return True


class ImportantOpsSimulator(Simulator):
    """
    The first level simulator.
    """

    def __init__(self, config_params, nx_graph):
        self.config_params = config_params
        self.nx_graph = nx_graph
        self.dev_names = ["/GPU:" + str(idx) for idx in range(self.config_params["n_devs"])]

        self.output_memorys = defaultdict(list)
        self.compute_costs = {}
        for node in self.nx_graph.nodes():
            self.compute_costs[node] = self.nx_graph.nodes[node]["compute_cost"]
            outedges = self.nx_graph.out_edges(node)
            output_memory = []
            for edge in outedges:
                output_memory.append(self.nx_graph[edge[0]][edge[1]]["weight"])
            self.output_memorys[node] = output_memory

        Simulator.__init__(self, self.nx_graph, self.dev_names)

    def simulate(self, placement):
        for node, dev_id in placement.items():
            placement[node] = self.dev_names[int(dev_id)]

        run_time = Simulator.simulate(self, placement)

        start_time = {}
        for name, node in self.node_dict.items():
            start_time[name] = node.start_time

        output_memory_list = []
        for node_name, time in start_time.items():
            output_memory = sum(self.output_memorys[node_name])
            dev = self.dev_names.index(self.node_dict[node_name].device)
            output_memory_list.append((time, output_memory, dev))
            for child in self.node_dict[node_name].children:
                t_out_done = max(
                    time, int(self.node_dict[child].start_time) + int(self.node_dict[child].compute_cost) - 1
                )
            output_memory_list.append((t_out_done, -output_memory, dev))
        output_memory_list.sort()

        output_memory_utils = [0] * len(self.dev_names)
        output_memory_peak_utils = [0] * len(self.dev_names)
        for time, output_memory, dev in output_memory_list:
            output_memory_utils[dev] += output_memory
            # 峰值 memory 损耗
            if output_memory_utils[dev] > output_memory_peak_utils[dev]:
                output_memory_peak_utils[dev] = output_memory_utils[dev]

        return run_time, start_time, output_memory_peak_utils


class AwareSimulator:
    """
    The top level simulator.
    """

    def __init__(self, config_params, nx_graph):
        self.simulator = ImportantOpsSimulator(config_params, nx_graph)
        self.nx_graph = nx_graph

    def simulate(self, placement):
        for node in placement:
            placement[node] = str(placement[node])

        run_time, start_time, output_memory_peak_utils = self.simulator.simulate(placement)

        return run_time / 1e6, start_time, output_memory_peak_utils
