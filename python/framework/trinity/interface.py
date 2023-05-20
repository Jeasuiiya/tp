import copy
from framework.trinity.graph_scheduling import GraphScheduling

__doc__ = "Interface from trinity c++ to trinity python"


class PyTrinityInterface:
    """
    Interface from trinity c++ to trinity python
    """

    def __init__(self, networkx_graph=None, gcluster=None, hparams=None, verbose=True, step=5000):
        self.networkx_graph = networkx_graph
        self.gcluster = gcluster
        self.hparams = hparams
        self.verbose = verbose
        self.step = step
        self.nx_graph = None
        self.graph_placer = None
        self.best_placement = None

    def startup_strategy(self):
        """
        startup strategy
        """
        self.nx_graph = copy.deepcopy(self.networkx_graph.get_nx_graph())
        # 初始化设备图的放置
        available_devices = [device.name for device in self.gcluster.ListDevices()]
        for node in self.nx_graph.nodes():
            if self.nx_graph.nodes[node]["device"] is not None:
                # node.device = available_devices[random.randint(0, devices_num)]
                # 全都初始化为第一个GPU
                self.nx_graph.nodes[node]["device"] = available_devices[0]
        print("-----------------Generate a scheduler object---------------")
        self.graph_placer = GraphScheduling(self.nx_graph, self.gcluster, self.hparams, self.verbose, self.step)
        print("-----------------start scheduling---------------")
        self.best_placement = self.graph_placer.schedule_graph(isall=True)
        print("--------------End of Parallel Strategies for Optimal Model Search Using Trinity Method--------------")
        print(self.best_placement)
        return True

    def get_best_placement(self):
        return self.best_placement
