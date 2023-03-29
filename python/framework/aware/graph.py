import copy
import networkx as nx

__doc__ = "The structure of aware graph"


class Graph:
    """
    The structure of aware input: an networkx graph
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, name, cost, memory, device):
        """
        Create the node in nx_graph
        """
        self.graph.add_node(name)
        self.graph.nodes[name]["compute_cost"] = cost
        self.graph.nodes[name]["memory"] = memory
        self.graph.nodes[name]["device"] = device

    def add_edge(self, name_from, name_to, output_memory):
        # output memory is euqal to comm cost
        self.graph.add_edge(name_from, name_to, weight=output_memory)

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def print_node(self, name):
        return self.graph.nodes[name]

    def get_node_cost(self, name):
        return self.graph.nodes[name]["compute_cost"]

    def get_node_mem(self, name):
        return self.graph.nodes[name]["memory"]

    def get_node_device(self, name):
        return self.graph.nodes[name]["device"]

    def get_all_nodes(self):
        return copy.deepcopy(self.graph.nodes)

    def get_nx_graph(self):
        return self.graph
