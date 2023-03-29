import random
import numpy as np
import networkx as nx

__doc__ = "Init the node embeddings in this file"


class NodeEmbeddings:
    """
    Init the node embedding and record the embedding
    """

    def __init__(self, compute_cost: int, output_memory: int, placement=None):
        self.compute_cost: int = compute_cost
        self.output_memory: int = output_memory
        self.placement: int = placement
        self.curr_bit: int = 0
        self.done_bit: int = 0
        self.start_time: float = None

    def get_embedding(self, n_devs):
        embedding = [self.compute_cost, self.output_memory, self.done_bit, self.curr_bit, self.start_time]
        embedding = embedding + self.placement_to_one_hot(self.placement, n_devs)
        return embedding

    def placement_to_one_hot(self, placement, n_devs):
        ret = [0] * n_devs
        if placement is not None:
            ret[placement] = 1
        return ret

    def update_start_time(self, start_time):
        self.start_time = start_time

    def update_placement(self, placement):
        self.placement = placement

    def reset_curr_bit(self):
        self.curr_bit = 0

    def reset_done_bit(self):
        self.done_bit = 0

    def set_curr_bit(self):
        self.curr_bit = 1

    def normalize_start_time(self, embeddings, curr_node):
        embeddings[:, 4] -= embeddings[curr_node, 4]
        return embeddings

    def normalize(self, embeddings, factors):
        # normalize cost, out_memory, start_time
        for i in [0, 1, 4]:
            if factors[i] != 0:
                embeddings[:, i] /= factors[i]
        return embeddings

    def inc_done_bit(self):
        self.done_bit += 1


class ProgressiveGraph:
    """
    Init and generate the embedding graph and save the embeddings of nodes
    """

    def __init__(self, nx_graph: nx.DiGraph(), n_devs: int, node_traversal_order: str, seed: int = 42):
        self.seed: int = seed
        random.seed(seed)
        self.nx_graph: nx.DiGraph() = nx_graph
        self.n_devs: int = n_devs
        self.node_embeddings = []
        (
            self.peer_mat,
            self.progenial_mat,
            self.ancestral_mat,
            self.adj_mat,
            self.undirected_adj_mat,
            self.badj,
            self.fadj,
            self.curr_node,
            self.embeddings,
        ) = (None, None, None, None, None, None, None, None, None)

        if node_traversal_order == "topo":
            self.node_traversal_order = list(nx.topological_sort(self.nx_graph))
        elif node_traversal_order == "random":
            self.node_traversal_order = list(self.nx_graph.nodes())
            random.shuffle(self.node_traversal_order)
        else:
            raise ValueError("Node traversal order not specified correctly")

        idx = {}
        for i, node in enumerate(self.node_traversal_order):
            idx[node] = i
        nx.set_node_attributes(self.nx_graph, idx, "idx")

        self.init_node_embeddings()
        self.init_positional_mats()
        self.init_adj_mat()
        self.init_badj_fadj()

    def init_node_embeddings(self):
        E = []
        for node in self.node_traversal_order:
            outedges = self.nx_graph.out_edges(node)
            all_out_memory = 0
            for edge in outedges:
                all_out_memory += self.nx_graph[edge[0]][edge[1]]["weight"]
            e = NodeEmbeddings(self.nx_graph.nodes[node]["compute_cost"], all_out_memory)
            E.append(e)
        self.node_embeddings: list(NodeEmbeddings) = E

    def init_positional_mats(self):
        """
        Init positional matrix.
        """
        path_mat = nx.floyd_warshall_numpy(self.nx_graph, nodelist=self.node_traversal_order)
        peer_mat: np.ndarray = np.isinf(path_mat)
        for i in range(len(peer_mat)):
            for j in range(len(peer_mat)):
                if i != j:
                    peer_mat[i, j] &= peer_mat[j, i]
                else:
                    peer_mat[i, j] = False

        self.peer_mat = peer_mat
        self.progenial_mat = np.logical_not(np.isinf(path_mat))
        np.fill_diagonal(self.progenial_mat, 0)
        self.ancestral_mat = self.progenial_mat.T

    def init_adj_mat(self):
        self.adj_mat = nx.to_numpy_array(self.nx_graph, nodelist=self.node_traversal_order)
        self.undirected_adj_mat = np.array(self.adj_mat)

        for i in range(len(self.adj_mat)):
            for j in range(len(self.adj_mat)):
                self.undirected_adj_mat[i, j] = max(self.undirected_adj_mat[i, j], self.undirected_adj_mat[j, i])

    def init_badj_fadj(self):
        self.badj = np.float32(nx.to_numpy_array(self.nx_graph, self.node_traversal_order))
        self.fadj = self.badj.transpose()

    def get_embed_size(self):
        # why is 5 : len([self.compute_cost, self.output_memory, self.done_bit, self.curr_bit, self.start_time])
        #  + onehot(devs)
        return 5 + self.n_devs

    def get_idx(self, node):
        return self.nx_graph.nodes[node]["idx"]

    def get_zero_placement(self):
        zero_placement = {}
        for node in self.node_traversal_order:
            zero_placement[node] = 0
        return zero_placement

    def get_random_placement(self, seed=None):
        random_placement = {}
        if seed:
            random.seed(seed)
        for node in self.node_traversal_order:
            random_placement[node] = random.randint(0, self.n_devs - 1)
        return random_placement

    def get_null_placement(self):
        null_placement = {}
        for node in self.node_traversal_order:
            null_placement[node] = None
        return null_placement

    def set_start_times(self, start_time):
        for idx, node in enumerate(self.node_traversal_order):
            self.node_embeddings[idx].update_start_time(start_time[node])

    def reset_placement(self, placement):
        for i, node in enumerate(self.node_traversal_order):
            self.node_embeddings[i].update_placement(placement[node])
        nx.set_node_attributes(self.nx_graph, placement, "placement")

    def new_episode(self):
        for node_embedding in self.node_embeddings:
            node_embedding.reset_curr_bit()
            node_embedding.reset_done_bit()

    def get_placement(self):
        return nx.get_node_attributes(self.nx_graph, "placement")

    def set_curr_node(self, node):
        for node_embedding in self.node_embeddings:
            node_embedding.reset_curr_bit()

        idx = self.get_idx(node)
        self.node_embeddings[idx].set_curr_bit()
        self.curr_node = idx

    def get_ancestral_mask(self, node):
        return self.ancestral_mat[self.get_idx(node), :]

    def get_progenial_mask(self, node):
        return self.progenial_mat[self.get_idx(node), :]

    def get_peer_mask(self, node):
        return self.peer_mat[self.get_idx(node), :]

    def get_self_mask(self, node):
        m = np.zeros((1, len(self.node_traversal_order)))
        m[:, self.get_idx(node)] = 1.0
        return m

    def get_embeddings(self):
        E = []
        for node_embedding in self.node_embeddings:
            E.append(node_embedding.get_embedding(self.n_devs))

        E = np.array(E, dtype=np.float32)
        E = NodeEmbeddings.normalize_start_time(self, E, self.curr_node)
        self.embeddings = NodeEmbeddings.normalize(self, E, np.amax(E, axis=0))

        return self.embeddings

    def refresh(self, node, placement):
        self.nx_graph.nodes[node]["placement"] = placement
        idx = self.get_idx(node)
        self.node_embeddings[idx].update_placement(placement)

    def inc_done_node(self, node):
        idx = self.get_idx(node)
        self.node_embeddings[idx].inc_done_bit()

    def update_start_time(self, start_time):
        for idx, node in enumerate(self.node_traversal_order):
            self.node_embeddings[idx].update_start_time(start_time[node])
