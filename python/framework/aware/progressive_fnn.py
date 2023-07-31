import random
import datetime

try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import numpy as np
import networkx as nx
from framework.tools import log

__doc__ = "Implementation of GraphSAGE-like algorithm for embedding to be used in the RL policy."

INIT_SCALE = 1.0


class FNN:
    """
    hidden_layer_sizes: list of hidden layer sizes
    out_size: Size of the last softmax layer
    """

    def __init__(self, inp_size, hidden_layer_sizes, out_size, name, dtype=tf.float32):
        layers = []
        sizes = [inp_size] + hidden_layer_sizes + [out_size]
        for i in range(len(sizes) - 1):
            w = self.glorot(shape=(sizes[i], sizes[i + 1]), scope=name, dtype=dtype)
            b = self.zero_init(shape=(1, sizes[i + 1]), scope=name, dtype=dtype)
            layers.append([w, b])

        self.layers = layers

    def build(self, inp_tensor):
        out = inp_tensor
        for idx, [w, b] in enumerate(self.layers):
            out = tf.matmul(out, w) + b
            if idx != len(self.layers) - 1:
                out = tf.nn.relu(out)
        return out

    def zero_init(self, shape, scope="default", dtype=tf.float32):
        with tf.variable_scope(scope):
            init = np.zeros(shape)
        return tf.Variable(init, dtype=dtype)

    def glorot(self, shape, scope="default", dtype=tf.float32):
        # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
        with tf.variable_scope(scope):
            init_range = np.sqrt(float(6.0 * INIT_SCALE) / int(shape[0] + shape[1]))
            init = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init, dtype=dtype)


class Aggregator:
    """
    N is the max number of children to be aggregated
    d is the degree of embeddings
    d1 is the degree of embedding transformation
    d2 is degree of aggregation
    """

    def __init__(self, d, d1=None, d2=None, use_mask=True, normalize_aggs=False, dtype=tf.float32):
        self.d = d
        self.d1 = d1
        self.d2 = d2
        self.normalize_aggs = normalize_aggs

        if d1 is None:
            d1 = self.d1 = d
        if d2 is None:
            d2 = self.d2 = d

        self.use_mask = use_mask

        if self.use_mask:
            self.Mask = tf.placeholder(dtype, shape=(None, None))

        self.f = FNN(self.d, [self.d], self.d1, "f", dtype=dtype)
        self.g = FNN(self.d1, [self.d1], self.d2, "g", dtype=dtype)

    def build(self, input_tensor, mask=None):
        """
        Build the Aggregator.
        """
        f_out = tf.nn.relu(self.f.build(input_tensor))
        if self.use_mask or mask is not None:
            if mask is None:
                mask = self.Mask
            g = tf.matmul(mask, f_out)
            if self.normalize_aggs:
                d = tf.cond(tf.reduce_sum(mask) > 0, lambda: tf.reduce_sum(mask), lambda: 1.0)
                g /= d
        else:
            g = tf.reduce_sum(f_out, 0, keepdims=True)
        g = tf.nn.relu(self.g.build(g))
        return g


class Classifier:
    def __init__(self, input_size, hidden_layer_sizes, out_size, dtype=tf.float32):
        self.nn = FNN(input_size, hidden_layer_sizes, out_size, "classifier", dtype=dtype)

    def build(self, input_tensor):
        return self.nn.build(input_tensor)


class RadioMessenger:
    """
    Implementation of GraphSAGE-like algorithm for embedding to be used in the RL policy.

    Paper: "Inductive Representation Learning on Large Graphs" (https://arxiv.org/pdf/1706.02216.pdf)

    Parameters:
        - `embedding_size` - int - degree of embeddings
        - `sample_ratio` - float [0,1] - what part of a node's neighbours are used to calculate its embeddings
        - `hops` - int [1,2] - how many hops away need to be aggregated
        - `aggregation` - {'mean', 'max', 'min', 'sum'} - how are a node's neighbours aggregated. Default is 'mean'
    """

    def __init__(self, config_params, embedding_size, progressive_graph):
        self.config_params = config_params
        self.embedding_size = embedding_size
        self.progressive_graph = progressive_graph
        self.nx_nodes = progressive_graph.node_traversal_order
        self.nx_graph = progressive_graph.nx_graph
        self.memo = {}
        self.samples = {}
        self.anchor_sets = []
        self.fnns = {}
        self.distances = {}
        self.bs = 1
        self.hops = 1
        self.transform = None

        self._init_fnns()

    def _init_fnns(self):
        with tf.name_scope("self_transform"):
            self.self_transform = FNN(
                hidden_layer_sizes=[self.embedding_size * 2],
                inp_size=self.embedding_size * 2,
                out_size=self.embedding_size * 2,
                name="self_transform",
            )
        # # forward pass
        with tf.name_scope("FPA"):
            self.fpa = Aggregator(self.embedding_size, self.embedding_size, self.embedding_size, False)
        with tf.name_scope("BPA"):
            self.bpa = Aggregator(self.embedding_size, self.embedding_size, self.embedding_size, False)
        with tf.name_scope("node_transform"):
            self.node_transform = FNN(
                self.embedding_size, [self.embedding_size, self.embedding_size], self.embedding_size, "fnn"
            )
        with tf.name_scope("positional_awareness"):
            self.fnns["pos"] = FNN(
                hidden_layer_sizes=[self.embedding_size * 2],
                inp_size=self.embedding_size * 2,
                out_size=self.embedding_size * 2,
                name="positional_awareness",
            )

    def build(self):
        """
        Build the RadioMessenger.
        """
        f_adj = self.progressive_graph.fadj
        b_adj = self.progressive_graph.badj

        assert np.trace(f_adj) == 0
        assert np.trace(b_adj) == 0

        f_adj = self.function(f_adj, self.bs)
        b_adj = self.function(b_adj, self.bs)

        input_tensor = tf.placeholder(tf.float32)
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        input_tensor = tf.reshape(input_tensor, [-1, tf.shape(input_tensor)[-1]])
        self_trans = self.node_transform.build(input_tensor)

        with tf.variable_scope("Forward_pass"):
            out_fpa = self.message_pass(f_adj, self_trans, self.fpa)
        with tf.variable_scope("Backward_pass"):
            out_bpa = self.message_pass(b_adj, self_trans, self.bpa)

        out = tf.concat([out_fpa, out_bpa], axis=-1)
        out = tf.cast(out, tf.float32)
        self.transform = out

        # 2. Generate samples of each node's neighbourhood. Based on the `sample_ratio` class parameter
        self._generate_samples()

        for node in list(nx.topological_sort(self.nx_graph)):
            self._get_embeddings(node, 0)
            self.samples[node][str(1) + "pooled"] = out

        # 4. Return either the concatenated node embeddings for all nodes for the given number of hops,
        # or the P-GNN position aware embeddings based on anchor sets

        # 4.1.1. Pre-calculate distances between all node pairs
        self._precalculate_distances(self.config_params["pgnn_neigh_cutoff"])

        # 4.1.2. Build the anchor sets based on the Bourgain theorem used in P-GNN
        self._build_anchor_sets()

        # 4.1.3. Generate all embeddings for nodes based on feature info of the node and feature info of the nodes in
        # all anchor sets. Anchor set aggregations can be obtained using max or mean aggregation
        positional_info_generator = self._aggregate_positional_info(
            self.nx_nodes, self.config_params["pgnn_aggregation"]
        )

        positions = list(positional_info_generator)

        out = tf.reshape(positions, shape=[-1, self.embedding_size * 2])
        log.debug("Returning P-GNN values with shape %s %s", out.shape, datetime.datetime.now())
        return out

    def function(self, adj, bs=1):
        n = adj.shape[0]
        t = np.zeros([bs * n] * 2, dtype=np.float32)
        for i in range(bs):
            t[i * n : (i + 1) * n, i * n : (i + 1) * n] = adj

        return t

    def message_pass(self, adj, self_trans, agg):
        """
        Message pass.
        """
        sink_mask = np.sum(adj, axis=-1) > 0
        sink_mask = tf.cast(sink_mask, tf.float32)
        adj = tf.cast(adj, tf.float32)
        x = self_trans

        for _ in range(self.config_params["radial_mp"]):
            x = agg.build(x, mask=adj)
            x = sink_mask * tf.transpose(x)
            x = tf.transpose(x)
            x += self_trans

        return x

    def _generate_samples(self):
        for node in self.nx_nodes:
            if self.samples.get(node) is None:
                self.samples[node] = {}
            self.samples[node]["init"] = self._get_sample(node)

    def _get_sample(self, node):
        # Get a random sample of neighbours based on the ratio
        # e.g. if the ratio is 0.5, we will return only half the successors of the current node
        neighbors = list(self.nx_graph.neighbors(node))
        samples = random.sample(neighbors, int(len(neighbors) * self.config_params["sage_sample_ratio"]))
        return samples

    def _get_embeddings(self, node, level=0):
        neigh_samples = self.samples[node]["init"]
        # if we don't have the initial embeddings of current node and its neighbours
        # 3.1.1 If it is the first level, we generate the embedding based only on the fetures of each node.
        # We return the embedding of the current node for the current level, as well as the list of its neighbours'
        # embeddings for the same level.
        if level == 0:
            self._generate_initial_self_embedding(node, self.transform)
        for neigh_sample in neigh_samples:
            self._generate_initial_self_embedding(neigh_sample, self.transform)

    def _generate_initial_self_embedding(self, node, transform):
        if self.samples[node].get("0") is None:
            self.samples[node]["0"] = tf.expand_dims(transform[self.progressive_graph.get_idx(node), :], axis=0)

    def _precalculate_distances(self, cutoff=6):
        self.distances = dict(nx.all_pairs_shortest_path_length(self.nx_graph, cutoff))

    def _build_anchor_sets(self):
        n = len(self.nx_nodes)
        m = int(np.log(n))
        copy = int(self.config_params["pgnn_c"] * m)
        for i in range(m):
            anchor_size = int(n / np.exp2(i + self.config_params["pgnn_anchor_exponent"]))
            for _ in range(np.maximum(copy, 1)):
                self.anchor_sets.append(random.sample(self.nx_nodes, anchor_size))
        log.debug(
            "Number of anchor sets: %s . Biggest set is: %s",
            len(self.anchor_sets),
            str(int(n / np.exp2(self.config_params["pgnn_anchor_exponent"]))),
        )

    def _aggregate_positional_info(self, nodes, aggregation="max"):
        log.debug("P-GNN aggregation is %s", aggregation)
        for _, node in enumerate(nodes):
            if self.memo.get(node) is None:
                self.memo[node] = {}
            # log.debug(i, n)
            positional_aggregation = []
            for anchor_set in self.anchor_sets:
                aggregated = None
                if aggregation == "max":
                    aggregated = self._max_aggregate_anchor(anchor_set, node)
                elif aggregation == "mean":
                    # This one has a big performance overhead
                    aggregated = self._mean_aggregate_anchor(anchor_set, node)
                positional_aggregation.append(aggregated)

            positional_aggregation = tf.concat(positional_aggregation, axis=0)
            positional_aggregation = tf.reduce_mean(positional_aggregation, axis=0)
            positional_aggregation = tf.expand_dims(positional_aggregation, axis=0)
            yield self.fnns["pos"].build(positional_aggregation)

    def _max_aggregate_anchor(self, anchor_set, node):
        # find the nodes of the anchor set which can be reached by the current node
        anchor_node_intersections = [
            (k, self.distances[node][k]) for k in anchor_set if self.distances[node].get(k) is not None and k != node
        ]

        # get the node with the maximum distance
        max_agg_anchor = max(anchor_node_intersections, key=lambda i: i[1], default=None)
        node_embedding = self.samples[node][str(1) + "pooled"]

        # if there is no such node, create a zero tensor to keep the dimensions
        if max_agg_anchor is None:
            return tf.zeros(shape=[node_embedding.shape[0] + node_embedding.shape[0], node_embedding.shape[-1]])

        # get the precalculated embedding of the max node
        anchor_embedding = self.samples[max_agg_anchor[0]][str(1) + "pooled"]

        # calculate the distance
        positional_info = 1 / (self.distances[node][max_agg_anchor[0]] + 1)

        # concatenate the embeddings of the node and the max anchor node
        feature_info = tf.concat((node_embedding, anchor_embedding), axis=0)
        node_anchor_relation = positional_info * feature_info

        return node_anchor_relation

    def _mean_aggregate_anchor(self, anchor_set, node):
        node_positions = []
        for anchor in anchor_set:
            if self.memo[node].get(anchor) is not None:
                node_anchor_relation = self.memo[node][anchor]
                node_positions.append(node_anchor_relation)
                continue

            node_embedding = self.samples[node][str(self.hops) + "pooled"]
            anchor_embedding = self.samples[anchor][str(self.hops) + "pooled"]

            # positional info between n and anchor node
            if self.distances.get(node) is not None and self.distances[node].get(anchor) is not None:
                positional_info = 1 / (self.distances[node][anchor] + 1)
                feature_info = tf.concat((node_embedding, anchor_embedding), axis=0)
                node_anchor_relation = positional_info * feature_info
            else:
                node_anchor_relation = tf.zeros(
                    shape=[node_embedding.shape[0] + anchor_embedding.shape[0], node_embedding.shape[-1]]
                )

            self.memo[node][anchor] = node_anchor_relation
            node_positions.append(node_anchor_relation)
        return tf.reduce_mean(node_positions, axis=0)
