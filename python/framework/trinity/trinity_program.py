"""Triniry program"""
import math
import time
import collections
from collections import defaultdict
import io
import networkx as nx
import six
import numpy as np

import tensorflow.compat.v1 as tf
from framework.trinity.sim.tf_placement_sim.tf_pl_rewarder import ImportantOpsRewarder
from framework.trinity.ge import Node2Vec
from framework.trinity.programer import Controller
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import training_util

# tf.disable_v2_behavior()

Cost = collections.namedtuple("Cost", ["compute_cost", "output_memory", "temporary_memory", "persistent_memory"])


class PlacerParams:
    """Class to hold a set of placement parameters as name-value pairs.

    A typical usage is as follows:

    ```python
    # Create a PlacerParams object specifying names and values of the model
    # parameters:
    params = PlacerParams(hidden_size=128, decay_steps=50)

    # The parameters are available as attributes of the PlacerParams object:
    hparams.hidden_size ==> 128
    hparams.decay_steps ==> 50
    ```

    """

    def __init__(self, **kwargs):
        """Create an instance of `PlacerParams` from keyword arguments.

        The keyword arguments specify name-values pairs for the parameters.
        The parameter types are inferred from the type of the values passed.

        The parameter names are added as attributes of `PlacerParams` object,
        and they can be accessed directly with the dot notation `params._name_`.

        Example:

        ```python
        # Define 1 parameter: 'hidden_size'
        params = PlacerParams(hidden_size=128)
        params.hidden_size ==> 128
        ```

        Args:
          **kwargs: Key-value pairs where the key is the parameter name and
            the value is the value for the parameter.
        """
        for name, value in six.iteritems(kwargs):
            self.add_param(name, value)

    def add_param(self, name, value):
        """Adds {name, value} pair to hyperparameters.

        Args:
          name: Name of the hyperparameter.
          value: Value of the hyperparameter. Can be one of the following types:
            int, float, string, int list, float list, or string list.

        Raises:
          ValueError: if one of the arguments is invalid.
        """
        # Keys in kwargs are unique, but 'name' could be the name of a pre-existing
        # attribute of this object.  In that case we refuse to use it as a
        # parameter name.
        if getattr(self, name, None) is not None:
            raise ValueError(f"Parameter name is reserved: {name}")
        setattr(self, name, value)


def trinity_mian_hparams():
    """Hyperparameters for Trinity planner."""
    return PlacerParams(
        hidden_size=512,
        forget_bias_init=1.0,
        temperature=1.0,
        logits_std_noise=0,
        stop_noise_step=500,
        decay_steps=50,
        max_num_outputs=5,
        max_output_size=5,
        tanh_constant=1.0,
        adj_embed_dim=20,
        grouping_hidden_size=64,
        num_groups=None,
        bi_lstm=True,
        failing_signal=10,
        stop_sampling=10000,
        start_with_failing_signal=True,
        always_update_baseline=False,
        bl_dec=0.9,
        grad_bound=1.0,
        # lr设置为0.1是否合适
        lr=0.01,
        lr_dec=0.95,
        start_decay_step=400,
        optimizer_type="adam",
        stop_updating_after_steps=2000,
        name="hierarchical_controller",
        keep_prob=None,
        reward_function="sqrt",
        seed=343,
        # distributed training params
        num_children=20,
        METHOD=[
            {"name": "kl_pen", "kl_target": 0.01, "lam": 0.5},  # KL散度
            {"name": "clip", "epsilon": 0.3},  # 取代KL散度的一种方法，效果会更好，只需要使得他的比例在一定范围内即可
        ][1],
    )


class TrinityProgram(Controller):
    """
    Google分层模型HierarchicalController class
    继承自Controller
    """

    def __init__(self, hparams, nx_graph, cluster, controller_id=0):
        self.old = "ppo_old"
        self.new = "ppo_new"
        self.METHOD = [
            {"name": "kl_pen", "kl_target": 0.01, "lam": 0.5},  # KL penalty
            {"name": "clip", "epsilon": 0.3},  # Clipped surrogate objective
        ][1]
        """HierarchicalController初始化

        Args:
          hparams: 有关训练模型的超参数
          item: 需要被放置的Items
          cluster: 要优化的硬件设备集群
          controller_id: 在多控制器训练过程当中的控制器ID.
        """
        super().__init__(nx_graph, cluster)
        self.ctrl_id = controller_id
        self.hparams = hparams
        self.cost_d = None
        self.out_d = None
        self._global_step = None
        self.grouping_actions_cache = None
        self.grouping_actions_cache_old = None
        self.dag_matrix = None
        self.actions_cache = None
        self.actions_cache_old = None
        self.ops = None

        self.num_groups = min(256, 20 * self.num_devices)
        # 创建self.op_embeddings and self.type_dict
        self.create_op_embeddings()
        # TODO(azalia) clean up embedding/group_embedding_size names
        # create_group
        self.group_emb_size = (
            1 + 2 * self.num_groups + len(self.type_dict) + self.hparams.max_num_outputs * self.hparams.max_output_size
        )
        self.embedding_size = self.group_emb_size
        # 均匀分布式初始化器
        self.initializer = init_ops.glorot_uniform_initializer(seed=self.hparams.seed)
        self.create_lstm_variables()

        seq2seq_input_layer = array_ops.placeholder_with_default(
            array_ops.zeros([self.hparams.num_children, self.num_groups, self.group_emb_size], dtypes.float32),
            shape=(self.hparams.num_children, self.num_groups, self.group_emb_size),
        )
        self.seq2seq_input_layer = seq2seq_input_layer

    def create_lstm_variables(self):
        """
        create lstm variables
        """
        with variable_scope.variable_scope(
            self.hparams.name, initializer=self.initializer, reuse=variable_scope.AUTO_REUSE
        ):
            with variable_scope.variable_scope("ppo_old"):
                with variable_scope.variable_scope("grouping"):
                    variable_scope.get_variable(
                        "w_grouping_ff", [2 + 64 + 1, self.hparams.grouping_hidden_size], trainable=False
                    )
                    variable_scope.get_variable(
                        "w_grouping_softmax", [self.hparams.grouping_hidden_size, self.num_groups], trainable=False
                    )
                with variable_scope.variable_scope("lstm"):
                    variable_scope.get_variable(
                        "encoder_lstm_forward",
                        [self.embedding_size + self.hparams.hidden_size // 2, 2 * self.hparams.hidden_size],
                        trainable=False,
                    )
                    variable_scope.get_variable(
                        "encoder_lstm_backward",
                        [self.embedding_size + self.hparams.hidden_size // 2, 2 * self.hparams.hidden_size],
                        trainable=False,
                    )
                    variable_scope.get_variable(
                        "device_embeddings", [self.num_devices, self.hparams.hidden_size], trainable=False
                    )
                    variable_scope.get_variable(
                        "decoder_lstm",
                        [2 * self.hparams.hidden_size, 4 * self.hparams.hidden_size],
                        trainable=False,
                    )
                    variable_scope.get_variable(
                        "device_softmax", [2 * self.hparams.hidden_size, self.num_devices], trainable=False
                    )
                    variable_scope.get_variable("device_go_embedding", [1, self.hparams.hidden_size], trainable=False)
                    variable_scope.get_variable(
                        "encoder_forget_bias",
                        shape=1,
                        dtype=dtypes.float32,
                        initializer=init_ops.constant_initializer(self.hparams.forget_bias_init),
                        trainable=False,
                    )
                    variable_scope.get_variable(
                        "decoder_forget_bias",
                        shape=1,
                        dtype=dtypes.float32,
                        initializer=init_ops.constant_initializer(self.hparams.forget_bias_init),
                        trainable=False,
                    )
                    variable_scope.get_variable(
                        "attn_w_1", [self.hparams.hidden_size, self.hparams.hidden_size], trainable=False
                    )
                    variable_scope.get_variable(
                        "attn_w_2", [self.hparams.hidden_size, self.hparams.hidden_size], trainable=False
                    )
                    variable_scope.get_variable("attn_v", [self.hparams.hidden_size, 1], trainable=False)

            with variable_scope.variable_scope("ppo_new"):
                with variable_scope.variable_scope("grouping"):
                    variable_scope.get_variable("w_grouping_ff", [2 + 64 + 1, self.hparams.grouping_hidden_size])
                    variable_scope.get_variable(
                        "w_grouping_softmax", [self.hparams.grouping_hidden_size, self.num_groups]
                    )
                with variable_scope.variable_scope("lstm"):
                    variable_scope.get_variable(
                        "encoder_lstm_forward",
                        [self.embedding_size + self.hparams.hidden_size // 2, 2 * self.hparams.hidden_size],
                    )
                    variable_scope.get_variable(
                        "encoder_lstm_backward",
                        [self.embedding_size + self.hparams.hidden_size // 2, 2 * self.hparams.hidden_size],
                    )
                    variable_scope.get_variable("device_embeddings", [self.num_devices, self.hparams.hidden_size])
                    variable_scope.get_variable(
                        "decoder_lstm", [2 * self.hparams.hidden_size, 4 * self.hparams.hidden_size]
                    )
                    variable_scope.get_variable("device_softmax", [2 * self.hparams.hidden_size, self.num_devices])
                    variable_scope.get_variable("device_go_embedding", [1, self.hparams.hidden_size])
                    variable_scope.get_variable(
                        "encoder_forget_bias",
                        shape=1,
                        dtype=dtypes.float32,
                        initializer=init_ops.constant_initializer(self.hparams.forget_bias_init),
                    )
                    variable_scope.get_variable(
                        "decoder_forget_bias",
                        shape=1,
                        dtype=dtypes.float32,
                        initializer=init_ops.constant_initializer(self.hparams.forget_bias_init),
                    )
                    variable_scope.get_variable("attn_w_1", [self.hparams.hidden_size, self.hparams.hidden_size])
                    variable_scope.get_variable("attn_w_2", [self.hparams.hidden_size, self.hparams.hidden_size])
                    variable_scope.get_variable("attn_v", [self.hparams.hidden_size, 1])

    def compute_reward(self, run_time):
        """
        define the method to compute reward
        """
        if self.hparams.reward_function == "id":
            reward = run_time
        elif self.hparams.reward_function == "sqrt":
            reward = math.sqrt(run_time)
        elif self.hparams.reward_function == "log":
            reward = math.log1p(run_time)
        elif self.hparams.reward_function == "err":
            reward = 1 / run_time
        else:
            raise NotImplementedError(
                f"Unrecognized reward function {self.hparams.reward_function}, consider your "
                "--reward_function flag value."
            )
        return reward

    def get_op_costs(self):
        """
        define the method to get op cost
        """
        cost_d = {}
        for node in self.nx_graph.nodes():
            cost_d[node] = self.nx_graph.nodes[node]["compute_cost"]
        return cost_d

    def get_op_mem(self):
        """
        define the method to get op mem
        """
        out_d = {}
        # for node in self.nx_graph.nodes():
        #     out_d[node] = self.nx_graph.nodes[node]["compute_cost"]
        for node in self.nx_graph.nodes():
            outedges = self.nx_graph.out_edges(node)
            output_memory = []
            for edge in outedges:
                output_memory.append(self.nx_graph[edge[0]][edge[1]]["weight"])
            out_d[node] = output_memory
        return out_d

    def define_controller_variables(self, ctr, failing_signal):
        """
        define controller variables
        """
        with variable_scope.variable_scope(f"controller_{self.ctrl_id}"):
            ctr["reward"] = {"value": [], "ph": [], "update": []}
            ctr["ready"] = {"value": [], "ph": [], "update": []}
            ctr["best_reward"] = {"value": [], "update": []}
            for i in range(self.hparams.num_children):
                reward_value = variable_scope.get_local_variable(
                    f"reward_{i}", initializer=0.0, dtype=dtypes.float32, trainable=False
                )
                reward_ph = array_ops.placeholder(dtypes.float32, shape=(), name=f"reward_ph_{i}")

                # 完成对于第一个节点的返回的数据的赋值过程节点
                reward_update = state_ops.assign(reward_value, reward_ph, use_locking=True)
                ctr["reward"]["value"].append(reward_value)
                ctr["reward"]["ph"].append(reward_ph)
                ctr["reward"]["update"].append(reward_update)

                best_reward = variable_scope.get_local_variable(
                    f"best_reward_{i}", initializer=failing_signal, dtype=dtypes.float32, trainable=False
                )
                # 将得定义好的best_reward放到ctr控制当中
                ctr["best_reward"]["value"].append(best_reward)
                # 用于选择当前最佳和刚更新reward较小的那个，赋值给best_reward
                ctr["best_reward"]["update"].append(
                    state_ops.assign(best_reward, math_ops.minimum(best_reward, reward_update))
                )

                ready_value = variable_scope.get_local_variable(
                    f"ready_{i}", initializer=True, dtype=dtypes.bool, trainable=False
                )
                ready_ph = array_ops.placeholder(dtypes.bool, shape=(), name=f"ready_ph_{i}")
                ready_update = state_ops.assign(ready_value, ready_ph, use_locking=True)
                ctr["ready"]["value"].append(ready_value)
                ctr["ready"]["ph"].append(ready_ph)
                ctr["ready"]["update"].append(ready_update)
            return ctr, best_reward

    def define_baseline_shape(self, ctr, failing_signal):
        with variable_scope.variable_scope(f"controller_{self.ctrl_id}"):
            # 定义baseline形状，本质上是Reward，所以采用failing_signal
            ctr["baseline"] = variable_scope.get_local_variable(
                "baseline",
                initializer=failing_signal if self.hparams.start_with_failing_signal else 0.0,
                dtype=dtypes.float32,
                trainable=False,
            )
            return ctr

    def calculate_controller_actions_probabilities(self, ctr, suff, group_ratio, place_ratio):
        """
        calculate controller actions probabilities
        """
        with variable_scope.variable_scope(f"controller_{self.ctrl_id}"):
            if self.METHOD["name"] == "kl_pen":
                pass
            else:  # clipping method, find this is better
                suff = array_ops.expand_dims(-suff, 1)
                t1 = tf.minimum(
                    group_ratio * suff,
                    tf.clip_by_value(group_ratio, 1.0 - self.METHOD["epsilon"], 1.0 + self.METHOD["epsilon"]) * suff,
                )
                t1 = math_ops.reduce_sum(t1, 1)

                t2 = tf.minimum(
                    place_ratio * suff,
                    tf.clip_by_value(place_ratio, 1.0 - self.METHOD["epsilon"], 1.0 + self.METHOD["epsilon"]) * suff,
                )
                t2 = math_ops.reduce_sum(t2, 1)
                # 正负权衡一下
                ctr["loss"] = t1 + t2
            return ctr, suff

    def variable_optimizer(self, ctr):
        """
        优化过程
        """
        with variable_scope.variable_scope("optimizer", reuse=variable_scope.AUTO_REUSE):
            (ctr["train_op"], ctr["lr"], ctr["grad_norm"], ctr["grad_norms"], ctr["debug"]) = self._get_train_ops(
                ctr["loss"],
                tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES),
                self.global_step,
                grad_bound=self.hparams.grad_bound,
                lr_init=self.hparams.lr,
                lr_dec=self.hparams.lr_dec,
                start_decay_step=self.hparams.start_decay_step,
                decay_steps=self.hparams.decay_steps,
                optimizer_type=self.hparams.optimizer_type,
            )
            return ctr

    def update_old_strategy(self, ctr):
        """
        更新旧策略，替换神经网络中的参数
        """
        with tf.variable_scope("update_old"):
            # 更新旧策略，替换神经网络中的参数
            ctr["update_old_grouping_op"] = [
                oldp.assign(p) for p, oldp in zip(ctr["new_grouping_param"], ctr["old_grouping_param"])
            ]
            ctr["update_old_lstm_op"] = [
                oldp.assign(p) for p, oldp in zip(ctr["new_lstm_param"], ctr["old_lstm_param"])
            ]
            return ctr

    def build_program(self):
        """强化学习方法的接口，在训练时只需要调用相应的图操作集合即可.

        Returns:
          强化学习方法的接口，在训练时只需要调用相应的图操作集合即可.
          ops: A dictionary holding handles of the model used for training.
        """
        # tensorflow支持采用该方法创建一个全局的步数
        self._global_step = training_util.get_or_create_global_step()
        # 强化学习操作句柄
        ctr, ops = {}, {"loss": 0}

        # 采用开平方的方法计算reward，初始值是100
        failing_signal = self.compute_reward(self.hparams.failing_signal)

        # 对各种对象和操作定义在name_scope中可以方便其管理，用于区别对象属于哪个区域，ctrl_id为多控制器时候的ID
        with tf_ops.name_scope(f"controller_{self.ctrl_id}"):
            ctr, best_reward = self.define_controller_variables(ctr, failing_signal)
            # grouping_y_preds一个分组的预测（action）；grouping_log_probs分布的log值（概率分布）
            # get_groupings本质上是Grouper网络，输入操作的embedding，输出的是每个ops的放置和放置概率

            (
                ctr["old_grouping_y_preds"],
                ctr["old_grouping_log_probs"],
                ctr["old_grouping_softmax"],
                ctr["grouping_actions_cache_old"],
            ) = self.get_groupings_old(self.old)
            (
                ctr["new_grouping_y_preds"],
                ctr["new_grouping_log_probs"],
                ctr["new_grouping_softmax"],
                ctr["debug_group2"],
            ) = self.get_groupings(self.new)
            ctr["group_ratio"] = self.cal_ratio(ctr["new_grouping_softmax"]["sample"], ctr["old_grouping_softmax"])
            ctr["old_grouping_param"] = self.get_param(self.old, "grouping")
            ctr["new_grouping_param"] = self.get_param(self.new, "grouping")

            summary.histogram(
                "grouping_actions",
                array_ops.slice(
                    ctr["new_grouping_y_preds"]["sample"], [0, 0], [1, array_ops.shape(self.op_embeddings)[0]]
                ),
            )
            ctr = self.define_baseline_shape(ctr, failing_signal)
            # 计算新的baseline
            new_baseline = self.hparams.bl_dec * ctr["baseline"] + (1 - self.hparams.bl_dec) * math_ops.reduce_mean(
                ctr["reward"]["value"]
            )
            # 是否经常更新baseline
            baseline_mask = math_ops.less(ctr["reward"]["value"], failing_signal)
            selected_reward = array_ops.boolean_mask(ctr["reward"]["value"], baseline_mask)
            selected_baseline = control_flow_ops.cond(
                math_ops.reduce_any(baseline_mask),
                lambda: math_ops.reduce_mean(selected_reward),
                lambda: constant_op.constant(0, dtype=dtypes.float32),
            )
            ctr["pos_reward"] = selected_baseline
            pos_ = math_ops.less(constant_op.constant(0, dtype=dtypes.float32), selected_baseline)
            selected_baseline = self.hparams.bl_dec * ctr["baseline"] + (1 - self.hparams.bl_dec) * selected_baseline
            selected_baseline = control_flow_ops.cond(pos_, lambda: selected_baseline, lambda: ctr["baseline"])
            new_baseline = control_flow_ops.cond(
                math_ops.less(self.global_step, self.hparams.stop_updating_after_steps),
                lambda: new_baseline,
                lambda: selected_baseline,
            )
            # 定义了baseline更新的一个过程
            ctr["baseline_update"] = state_ops.assign(ctr["baseline"], new_baseline, use_locking=True)

            ctr["old_y_preds"], ctr["old_log_probs"], ctr["old_probs"] = self.get_placements_old(self.old)
            ctr["new_y_preds"], ctr["new_log_probs"], ctr["new_probs"] = self.get_placements(self.new)
            (
                ctr["place_ratio"],
                ctr["debug_place_ratio"],
            ) = self.cal_ratio_place(ctr["new_probs"]["target"], ctr["old_probs"]["sample"])
            ctr["old_lstm_param"] = self.get_param(self.old, "lstm")
            ctr["new_lstm_param"] = self.get_param(self.new, "lstm")

            summary.histogram("actions", ctr["new_y_preds"]["sample"])
            mask = math_ops.less(ctr["reward"]["value"], failing_signal)
            suff = ctr["reward"]["value"] - ctr["baseline"]
            # 原本是两个概率分布相乘，现在取log变成相加
            group_ratio = ctr["group_ratio"]
            place_ratio = ctr["place_ratio"]
            ctr, suff = self.calculate_controller_actions_probabilities(ctr, suff, group_ratio, place_ratio)
            ctr["debug_reward"] = {"suff": suff}
            selected_loss = array_ops.boolean_mask(ctr["loss"], mask)
            selected_loss = control_flow_ops.cond(
                math_ops.reduce_any(mask),
                lambda: math_ops.reduce_mean(-selected_loss),
                lambda: constant_op.constant(0, dtype=dtypes.float32),
            )

            ctr["loss"] = control_flow_ops.cond(
                math_ops.less(self.global_step, self.hparams.stop_updating_after_steps),
                lambda: math_ops.reduce_mean(-ctr["loss"]),
                lambda: selected_loss,
            )

            ctr["reward_s"] = math_ops.reduce_mean(ctr["reward"]["value"])
            summary.scalar("loss", ctr["loss"])
            summary.scalar("avg_reward", ctr["reward_s"])
            summary.scalar("best_reward_so_far", best_reward)
            summary.scalar("advantage", math_ops.reduce_mean(ctr["reward"]["value"] - ctr["baseline"]))
        ctr = self.variable_optimizer(ctr)
        ctr = self.update_old_strategy(ctr)
        summary.scalar("gradnorm", ctr["grad_norm"])
        summary.scalar("lr", ctr["lr"])
        ctr["summary"] = summary.merge_all()
        ops["controller"] = ctr
        self.ops = ops
        return ops

    @property
    def global_step(self):
        return self._global_step

    def create_op_embeddings(self):
        """
        create embeddings
        """
        cost_dict = self.get_op_costs()
        output_dict = self.get_op_mem()
        self.cost_d = defaultdict(int, cost_dict)
        self.out_d = defaultdict(list, output_dict)
        self.num_ops = len(self.important_ops)
        # topological sort of important nodes
        topo_order = self.nx_graph.nodes()
        # create index to name for topologicaly sorted important nodes
        name_to_topo_order_index = {}
        for idx, x in enumerate(topo_order):
            name_to_topo_order_index[x] = idx
        self.name_to_topo_order_index = name_to_topo_order_index
        # create adj matrix
        adj_dict, temp = {}, ""
        for name in self.nx_graph.nodes():
            outedges = self.nx_graph.out_edges(name)
            for out_edge in outedges:
                output_op_name = out_edge[1]
                temp = (
                    temp
                    + str(name_to_topo_order_index[name])
                    + " "
                    + str(name_to_topo_order_index[output_op_name])
                    + "\n"
                )
                if name_to_topo_order_index[name] not in adj_dict:
                    adj_dict[name_to_topo_order_index[name]] = []
                adj_dict[name_to_topo_order_index[name]].extend([name_to_topo_order_index[output_op_name], 1])
                if output_op_name not in adj_dict:
                    adj_dict[name_to_topo_order_index[output_op_name]] = []
                adj_dict[name_to_topo_order_index[output_op_name]].extend([name_to_topo_order_index[name], -1])

        # get op_type op_output_shape, and adj info
        file = io.StringIO(temp)
        G = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=None, data=[("weight", int)])
        model = Node2Vec(G, walk_length=2, num_walks=80, p=0.25, q=4, workers=1, use_rejection_sampling=0)
        model.train(window_size=2, iter=50, embed_size=64)
        node_embeddings = model.get_embeddings()
        # TODO(bsteiner): don't filter based on used ops so that we can generalize
        # to models that use other types of ops.
        used_ops = set()
        for node in self.nx_graph.nodes():
            op_type = str(node)
            used_ops.add(op_type)
        self.type_dict = {}
        for op_type in used_ops:
            self.type_dict[op_type] = len(self.type_dict)
        op_types = np.zeros([self.num_ops], dtype=np.int32)
        op_comcost = np.zeros([self.num_ops], dtype=np.float32)
        op_outputs = np.zeros([self.num_ops], dtype=np.float32)
        for idx, node in enumerate(self.nx_graph.nodes()):
            op_types[idx] = self.type_dict[node]
            op_comcost[idx] = self.cost_d[node]
            op_outputs[idx] = sum(self.out_d[node])
        # adj for padding
        op_adj = np.full([self.num_ops, self.hparams.adj_embed_dim], 0, dtype=np.float32)
        for idx in adj_dict:
            neighbors = adj_dict[int(idx)]
            min_dim = min(self.hparams.adj_embed_dim, len(neighbors))
            padding_size = self.hparams.adj_embed_dim - min_dim
            neighbors = neighbors[:min_dim] + [0] * padding_size
            op_adj[int(idx)] = neighbors

        # op_embedding   starts here self.hparams.adj_embed_dim 修改
        op_embeddings = np.zeros([self.num_ops, 3 + 64], dtype=np.float32)
        node2vec = np.zeros([self.num_ops, 64], dtype=np.float32)
        for idx in range(len(topo_order)):
            if str(idx) in node_embeddings:
                node2vec[idx] = node_embeddings[str(idx)]
            node2vec[idx] = -1
        for idx in range(len(topo_order)):
            op_embeddings[idx] = np.concatenate(
                (
                    np.array([op_comcost[idx]]) / np.mean(op_comcost),
                    np.array([op_types[idx]]),
                    np.array([op_outputs[idx]]) / np.mean(op_outputs),
                    np.array(node2vec[idx]),
                )
            )
        self.op_embeddings = constant_op.constant(op_embeddings, dtype=dtypes.float32)

    def get_groupings(self, ppo):
        """
        获取分组信息
        """
        num_children = self.hparams.num_children
        input_layer = self.op_embeddings
        input_layer = array_ops.expand_dims(input_layer, 0)
        feed_ff_input_layer = array_ops.tile(input_layer, [num_children, 1, 1])
        grouping_actions, grouping_log_probs, grouping_softmax = {}, {}, {}
        # 每一次采样都记录样本和概率分布（group_actions和group_probs）
        grouping_actions["sample"], grouping_log_probs["sample"], grouping_softmax["sample"], debug = self.grouping_pro(
            feed_ff_input_layer, ppo
        )
        # cache只是起到缓存作用
        self.grouping_actions_cache = self.grouping_actions_cache_old

        return grouping_actions, grouping_log_probs, grouping_softmax, debug

    def cal_ratio(self, new_softmax, old_softmax):
        grouping_actions_cache = self.grouping_actions_cache
        batch_size = array_ops.shape(grouping_actions_cache)[0]
        ratio1 = math_ops.divide(new_softmax, old_softmax)
        ratio2 = array_ops.reshape(ratio1, [batch_size, -1])
        return ratio2

    def cal_ratio_place(self, new_softmax, old_softmax):
        new_softmax = tf.exp(-new_softmax)
        old_softmax = tf.exp(-old_softmax)
        ratio1 = math_ops.divide(new_softmax, old_softmax)
        return ratio1, {"new_softmax": new_softmax, "old_softmax": old_softmax, "ratio1": ratio1}

    def get_groupings_old(self, ppo):
        """
        获取分组信息
        """
        num_children = self.hparams.num_children
        with variable_scope.variable_scope(f"controller_{self.ctrl_id}"):
            grouping_actions_cache_old = variable_scope.get_local_variable(
                "grouping_actions_cache_old",
                initializer=init_ops.zeros_initializer,
                dtype=dtypes.int32,
                # reuse=tf.AUTO_REUSE,
                # num_ops维度是操作的数量
                shape=[num_children, self.num_ops],
            )
        input_layer = self.op_embeddings
        input_layer = array_ops.expand_dims(input_layer, 0)
        feed_ff_input_layer = array_ops.tile(input_layer, [num_children, 1, 1])
        grouping_actions, grouping_log_probs, grouping_softmax = {}, {}, {}
        # 每一次采样都记录样本和概率分布（group_actions和group_probs）
        (
            grouping_actions["sample"],
            grouping_log_probs["sample"],
            grouping_softmax["sample"],
        ) = self.make_grouping_predictions(feed_ff_input_layer, ppo)
        # cache只是起到缓存作用
        grouping_actions["sample"] = state_ops.assign(grouping_actions_cache_old, grouping_actions["sample"])
        self.grouping_actions_cache_old = grouping_actions_cache_old

        return grouping_actions, grouping_log_probs, grouping_softmax["sample"], grouping_actions_cache_old

    def grouping_pro(self, input_layer, ppo, reuse=None):
        """
        获取分组信息
        """
        with variable_scope.variable_scope(self.hparams.name, reuse=True):
            with variable_scope.variable_scope(ppo, reuse):
                with variable_scope.variable_scope("grouping"):
                    # input_layer: tensor of size [1, num_ops, hidden_size]
                    w_grouping_ff = variable_scope.get_variable("w_grouping_ff")
                    w_grouping_softmax = variable_scope.get_variable("w_grouping_softmax")

        batch_size = array_ops.shape(input_layer)[0]
        embedding_dim = array_ops.shape(input_layer)[2]

        reshaped = array_ops.reshape(input_layer, [batch_size * self.num_ops, embedding_dim])
        # 修改
        ff_output = math_ops.matmul(reshaped, w_grouping_ff)
        out_logs = logits = math_ops.matmul(ff_output, w_grouping_softmax)
        if self.hparams.logits_std_noise > 0:
            num_in_logits = math_ops.cast(array_ops.size(logits), dtype=dtypes.float32)
            avg_norm = math_ops.divide(linalg_ops.norm(logits), math_ops.sqrt(num_in_logits))
            logits_noise = random_ops.random_normal(
                array_ops.shape(logits), stddev=self.hparams.logits_std_noise * avg_norm
            )
            logits = control_flow_ops.cond(
                self.global_step > self.hparams.stop_noise_step, lambda: logits, lambda: logits + logits_noise
            )
        debug1 = logits = array_ops.reshape(logits, [batch_size * self.num_ops, self.num_groups])
        action_label = array_ops.reshape(self.grouping_actions_cache_old, [-1])
        softmax = debug3 = log_probs = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=action_label
        )
        softmax = tf.exp(-softmax)
        debug3 = log_probs = array_ops.reshape(log_probs, [batch_size, -1])
        log_probs = math_ops.reduce_sum(log_probs, 1)
        grouping_actions = self.grouping_actions_cache_old
        grouping_log_probs = log_probs
        return (
            grouping_actions,
            grouping_log_probs,
            softmax,
            {
                "softmax": softmax,
                "debug1": debug1,
                "action_label": action_label,
                "out_logs": out_logs,
                "grouping_log_probs": grouping_log_probs,
                "debug3": debug3,
            },
        )

    def make_grouping_predictions(self, input_layer, ppo, reuse=None):
        """model that predicts grouping (grouping_actions)分组动作.

        Args:
          input_layer: group_input_layer
          reuse: reuse

        Returns:
           grouping_actions: action
           grouping_log_probs: 记录action对应的概率
        """
        with variable_scope.variable_scope(self.hparams.name, reuse=True):
            with variable_scope.variable_scope(ppo, reuse):
                with variable_scope.variable_scope("grouping"):
                    # input_layer: tensor of size [1, num_ops, hidden_size]
                    w_grouping_ff = variable_scope.get_variable("w_grouping_ff")
                    w_grouping_softmax = variable_scope.get_variable("w_grouping_softmax")

        batch_size = array_ops.shape(input_layer)[0]
        embedding_dim = array_ops.shape(input_layer)[2]

        reshaped = array_ops.reshape(input_layer, [batch_size * self.num_ops, embedding_dim])
        # 修改
        ff_output = math_ops.matmul(reshaped, w_grouping_ff)
        logits = math_ops.matmul(ff_output, w_grouping_softmax)
        if self.hparams.logits_std_noise > 0:
            num_in_logits = math_ops.cast(array_ops.size(logits), dtype=dtypes.float32)
            avg_norm = math_ops.divide(linalg_ops.norm(logits), math_ops.sqrt(num_in_logits))
            logits_noise = random_ops.random_normal(
                array_ops.shape(logits), stddev=self.hparams.logits_std_noise * avg_norm
            )
            logits = control_flow_ops.cond(
                self.global_step > self.hparams.stop_noise_step, lambda: logits, lambda: logits + logits_noise
            )
        logits = array_ops.reshape(logits, [batch_size * self.num_ops, self.num_groups])
        actions = random_ops.multinomial(logits, 1, seed=self.hparams.seed)
        actions = math_ops.cast(actions, dtypes.int32)
        actions = array_ops.reshape(actions, [batch_size, self.num_ops])
        action_label = array_ops.reshape(actions, [-1])
        softmax = log_probs = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_label)
        softmax = tf.exp(-softmax)
        log_probs = array_ops.reshape(log_probs, [batch_size, -1])
        log_probs = math_ops.reduce_sum(log_probs, 1)
        grouping_actions = actions
        grouping_log_probs = log_probs
        return (grouping_actions, grouping_log_probs, softmax)

    def create_group_embeddings(self, grouping_actions, verbose=False):
        """Approximating the blocks of a TF graph from a graph_def.

        Args:
          grouping_actions: 分组预测.
          verbose: 打印训练日志.

        Returns:
          groups: 一系列组.
        """
        groups = [self._create_group_embeddings(grouping_actions, i, verbose) for i in range(self.hparams.num_children)]
        return np.stack(groups, axis=0)

    def _create_group_embeddings(self, grouping_actions, child_id, verbose=False):
        """Approximating the blocks of a TF graph from a graph_def for each child.

        Args:
          grouping_actions: grouping predictions.
          child_id: child_id for the group.
          verbose: print stuffs.

        Returns:
          groups: group embedding for the child_id.
        """
        if verbose:
            print("execute input_graph")

        # TODO(azalia): Build inter-adjacencies dag matrix.
        # record dag_matrix
        com_cost = np.zeros([self.num_groups, 1], dtype=np.float32)
        temp = ""
        dag_matrix = np.zeros([self.num_groups, self.num_groups], dtype=np.float32)
        for name in self.nx_graph.nodes():
            topo_op_index = self.name_to_topo_order_index[name]
            group_index = grouping_actions[child_id][topo_op_index]
            outedges = self.nx_graph.out_edges(name)
            for out_edge in outedges:
                output_op_name = out_edge[1]
                if output_op_name not in self.important_op_names:
                    continue
                output_group_index = grouping_actions[child_id][self.name_to_topo_order_index[output_op_name]]
                dag_matrix[group_index, output_group_index] += 1.0
                temp = temp + str(group_index) + " " + str(output_group_index) + "\n"
            # 添加组内的计算值
            com_cost[group_index] = com_cost[group_index] + self.cost_d[name]
        self.dag_matrix = dag_matrix

        # output_shape
        op_output_shapes = np.zeros(
            [len(self.important_ops), self.hparams.max_num_outputs * self.hparams.max_output_size], dtype=np.float32
        )

        # group_embedding
        group_embedding = np.zeros(
            [self.num_groups, len(self.type_dict) + self.hparams.max_num_outputs * self.hparams.max_output_size],
            dtype=np.float32,
        )
        for op_index, name in enumerate(self.nx_graph.nodes()):
            group_index = grouping_actions[child_id][self.name_to_topo_order_index[name]]
            type_name = str(name)
            type_index = self.type_dict[type_name]
            group_embedding[group_index, type_index] += 1
            group_embedding[
                group_index, : self.hparams.max_num_outputs * self.hparams.max_output_size
            ] += op_output_shapes[op_index]

        grouping_adjacencies = np.concatenate([dag_matrix, np.transpose(dag_matrix)], axis=1)
        # 修改
        group_embedding = np.concatenate([group_embedding, com_cost], axis=1)
        group_embedding = np.concatenate([grouping_adjacencies, group_embedding], axis=1)
        group_normalizer = np.amax(group_embedding, axis=1, keepdims=True)
        group_embedding /= group_normalizer + 1.0
        if verbose:
            print("End the execution of the input calculation graph")
        return group_embedding

    def get_placements(self, ppo):
        """
        获取放置信息
        """
        num_children = self.hparams.num_children
        with variable_scope.variable_scope(f"controller_{self.ctrl_id}"):
            actions_cache = variable_scope.get_local_variable(
                f"actions_cache_{ppo}",
                initializer=init_ops.zeros_initializer,
                dtype=dtypes.int32,
                shape=[num_children, self.num_groups],
                trainable=False,
            )
        actions_cache = self.actions_cache_old
        x = self.seq2seq_input_layer
        last_c, last_h, attn_mem = self.encode(x, ppo)
        actions, log_probs, probs = {}, {}, {}
        actions["sample"], log_probs["sample"], probs["sample"] = self.decode(
            last_c, last_h, attn_mem, ppo=ppo, mode="sample"
        )
        actions["target"], log_probs["target"], probs["target"] = self.decode(
            last_c, last_h, attn_mem, ppo=ppo, mode="target", y=actions_cache
        )
        actions["greedy"], log_probs["greedy"], probs["greedy"] = self.decode(
            last_c, last_h, attn_mem, ppo=ppo, mode="greedy"
        )
        actions["sample"] = control_flow_ops.cond(
            self.global_step < self.hparams.stop_sampling,
            lambda: state_ops.assign(actions_cache, actions["sample"]),
            lambda: state_ops.assign(actions_cache, actions["target"]),
        )
        self.actions_cache = actions_cache

        return actions, log_probs, probs

    def get_placements_old(self, ppo):
        """
        获取放置信息
        """
        num_children = self.hparams.num_children
        with variable_scope.variable_scope(f"controller_{self.ctrl_id}"):
            actions_cache_old = variable_scope.get_local_variable(
                "actions_cache_old",
                initializer=init_ops.zeros_initializer,
                dtype=dtypes.int32,
                shape=[num_children, self.num_groups],
                trainable=False,
            )

        x = self.seq2seq_input_layer
        last_c, last_h, attn_mem = self.encode(x, ppo)
        actions, log_probs, probs = {}, {}, {}
        actions["sample"], log_probs["sample"], probs["sample"] = self.decode(
            last_c, last_h, attn_mem, ppo=ppo, mode="sample"
        )
        actions["target"], log_probs["target"], probs["target"] = self.decode(
            last_c, last_h, attn_mem, ppo=ppo, mode="target", y=actions_cache_old
        )
        actions["greedy"], log_probs["greedy"], probs["greedy"] = self.decode(
            last_c, last_h, attn_mem, ppo=ppo, mode="greedy"
        )
        actions["sample"] = control_flow_ops.cond(
            self.global_step < self.hparams.stop_sampling,
            lambda: state_ops.assign(actions_cache_old, actions["sample"]),
            lambda: state_ops.assign(actions_cache_old, actions["target"]),
        )
        self.actions_cache_old = actions_cache_old

        return actions, log_probs, probs

    def encode(self, x, ppo):
        """Encoder using LSTM.

        Args:
          x: tensor of size [num_children, num_groups, embedding_size]

        Returns:
          last_c, last_h: tensors of size [num_children, hidden_size], the final
            LSTM states
          attn_mem: tensor of size [num_children, num_groups, hidden_size], the
          attention
            memory, i.e. concatenation of all hidden states, linearly transformed by
            an attention matrix attn_w_1
        """
        if self.hparams.bi_lstm:
            with variable_scope.variable_scope(self.hparams.name, reuse=True):
                with variable_scope.variable_scope(ppo):
                    with variable_scope.variable_scope("lstm"):
                        w_lstm_forward = variable_scope.get_variable("encoder_lstm_forward")
                        w_lstm_backward = variable_scope.get_variable("encoder_lstm_backward")
                        forget_bias = variable_scope.get_variable("encoder_forget_bias")
                        attn_w_1 = variable_scope.get_variable("attn_w_1")
        else:
            with variable_scope.variable_scope(self.hparams.name, reuse=True):
                with variable_scope.variable_scope(ppo):
                    with variable_scope.variable_scope("lstm"):
                        w_lstm = variable_scope.get_variable("encoder_lstm")
                        forget_bias = variable_scope.get_variable("encoder_forget_bias")
                        attn_w_1 = variable_scope.get_variable("attn_w_1")

        embedding_size = array_ops.shape(x)[2]

        signals = array_ops.split(x, self.num_groups, axis=1)
        for i, signal in enumerate(signals):
            signals[i] = array_ops.reshape(signal, [self.hparams.num_children, embedding_size])

        if self.hparams.bi_lstm:

            def body(i, prev_c_forward, prev_h_forward, prev_c_backward, prev_h_backward):
                """while loop for LSTM."""
                signal_forward = signals[i]
                next_c_forward, next_h_forward = lstm(
                    signal_forward, prev_c_forward, prev_h_forward, w_lstm_forward, forget_bias
                )

                signal_backward = signals[self.num_groups - 1 - i]
                next_c_backward, next_h_backward = lstm(
                    signal_backward, prev_c_backward, prev_h_backward, w_lstm_backward, forget_bias
                )

                next_h = array_ops.concat([next_h_forward, next_h_backward], axis=1)
                all_h.append(next_h)

                return (next_c_forward, next_h_forward, next_c_backward, next_h_backward)

            c_forward = array_ops.zeros(
                [self.hparams.num_children, self.hparams.hidden_size // 2], dtype=dtypes.float32
            )
            h_forward = array_ops.zeros(
                [self.hparams.num_children, self.hparams.hidden_size // 2], dtype=dtypes.float32
            )

            c_backward = array_ops.zeros(
                [self.hparams.num_children, self.hparams.hidden_size // 2], dtype=dtypes.float32
            )
            h_backward = array_ops.zeros(
                [self.hparams.num_children, self.hparams.hidden_size // 2], dtype=dtypes.float32
            )
            all_h = []

            for i in range(0, self.num_groups):
                c_forward, h_forward, c_backward, h_backward = body(i, c_forward, h_forward, c_backward, h_backward)

            last_c = array_ops.concat([c_forward, c_backward], axis=1)
            last_h = array_ops.concat([h_forward, h_backward], axis=1)
            attn_mem = array_ops.stack(all_h)

        else:

            def body(i, prev_c, prev_h):
                signal = signals[i]
                next_c, next_h = lstm(signal, prev_c, prev_h, w_lstm, forget_bias)
                all_h.append(next_h)
                return next_c, next_h

            c = array_ops.zeros([self.hparams.num_children, self.hparams.hidden_size], dtype=dtypes.float32)
            h = array_ops.zeros([self.hparams.num_children, self.hparams.hidden_size], dtype=dtypes.float32)
            all_h = []

            for i in range(0, self.num_groups):
                c, h = body(i, c, h)

            last_c = c
            last_h = h
            attn_mem = array_ops.stack(all_h)

        attn_mem = array_ops.transpose(attn_mem, [1, 0, 2])
        attn_mem = array_ops.reshape(attn_mem, [self.hparams.num_children * self.num_groups, self.hparams.hidden_size])
        attn_mem = math_ops.matmul(attn_mem, attn_w_1)
        attn_mem = array_ops.reshape(attn_mem, [self.hparams.num_children, self.num_groups, self.hparams.hidden_size])

        return last_c, last_h, attn_mem

    def get_variables(self, ppo):
        """
        get variables
        """
        with variable_scope.variable_scope(self.hparams.name, reuse=True):
            with variable_scope.variable_scope(ppo):
                with variable_scope.variable_scope("lstm"):
                    w_lstm = variable_scope.get_variable("decoder_lstm")
                    forget_bias = variable_scope.get_variable("decoder_forget_bias")
                    device_embeddings = variable_scope.get_variable("device_embeddings")
                    device_softmax = variable_scope.get_variable("device_softmax")
                    device_go_embedding = variable_scope.get_variable("device_go_embedding")
                    attn_w_2 = variable_scope.get_variable("attn_w_2")
                    attn_v = variable_scope.get_variable("attn_v")
        return w_lstm, forget_bias, device_embeddings, device_softmax, device_go_embedding, attn_w_2, attn_v

    def decode(self, last_c, last_h, attn_mem, ppo, mode="target", y=None):
        """
        Decoder using LSTM.
        """
        (
            w_lstm,
            forget_bias,
            device_embeddings,
            device_softmax,
            device_go_embedding,
            attn_w_2,
            attn_v,
        ) = self.get_variables(ppo)

        probs = tensor_array_ops.TensorArray(
            dtypes.float32, size=1, dynamic_size=True, infer_shape=False, clear_after_read=False
        )

        actions = tensor_array_ops.TensorArray(
            dtypes.int32, size=self.num_groups, infer_shape=False, clear_after_read=False
        )

        # pylint: disable=unused-argument
        def condition(i, *args):
            return math_ops.less(i, self.num_groups)

        # pylint: disable=missing-docstring
        def body(i, prev_c, prev_h, actions, log_probs, probs):
            # pylint: disable=W0105
            signal = control_flow_ops.cond(
                math_ops.equal(i, 0),
                lambda: array_ops.tile(device_go_embedding, [self.hparams.num_children, 1]),
                lambda: embedding_ops.embedding_lookup(device_embeddings, actions.read(i - 1)),
            )
            if self.hparams.keep_prob is not None:
                signal = nn_ops.dropout(signal, self.hparams.keep_prob)
            next_c, next_h = lstm(signal, prev_c, prev_h, w_lstm, forget_bias)
            query = math_ops.matmul(next_h, attn_w_2)
            query = array_ops.reshape(query, [self.hparams.num_children, 1, self.hparams.hidden_size])
            query = math_ops.tanh(query + attn_mem)
            query = array_ops.reshape(query, [self.hparams.num_children * self.num_groups, self.hparams.hidden_size])
            query = math_ops.matmul(query, attn_v)
            query = array_ops.reshape(query, [self.hparams.num_children, self.num_groups])
            query = nn_ops.softmax(query)
            query = array_ops.reshape(query, [self.hparams.num_children, self.num_groups, 1])
            query = math_ops.reduce_sum(attn_mem * query, axis=1)
            query = array_ops.concat([next_h, query], axis=1)
            logits = math_ops.matmul(query, device_softmax)
            logits /= self.hparams.temperature
            logits = math_ops.tanh(logits) * self.hparams.tanh_constant
            num_in_logits = math_ops.cast(array_ops.size(logits), dtype=dtypes.float32)
            avg_norm = math_ops.divide(linalg_ops.norm(logits), math_ops.sqrt(num_in_logits))
            logits_noise = random_ops.random_normal(
                array_ops.shape(logits), stddev=self.hparams.logits_std_noise * avg_norm
            )
            logits = control_flow_ops.cond(
                self.global_step > self.hparams.stop_noise_step, lambda: logits, lambda: logits + logits_noise
            )

            if mode == "sample":
                next_y = random_ops.multinomial(logits, 1, seed=self.hparams.seed)
            elif mode == "greedy":
                next_y = math_ops.argmax(logits, 1)
            elif mode == "target":
                next_y = array_ops.slice(y, [0, i], [-1, 1])
            else:
                raise NotImplementedError
            next_y = math_ops.cast(next_y, dtypes.int32)
            next_y = array_ops.reshape(next_y, [self.hparams.num_children])
            actions = actions.write(i, next_y)
            softpros = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=next_y)
            probs = probs.write(i, softpros)
            log_probs += softpros
            return i + 1, next_c, next_h, actions, log_probs, probs

        loop_vars = [
            constant_op.constant(0, dtype=dtypes.int32),
            last_c,
            last_h,
            actions,
            array_ops.zeros([self.hparams.num_children], dtype=dtypes.float32),
            probs,
        ]
        loop_outputs = control_flow_ops.while_loop(condition, body, loop_vars)

        actions = loop_outputs[-3].stack()
        actions = array_ops.transpose(actions, [1, 0])
        log_probs = loop_outputs[-2]
        probs = loop_outputs[-1].stack()
        probs = array_ops.transpose(probs, [1, 0])
        return actions, log_probs, probs

    def eval_grouping_and_scheduling_sim(self, sess, child_id=0):
        """
        执行模拟过程
        """
        grouping_actions_old, actions_old = sess.run([self.grouping_actions_cache_old, self.actions_cache_old])
        grouping_actions = grouping_actions_old[child_id]
        actions = actions_old[child_id]
        for name in self.nx_graph.nodes():
            topo_order_index = self.name_to_topo_order_index[name]
            group_index = grouping_actions[topo_order_index]
            self.nx_graph.nodes[name]["device"] = self.devices[actions[group_index]].name
        try:
            available_devices = [device.name for device in self.devices]
            ungrouped_pl = {}
            ungrouped_pl_ = {}
            for node in self.nx_graph.nodes():
                ungrouped_pl[node] = available_devices.index(self.nx_graph.nodes[node]["device"])
            ungrouped_pl_ = ungrouped_pl.copy()
            ios = ImportantOpsRewarder(self.nx_graph, available_devices)
            start_time = time.time()
            run_time, peak_utils, _, comm_utils = ios.simulate(ungrouped_pl, sim_mem_usage=True, sim_com_usage=True)
            slim_time = time.time() - start_time
        except errors.ResourceExhaustedError:
            run_time = self.hparams.failing_signal
            peak_utils = self.hparams.failing_signal
            comm_utils = self.hparams.failing_signal
            print("eval placement error!!!")
        return run_time / 1e6, peak_utils, comm_utils, slim_time, ungrouped_pl, ungrouped_pl_

    def update_reward(self, sess, run_time, child_id=0):
        """
        更新奖励reward
        """
        reward = self.compute_reward(run_time)
        controller_ops = self.ops["controller"]
        _, best_reward = sess.run(
            [
                controller_ops["reward"]["update"][child_id],
                controller_ops["best_reward"]["update"][child_id],
            ],
            feed_dict={
                controller_ops["reward"]["ph"][child_id]: reward,
            },
        )
        print((f"original reward={run_time:.5f} reward={reward:.5f} The best award in history={best_reward:.5f}"))

        # Reward is a double, best_reward a float: allow for some slack in the
        # comparison.
        updated = abs(best_reward - reward) < 1e-6
        return updated

    def partitioning(self, sess):
        """
        进行分组
        """
        controller_ops = self.ops["controller"]
        grouping_actions = sess.run([controller_ops["old_grouping_y_preds"]["sample"]])
        return grouping_actions[0]

    def scheduling(self, grouping, sess):
        """
        进行调度
        """
        controller_ops = self.ops["controller"]
        feed_seq2seq_input_dict = {}
        feed_seq2seq_input_dict[self.seq2seq_input_layer] = grouping
        sess.run([controller_ops["old_y_preds"]["sample"]], feed_dict=feed_seq2seq_input_dict)

    def update_ppo(self, sess):
        """
        ppo梯度优化
        """
        controller_ops = self.ops["controller"]
        sess.run([controller_ops["new_grouping_param"], controller_ops["old_grouping_param"]])
        sess.run([controller_ops["update_old_grouping_op"], controller_ops["update_old_lstm_op"]])

    def process_reward(self, sess, METHOD):
        """
        计算reward
        """
        controller_ops = self.ops["controller"]
        if METHOD["name"] == "kl_pen":
            kl = sess.run(controller_ops["kl_mean"], {controller_ops["tflam"]: METHOD["lam"]})
            if kl < METHOD["kl_target"] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD["lam"] /= 2
            elif kl > METHOD["kl_target"] * 1.5:
                METHOD["lam"] *= 2
            METHOD["lam"] = np.clip(METHOD["lam"], 1e-4, 10)
        run_ops = [
            controller_ops["loss"],
            controller_ops["lr"],
            controller_ops["grad_norm"],
            controller_ops["grad_norms"],
            controller_ops["debug"],
            controller_ops["train_op"],
            controller_ops["debug_reward"],
        ]
        sess.run(run_ops)
        sess.run(controller_ops["baseline_update"])

    def _get_train_ops(
        self,
        loss,
        tf_variables,
        global_step,
        grad_bound=1.25,
        lr_init=1e-3,
        lr_dec=0.9,
        start_decay_step=10000,
        decay_steps=100,
        optimizer_type="adam",
    ):
        """Loss optimizer.

        Args:
          loss: scalar tf tensor
          tf_variables: 训练变量的列表
            tf.compat.v1.trainable_variables()
          global_step: 全局step
          grad_bound: 最大梯度norm
          lr_init: 初始化学习率
          lr_dec: 学习率衰减系数，和adam算法有关
          start_decay_step: 经过多少步之后，学习速率开始下降
          decay_steps: 在这个步骤间隔应用衰减率因子
          optimizer_type: 优化方式adam还是SGD

        Returns:
          train_op: 训练操作
          learning_rate: 学习速率tensor
          grad_norm: l2梯度向量正则项
          all_grad_norms: 每个部分的l2范数
        """
        lr_gstep = global_step - start_decay_step

        def f1():
            return constant_op.constant(lr_init)

        def f2():
            return learning_rate_decay.exponential_decay(lr_init, lr_gstep, decay_steps, lr_dec, True)

        learning_rate = control_flow_ops.cond(
            math_ops.less(global_step, start_decay_step), f1, f2, name="learning_rate"
        )

        if optimizer_type == "adam":
            opt = adam.AdamOptimizer(learning_rate)
        elif optimizer_type == "sgd":
            opt = gradient_descent.GradientDescentOptimizer(learning_rate)
        grads_and_vars = opt.compute_gradients(loss, tf_variables)
        grad_norm = clip_ops.global_norm([g for g, v in grads_and_vars])
        all_grad_norms = {}
        clipped_grads = []
        clipped_rate = math_ops.maximum(grad_norm / grad_bound, 1.0)
        for g, v in grads_and_vars:
            if g is not None:
                if isinstance(g, tf_ops.IndexedSlices):
                    clipped = g.values / clipped_rate
                    norm_square = math_ops.reduce_sum(clipped * clipped)
                    clipped = tf_ops.IndexedSlices(clipped, g.indices)
                else:
                    clipped = g / clipped_rate
                    norm_square = math_ops.reduce_sum(clipped * clipped)
                all_grad_norms[v.name] = math_ops.sqrt(norm_square)
                clipped_grads.append((clipped, v))

        train_op = opt.apply_gradients(clipped_grads, global_step)
        return (
            train_op,
            learning_rate,
            grad_norm,
            all_grad_norms,
            {"grads_and_vars": grads_and_vars, "clipped_grads": clipped_grads, "tf_variables": tf_variables},
        )

    def get_param(self, ppo, part):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.hparams.name + "/" + ppo + "/" + part)


def lstm(x, prev_c, prev_h, w_lstm, forget_bias):
    """LSTM cell.

    Args:
      x: tensors of size [num_children, hidden_size].
      prev_c: tensors of size [num_children, hidden_size].
      prev_h: same as prev_c.
      w_lstm: .
      forget_bias: .

    Returns:
      next_c:
      next_h:
    """
    ifog = math_ops.matmul(array_ops.concat([x, prev_h], axis=1), w_lstm)
    i, f, o, g = array_ops.split(ifog, 4, axis=1)
    i = math_ops.sigmoid(i)
    f = math_ops.sigmoid(f + forget_bias)
    o = math_ops.sigmoid(o)
    g = math_ops.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * math_ops.tanh(next_c)
    return next_c, next_h
