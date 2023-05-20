from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.training import training
from tensorflow.core.protobuf import rewriter_config_pb2
from framework.trinity import trinity_program

# tf.disable_v2_behavior()
__doc__ = "The Method For Graph Scheduling "


def compute_rewards(run_times, mem_utils, com_utils, isall=False):
    """
    A method to compute rewards
    """
    reward = 0
    print("scheduling time: " + str(run_times))
    print("scheduling peak memory:" + str(max(mem_utils) / 1073741824))
    print("scheduling communication cost:" + str(max(com_utils) / 1073741824))
    if isall:
        mem_excess = max(mem_utils) / 1073741824
        # 7600表示传输速度
        com_excess = max(com_utils) / 1073741824
        if mem_excess > 11:
            mem_excess = mem_excess * 10
        else:
            mem_excess = 0
        # else:
        #     mem_excess = mem_excess * 0.1
        reward += 0.0 * mem_excess + 0.0 * com_excess + run_times
        return reward
    return run_times


class GraphScheduling:
    """将提供的MetaGraph进行放置.
    Args:
      nx_graph: 需要进行放置的nx_graph.
      cluster: 一组可选的硬件资源，用于优化布局。
        如果没有指定，那么将根据本地资源进行自动创建
      allotted_time: 花费在优化放置上的最长时间（以秒作为单位）
      hparams: 寻找最佳放置的超参数.
      verbose: 如果为True那么将会输出调试信息.

    Returns:
      将会返回已经配置好放置后的MetaGraph
    """

    # 度量使用原始位置可实现的运行时。
    def __init__(self, nx_graph, cluster, hparams, verbose, step=500, issim=True, isbase=True):
        self.issim = issim
        self.isbase = isbase
        self.nx_graph = nx_graph
        self.cluster = cluster
        # self.allotted_time = allotted_time
        # 超参数
        self.hparams = hparams
        self.ungrouped_pl = None
        self.ungrouped_pl_ = None
        # 打印日志
        self.verbose = verbose
        self.model = None
        self.step = step
        self.run_time = []
        self.mem = []
        self.comm = []
        self.run_time_list = []
        self.comm_list = []
        self.best_com_list = []
        self.best_mem_list = []
        self.mem_list = []
        self.best_run_time_list = []
        self.all_time = 0
        self.train_time_list = []
        self.reward = 0
        self.run_time_sim = []
        self.run_time_sim_list = []
        self.real_time_list = []
        self.sim_time_list = []
        self.reward_list = []
        self.children = {}
        if cluster is None:
            cluster = gcluster.Cluster(allow_soft_placement=False, disable_detailed_stats=False, disable_timeline=False)
        self.original_run_time = self.hparams.failing_signal
        self.best_run_time = self.original_run_time

    def schedule_graph(self, isall=False):
        """
        Build trinity model and start graph schedule
        """
        with tf_ops.Graph().as_default():
            # 将所有节点都放置在CPU上，我们不想让他们和模型的放置优化做竞争
            # Place all the nodes of the controller on the CPU. We don't want them to
            # fight for accelerator memory with the model to optimize.
            # 为了防止影响其他GPU对模型性能的测评，将有关策略搜索的程序都运行在CPU上
            with tf_ops.device("/device:CPU:0"):
                # Trinity分层架构的程序
                self.model = trinity_program.TrinityProgram(self.hparams, self.nx_graph, self.cluster)
                self.model.build_program()
                config_proto = config_pb2.ConfigProto()
                off = rewriter_config_pb2.RewriterConfig.OFF
                config_proto.graph_options.rewrite_options.arithmetic_optimization = off
                session_creator = training.ChiefSessionCreator(config=config_proto)
                with training.MonitoredSession(session_creator=session_creator) as sess:
                    # while current_time - start_time < self.allotted_time:
                    current_step = 0
                    while current_step < self.step:
                        # 首先对神经网络进行切分操作
                        if current_step % 5 == 0:
                            self.model.update_ppo(sess)
                            # 再对神经网络进行分组操作
                            partitioning_actions = self.model.partitioning(sess)
                            input_to_seq2seq = self.model.create_group_embeddings(partitioning_actions, verbose=False)
                            self.model.scheduling(input_to_seq2seq, sess)
                            for child_id in range(0, self.hparams.num_children):
                                (
                                    self.run_time,
                                    self.mem,
                                    self.comm,
                                    sim_time,
                                    ungrouped_pl,
                                    ungrouped_pl_,
                                ) = self.model.eval_grouping_and_scheduling_sim(sess, child_id=child_id)
                                if not child_id:
                                    (
                                        self.children["run_time"],
                                        self.children["mem"],
                                        self.children["comm"],
                                        self.children["sim_time"],
                                        self.children["ungrouped_pl"],
                                        self.children["ungrouped_pl_"],
                                    ) = ([], [], [], [], [], [])
                                self.children["run_time"].append(self.run_time)
                                self.children["mem"].append(self.mem)
                                self.children["comm"].append(self.comm)
                                self.children["sim_time"].append(sim_time)
                                self.children["ungrouped_pl"].append(ungrouped_pl)
                                self.children["ungrouped_pl_"].append(ungrouped_pl_)
                            # print(ungrouped_pl)
                            self.run_time_sim = self.run_time
                            self.sim_time_list.append(sim_time)
                            self.run_time_sim_list.append(self.run_time)
                            self.comm_list.append(str(max(self.comm) / 1e9))
                            self.mem_list.append(str(max(self.mem) / 1e9))
                            print("Simulation environment (execution time):", sim_time)
                            for child_id in range(0, self.hparams.num_children):
                                reward = compute_rewards(
                                    self.children["run_time"][child_id],
                                    self.children["mem"][child_id],
                                    self.children["comm"][child_id],
                                    isall,
                                )
                                if not child_id:
                                    self.children["reward"] = []
                                self.children["reward"].append(reward)
                                updated = self.model.update_reward(sess, reward, child_id=child_id)
                                self.reward_list.append(reward)
                                if updated and self.children["run_time"][child_id] < self.best_run_time:
                                    self.best_run_time = self.children["run_time"][child_id]
                                    print("------------------Search for the best scheduling strategy------------------")
                                    print("Optimal scheduling time: " + str(self.children["run_time"][child_id]))
                                    print(
                                        "Scheduling peak memory at this time:"
                                        + str(max(self.children["mem"][child_id]) / 1e9)
                                    )
                                    print(
                                        "Scheduling total communication at this time:"
                                        + str(max(self.children["comm"][child_id]) / 1e9)
                                    )
                                    print("-----------------------------------------------------------------------")
                                    self.best_run_time_list.append(self.best_run_time)
                                    self.best_com_list.append(self.comm)
                                    self.best_mem_list.append(self.mem)
                                    self.ungrouped_pl = ungrouped_pl
                        # 采用模拟执行引擎
                        self.model.process_reward(sess, self.model.METHOD)
                        current_step = current_step + 1
        return ungrouped_pl
