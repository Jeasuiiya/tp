import copy
import time
import os
from collections import deque
from heapq import heappushpop

try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import numpy as np
from framework.aware.progressive_graph import ProgressiveGraph
from framework.aware.progressive_nn import MessagePassingProgressiveNN
from framework.tools import log

__doc__ = "The placer of aware algorithm, including node embedding and reinforcement learning"


tf.disable_eager_execution()


class ProgressivePlacer:
    """
    The placer of aware algorithm, including node embedding and reinforcement learning
    """

    def __init__(self, nx_graph, config_params, simulator_params, aware_simuator):
        self.nx_graph = nx_graph
        self.max_rounds = self.nx_graph.number_of_nodes()
        self.config_params = config_params
        self.simulator_params = simulator_params
        self.aware_simuator = aware_simuator
        self.best_runtimes = []
        self.n_max_best_runtimes = 5
        self.epsiode_to_placement = {}
        self.init_placement = None
        self.progressive_graph = ProgressiveGraph(
            self.nx_graph,
            self.config_params["n_devs"],
            self.config_params["node_traversal_order"],
            seed=self.config_params["seed"],
        )
        self.model = MessagePassingProgressiveNN(
            progressive_graph=self.progressive_graph,
            config_params=self.config_params,
            simulator_params=self.simulator_params,
        )
        self.model_saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)

        self.best_placement = {}

    def place(self):
        """
        The body of aware placement (embedding and reinforcement learning)
        """
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(self.model.init_global_step)
            rand_placement = self.progressive_graph.get_random_placement(seed=0)

            best_reward_runtime = 1e9
            best_placement = rand_placement
            best_runtime = 1e9
            for episode in range(self.config_params["n_eps"]):
                self.init_placement = rand_placement
                is_eval_episode = episode % self.config_params["eval_freq"] == 0
                is_save_episode = episode % self.config_params["save_freq"] == 0 and episode > 0

                (
                    _,
                    _,
                    _,
                    run_times,
                    memory_utils,
                    _,
                    _,
                    placements,
                ) = self.run_episode(sess, is_eval_episode)

                batched_cum_rewards = []
                reward_runtimes = []
                cum_rewards, reward_runtime = self.compute_rewards(run_times, memory_utils)
                batched_cum_rewards.append(cum_rewards)
                reward_runtimes.append(reward_runtime)

                (
                    ep_best_placement,
                    ep_best_reward_runtime,
                    ep_best_placement_runtime,
                    _,
                ) = self.identify_best_pl(reward_runtimes, run_times, memory_utils, placements)

                if ep_best_reward_runtime < best_reward_runtime and not is_eval_episode:
                    best_reward_runtime = ep_best_reward_runtime
                    best_placement = ep_best_placement
                    best_runtime = ep_best_placement

                self.update_best_pl(ep_best_reward_runtime, ep_best_placement, episode, is_eval_episode)
                log.debug("Episode best pl runtime %s", ep_best_placement_runtime)
                log.debug("Episode best pl (simplified) %s", [best_placement[node] for node in self.nx_graph.nodes()])

                self.best_placement = best_placement

                if is_save_episode:
                    self.save_model(sess, episode)

                batched_cum_rewards = np.float32(batched_cum_rewards)
                avg_cum_rewards = np.mean(batched_cum_rewards, axis=0)
                rnd_cum_rewards = [deque(maxlen=self.config_params["bl_n_rnds"]) for _ in range(self.max_rounds)]
                for i, avg_cum_reward in enumerate(avg_cum_rewards):
                    rnd_cum_rewards[i].append(avg_cum_reward)

                os.makedirs(os.path.join(self.config_params["output_save_path"], "summary"), exist_ok=True)
                with open(
                    self.config_params["output_save_path"] + "/summary/summary_{episode}.txt", "a+", encoding="utf-8"
                ) as f:
                    f.write("\nEpisode: " + str(episode))
                    f.write("\nep best runtime: " + str(ep_best_placement_runtime))
                    f.write("\nbest so far: " + str(best_runtime))
                    f.write("\nep best reward runtime: " + str(ep_best_reward_runtime))
                    f.write("\nbest reward runtime: " + str(best_reward_runtime))
                    f.write("\n------------\n")

    def run_episode(self, sess, is_eval_episode):
        """
        update the best results after every episode
        """
        self.progressive_graph.reset_placement(self.init_placement)  # 将所有的设备都设置成init_pl
        self.progressive_graph.new_episode()  # 将所有的节点和当前节点的flag标记为unseen

        start_time = np.array([[-1] * self.progressive_graph.nx_graph.number_of_nodes()])
        run_time, start_time, output_memory_peak_utils = self.eval_placement()
        self.progressive_graph.set_start_times(start_time)

        # if (1 + episode) % self.config_params["print_freq"] == 0:
        #     groups = set(range(self.config_params["n_devs"]))
        #     nodesToPrint = self.nx_graph.nodes
        #     mappingToPrint = dict(zip(sorted(groups), count()))
        #     colorsToPrint = [mappingToPrint[self.nx_graph.nodes[node]['placement']] for node in nodesToPrint]

        #     file_path = open(
        #         self.config_params["output_save_path"] + "/pkl/graph_data_{}.pkl".format(str(episode)), 'wb')
        #     pickle.dump(
        #         {"Gg": self.nx_graph, "nodesToPrint": nodesToPrint, "colorsToPrint": colorsToPrint}, file_path)

        episode_best_time = run_time
        episode_best_placement_memory = output_memory_peak_utils
        episode_best_pl = self.init_placement
        run_times = []
        memory_utils = []
        states = []
        explor_acts = []
        placements = [self.init_placement]

        run_times.append(run_time)
        memory_utils.append(output_memory_peak_utils)

        nn_time = 0
        s1 = time.time()
        for i_round in range(self.max_rounds):
            nodes = self.progressive_graph.node_traversal_order
            node = nodes[i_round]
            self.progressive_graph.set_curr_node(node)

            s2 = time.time()
            device, lo, feed, expl, train_outs = self.get_improvement(sess, node, is_eval_episode)
            nn_time += time.time() - s2

            explor_acts.append(expl)
            self.progressive_graph.refresh(node, device[0])
            placements.append(self.progressive_graph.get_placement())

            run_time, start_time, output_memory_peak_utils = self.eval_placement()
            run_times.append(run_time)
            memory_utils.append(output_memory_peak_utils)
            states.append([feed, device, lo, train_outs])
            self.progressive_graph.update_start_time(start_time)
            self.progressive_graph.inc_done_node(node)

        for i, rnd_rt in enumerate(run_times):
            if episode_best_time > rnd_rt:
                episode_best_time = rnd_rt
                episode_best_placement_memory = memory_utils[i]
                episode_best_pl = placements[i]

        run_times = np.transpose(run_times)
        memory_utils = np.array(memory_utils)

        log.debug("Total time: %s   NN time: %s", time.time() - s1, nn_time)

        return (
            episode_best_pl,
            episode_best_time,
            episode_best_placement_memory,
            run_times,
            memory_utils,
            states,
            explor_acts,
            placements,
        )

    def eval_placement(self, placement=None):
        if placement is None:
            placement = copy.copy(self.progressive_graph.get_placement())
        run_time, start_time, mem_util = self.aware_simuator.simulate(placement)
        return run_time, start_time, mem_util

    def get_improvement(self, sess, node, is_eval_episode):
        """
        Try to get the better results
        """
        feed = self.model.get_feed_dict(self.progressive_graph, node)
        if is_eval_episode:
            feed[self.model.is_eval_ph] = 1.0

        train_ops = [
            self.model.logprob_grad_outs,
            self.model.ent_grad_outs,
            self.model.log_probs,
            self.model.sample,
            self.model.pl_ent_loss,
            self.model.log_prob_loss,
            self.model.no_noise_classifier,
            self.model.entropy,
            self.model.ent_dec,
        ]

        s, lo, _, expl, *train_outs = sess.run(
            self.model.get_eval_ops() + [self.model.expl_act] + train_ops, feed_dict=feed
        )

        return s, lo, feed, expl, train_outs

    def compute_rewards(self, run_times, memory_utils):
        """
        compute the reward of every placements
        """
        run_times = np.log(run_times)

        for i, memory_util in enumerate(memory_utils):
            memory_excess = max(memory_util) / 1e9 - self.config_params["max_mem"]
            memory_excess = max(0, memory_excess)
            run_times[i] += self.config_params["mem_penalty"] * memory_excess
            run_times[i] = min(run_times[i], self.config_params["max_runtime_mem_penalized"])

        cum_rewards = []
        for i in range(len(run_times) - 1):
            cum_rewards.append([run_times[i + 1] - run_times[i]])
        for i in reversed(range(len(cum_rewards) - 1)):
            cum_rewards[i] += self.config_params["disc_factor"] * cum_rewards[i + 1][0]

        return cum_rewards, run_times

    def identify_best_pl(self, reward_runtimes, run_times, memory_utils, placements):
        """
        get the best placements in all placements
        """
        best_rew_runtime = 1e20
        best_placement = None
        best_placement_runtime = []
        best_placement_memory = []

        min_reward = np.min(reward_runtimes)
        min_idx = np.argmin(reward_runtimes)

        if best_rew_runtime > min_reward:
            best_rew_runtime = min_reward
            best_placement = placements[min_idx]
            best_placement_runtime = run_times[min_idx]
            best_placement_memory = max(memory_utils[min_idx])

        return best_placement, best_rew_runtime, best_placement_runtime, best_placement_memory

    def update_best_pl(self, ep_best_reward_runtime, ep_best_placement, episode, is_eval_episode):
        if len(self.best_runtimes) < self.n_max_best_runtimes:
            self.epsiode_to_placement[episode] = [ep_best_placement, is_eval_episode]
        elif -self.best_runtimes[0][0] > ep_best_reward_runtime:
            _, del_episode = heappushpop(self.best_runtimes, (-ep_best_reward_runtime, episode))
            self.epsiode_to_placement.pop(del_episode, None)
            self.epsiode_to_placement[episode] = [ep_best_placement, is_eval_episode]

    def save_model(self, sess, episode):
        save_path = self.config_params["output_save_path"] + "/model"
        self.model_saver.save(sess, save_path, global_step=episode, write_meta_graph=False)
        log.debug("Saved model at %s", save_path)
