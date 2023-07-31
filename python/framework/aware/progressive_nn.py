try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.training import adam, rmsprop, gradient_descent
from framework.aware.progressive_fnn import RadioMessenger, Aggregator, Classifier
import numpy as np

__doc__ = "The main nerual network of aware"


class ReinforceAgent:
    """
    The third level of embedding layers
    """

    def __init__(self, config_params, simulator_params):
        self.config_params = config_params
        self.simulator_params = simulator_params
        self.bs = 1
        self.global_step = tf.train.get_or_create_global_step()
        self.init_global_step = tf.assign(self.global_step, 0)
        self.is_eval_ph = tf.placeholder_with_default(0.0, None)
        self.lr_start_decay_step = int(self.simulator_params["lr_start_decay_step"])
        self.lr_gstep: int = self.global_step - self.lr_start_decay_step
        self.ent_start_dec_step = int(self.simulator_params["ent_start_dec_step"])
        self.ent_gstep = self.global_step - self.ent_start_dec_step
        self.lr, self.grad_norm, self.gradphs_and_vars = None, None, None

        if self.simulator_params["ent_dec_approach"] == "exponential":
            self.ent_dec_func = (
                tf.train.exponential_decay(
                    self.simulator_params["ent_dec_init"],
                    self.ent_gstep,
                    self.simulator_params["ent_dec_steps"],
                    self.simulator_params["ent_dec"],
                    False,
                ),
            )
        elif self.simulator_params["ent_dec_approach"] == "linear":
            self.ent_dec_func = tf.train.polynomial_decay(
                self.simulator_params["ent_dec_init"],
                self.ent_gstep,
                self.simulator_params["ent_dec_lin_steps"],
                self.simulator_params["ent_dec_min"],
            )
        elif self.simulator_params["ent_dec_approach"] == "step":
            self.ent_dec_func = tf.constant(self.simulator_params["ent_dec_min"])

        ent_dec = tf.cond(
            tf.less(self.global_step, self.ent_start_dec_step),
            lambda: tf.constant(self.simulator_params["ent_dec_init"]),
            lambda: self.ent_dec_func,
            name="ent_decay",
        )
        self.ent_dec = tf.maximum(ent_dec, self.simulator_params["ent_dec_min"])

    def _sample(self, classifier):
        sample_argmax = tf.argmax(classifier, axis=-1)
        sample = tf.random.categorical(classifier, 1, seed=self.config_params["seed"])
        sample = tf.reshape(tf.cast(sample, tf.int32), [-1])
        sample_argmax = tf.reshape(tf.cast(sample_argmax, tf.int32), [-1])
        # use during eval phase
        expl_act = tf.logical_not(tf.equal(sample, sample_argmax))
        log_probs = -1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classifier, labels=sample)

        return sample, log_probs, expl_act

    def _get_entropy(self, classifier):
        # with tf.name_scope('Entropy_logits'):
        p = tf.nn.softmax(classifier)
        lp = tf.math.log(p + 1e-3)
        entropy = -1 * p * lp
        entropy = tf.reduce_sum(entropy, axis=-1)
        return entropy

    def _build_train_ops(self, log_prob_loss, pl_ent_loss, grad_bound=1.25):
        tf_variables = tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = self._get_optimizer()

        # print some ent, adv stats
        b_grads = []
        for i in range(self.bs):
            with tf.variable_scope("log_prob_grads"):
                grads_and_vars = optimizer.compute_gradients(log_prob_loss[i], tf_variables)
            b_grads.append(grads_and_vars)
        logprob_grad_outs = [[g for g, _ in b_grads[i] if g is not None] for i in range(self.bs)]

        # print some ent, adv stats
        b_grads2 = []
        for i in range(self.bs):
            with tf.variable_scope("placement_ent_grads"):
                grads_and_vars2 = optimizer.compute_gradients(pl_ent_loss[i], tf_variables)
            b_grads2.append(grads_and_vars2)
        ent_grad_outs = [[g for g, _ in b_grads2[i] if g is not None] for i in range(self.bs)]

        grad_phs = []
        gradphs_and_vars = []
        for i, [g, v] in enumerate(grads_and_vars):
            if g is not None:
                grad_vtype = tf.float32
                if v.dtype == tf.as_dtype("float16_ref"):
                    grad_vtype = tf.float16
                p = tf.placeholder(grad_vtype, name=f"grad_phs_{i}")
                grad_phs.append(p)
                gradphs_and_vars.append((p, v))

        self.grad_norm = tf.linalg.global_norm([tf.cast(g, tf.float32) for g in grad_phs])
        self.gradphs_and_vars = gradphs_and_vars

        clipped_grads = self._clip_grads_and_vars(gradphs_and_vars, grad_bound)
        train_op = optimizer.apply_gradients(clipped_grads, self.global_step)

        return train_op, logprob_grad_outs, ent_grad_outs

    def _clip_grads_and_vars(self, grads_and_vars, grad_bound):
        clipped_grads = []
        clipped_rate = tf.maximum(self.grad_norm / grad_bound, 1.0)

        for grad, var in grads_and_vars:
            if grad is not None:
                clipped = grad / tf.cast(clipped_rate, grad.dtype)
                clipped_grads.append((clipped, var))

        return clipped_grads

    def _get_optimizer(self):
        self.setup_lr()
        # tf.summary.scalar('lr', self.lr)
        optimizer_type = self.simulator_params["optimizer_type"]
        if optimizer_type == "adam":
            opt = adam.AdamOptimizer(self.lr)
        elif optimizer_type == "sgd":
            opt = gradient_descent.GradientDescentOptimizer(self.lr)
        elif optimizer_type == "rmsprop":
            opt = rmsprop.RMSPropOptimizer(self.lr)
        return opt

    def setup_lr(self):
        """
        set the learning rate of network
        """

        def constant_func():
            return tf.constant(self.simulator_params["lr_init"])

        def poly_func():
            return tf.train.polynomial_decay(
                self.simulator_params["lr_init"],
                self.lr_gstep,
                self.simulator_params["lr_decay_steps"],
                self.simulator_params["lr_min"],
            )

        def exp_func():
            return tf.train.exponential_decay(
                self.simulator_params["lr_init"],
                self.lr_gstep,
                self.simulator_params["lr_decay_steps"],
                self.simulator_params["lr_dec"],
                True,
            )

        if self.simulator_params["lr_dec_approach"] == "linear":
            learning_rate = tf.cond(
                pred=tf.less(self.global_step, self.lr_start_decay_step),
                true_fn=constant_func,
                false_fn=poly_func,
                name="learning_rate",
            )
        else:
            learning_rate = tf.cond(
                pred=tf.less(self.global_step, self.lr_start_decay_step),
                true_fn=constant_func,
                false_fn=exp_func,
                name="learning_rate",
            )

        self.lr = tf.maximum(learning_rate, self.simulator_params["lr_min"])


class ProgressiveNN(ReinforceAgent):
    """
    The second level of embedding layers
    """

    def __init__(self, config_params, simulator_params):
        ReinforceAgent.__init__(self, config_params, simulator_params)
        self.config_params = config_params
        self.classifier, self.sample, self.log_probs, self.expl_act, self.entropy = None, None, None, None, None
        (
            self.pl_ent_loss,
            self.log_prob_loss,
            self.train_op,
            self.grad_outs,
            self.logprob_grad_outs,
            self.ent_grad_outs,
        ) = (None, None, None, None, None, None)

    def build_train_ops(self, classifier):
        self.classifier = classifier
        self.sample, self.log_probs, self.expl_act = self._sample(classifier)
        self.entropy = self._get_entropy(classifier)
        # note that loss resamples instead of reading from ph
        self.pl_ent_loss = self.entropy * -self.simulator_params["ent_dec"]
        self.log_prob_loss = -1 * self.log_probs
        self.train_op, self.logprob_grad_outs, self.ent_grad_outs = self._build_train_ops(
            self.log_prob_loss, self.pl_ent_loss
        )

    def get_eval_ops(self):
        return [self.sample, self.classifier, self.log_probs]


class MessagePassingProgressiveNN(ProgressiveNN):
    """
    This class is used to get the embedding tensors of the inputs
    """

    def __init__(self, progressive_graph, config_params, simulator_params):
        ProgressiveNN.__init__(self, config_params, simulator_params)
        self.progressive_graph = progressive_graph
        self.emb_size = progressive_graph.get_embed_size()
        self.node_num = progressive_graph.nx_graph.number_of_nodes()
        self.config_params = config_params
        self.simulator_params = simulator_params
        self.input_tensor = tf.placeholder(tf.float32)
        self.bs = 1

        self.radio_messenger = RadioMessenger(
            config_params=self.config_params, embedding_size=self.emb_size, progressive_graph=progressive_graph
        )
        self.radio_messenger_output = self.radio_messenger.build()

        args = [2 * self.emb_size, 2 * self.emb_size, 2 * self.emb_size, True, False]
        with tf.variable_scope("Parent-Aggregator"):
            self.agg_p = Aggregator(*args)
            agg_p_out = self.agg_p.build(self.radio_messenger_output)
        with tf.variable_scope("Child-Aggregator"):
            self.agg_c = Aggregator(*args)
            agg_c_out = self.agg_c.build(self.radio_messenger_output)
        with tf.variable_scope("Parallel-Aggregator"):
            self.agg_r = Aggregator(*args)
            agg_r_out = self.agg_r.build(self.radio_messenger_output)
        with tf.variable_scope("Self-Embedding"):
            self.mask = tf.placeholder(tf.float32, [self.bs, None])
            self_out = tf.matmul(self.mask, self.radio_messenger_output)

        output = [agg_p_out, agg_c_out, agg_r_out, self_out]
        output = tf.reshape(output, [self.bs, -1])
        input_size = output.get_shape()[-1]
        classifier_hidden_layers = [2 * input_size, input_size]
        classifier = Classifier(input_size, classifier_hidden_layers, self.config_params["n_devs"]).build(output)
        self.no_noise_classifier = classifier
        self.build_train_ops(classifier)

        self.restore_saver = tf.train.Saver()
        self.save_saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)

    def get_feed_dict(self, progressive_graph, node):
        """
        Get the feed of nerual network
        """
        input_tensor = progressive_graph.get_embeddings()
        node_num = progressive_graph.nx_graph.number_of_nodes()

        p_masks = self.get_mask(progressive_graph.get_ancestral_mask, node_num, node)
        c_masks = self.get_mask(progressive_graph.get_progenial_mask, node_num, node)
        r_masks = self.get_mask(progressive_graph.get_peer_mask, node_num, node)
        self_masks = self.get_mask(progressive_graph.get_self_mask, node_num, node)
        feed = {
            self.input_tensor: np.array(input_tensor),
            self.agg_p.Mask: np.array(p_masks),
            self.agg_c.Mask: np.array(c_masks),
            self.agg_r.Mask: np.array(r_masks),
            self.mask: np.array(self_masks),
        }
        return feed

    def get_mask(self, mask_fns, node_num, node):
        mask = np.zeros((1, node_num), dtype=np.int32)
        mask[0, 0:node_num] = mask_fns(node)
        return mask
