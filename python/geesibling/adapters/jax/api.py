import functools
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from geesibling.adapters.jax.jaxpr2graph import jaxpr2graph
from geesibling.adapters.jax.block2jaxpr import block2jaxpr
from geesibling.core.lib._graph import divide_graph, search_policy, Device
from geesibling.adapters.jax.schedule import ScheduleContext
from geesibling.tools import log

__doc__ = """
parallelize api
Author: yiguangzheng
datetime: 2023.7.4
version: 1 2023.7.4 first commit
"""


DEVICE_MAP = {"": jax.devices("cpu")[0] if len(jax.devices("gpu")) == 0 else jax.devices("gpu")[0]}


def register_device():
    for i in jax.devices("cpu"):
        DEVICE_MAP[str(i)] = i
    for i in jax.devices("gpu"):
        DEVICE_MAP[str(i)] = i
    log.debug(DEVICE_MAP)


register_device()


def device_config(attrs):
    d = []
    for k, v in attrs.items():
        d.append(Device(v["type"], k, v["memory"], v["free_memory"], v["execute_time"]))
    return d


def _abstractify(args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    return map(jax.api_util.shaped_abstractify, flat_args), flat_args, in_tree


class MakeScheduleContext:
    """
    schedule context.
    used for saving arguments for parallelizing function
    """

    def __init__(self, func, devices=(), policy="fddps") -> None:
        self.func = func
        self.devices = devices
        self.policy = policy
        self.args = None
        self.kwargs = None

    def a(self, args, kwargs):
        self.args, self.kwargs = args, kwargs

    @functools.lru_cache()
    def __call__(self, in_avals):
        pr, out_tree = jax.make_jaxpr(self.func, return_shape=True)(*self.args, **self.kwargs)
        gw = jaxpr2graph(pr)
        log.debug("jaxpr2graph finished")
        g = gw.graph
        # call strategy search

        device_map = search_policy(g, self.devices, self.policy)

        log.debug("search policy finished. placement: %s", device_map)

        if device_map is not None:
            for k, v in device_map.items():
                g.get_node(k).device = v
        else:
            log.warning("search policy failed.")
        sub_graphs = divide_graph(g)

        # prepare context
        @functools.lru_cache()
        def cache_executable(ctx, block):
            pr, const_names = block2jaxpr(ctx, block, gw.params)

            const = list(map(lambda x: gw.node_ref_const[x], const_names))
            return jax.jit(functools.partial(jax.core.eval_jaxpr, pr, const), device=DEVICE_MAP[block.device])

        ctx = ScheduleContext(gw.invars, gw.returns, gw.node_output_type, out_tree, cache_executable)
        ctx.blocks(sub_graphs)
        ctx.regular_blocks()
        # ctx.topo_order = tuple(filter(lambda b: b.outputports_size != 0, ctx.order()))
        ctx.topo_order = tuple(ctx.order())
        log.debug(
            "scheduled: %s, all blocks: %s ",
            functools.reduce(lambda a, b: a + len(b), ctx.topo_order, 0),
            len(ctx.graph2block),
        )
        return ctx


def parallelize(func: Optional[Callable] = None, *, devices=None, policy="fddps"):
    """
    parallelize a function

    Example:
    ```python
    @parallelize
    def compute(x, y):
        out = x + y
        out = x * out
        return out
    ```
    """

    def decorator(func):
        make_ctx = MakeScheduleContext(func, devices or (), policy or "fddps")

        def exec_block(ctx, flat_args, block):
            bargs = []
            for i in block.inputports:
                if i.source == 0:  # block id is 0, global input
                    bargs.append(flat_args[i.source_index])
                else:
                    a = ctx.block_outputs[i.source][i.source_index]
                    if not isinstance(a, np.ndarray):
                        if a.device() is not DEVICE_MAP[block.device]:
                            a = jax.device_put(a, DEVICE_MAP[block.device]).block_until_ready()
                    bargs.append(a)
            with jax.default_device(DEVICE_MAP[block.device]):
                return jax.block_until_ready(ctx.cache_block_executable(ctx, block)(*bargs))

        def schedule_level(ctx, level, flat_args):
            # todo(huangchengchuang): all block in level could be parallelized
            with ThreadPoolExecutor(max_workers=128) as executor:
                future_to_results = {executor.submit(exec_block, ctx, flat_args, block): block for block in level}
                for future in as_completed(future_to_results):
                    block = future_to_results[future]
                    ctx.block_outputs[block.id] = future.result()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            make_ctx.args, make_ctx.kwargs = args, kwargs
            in_avals, flat_args, _ = _abstractify(args, kwargs)
            ctx = make_ctx(tuple(in_avals))
            for level in ctx.topo_order:
                schedule_level(ctx, level, flat_args)

            def returns(r):
                block_ref = ctx.nodeoutput_blockoutput[r]
                return ctx.block_outputs[block_ref.block][block_ref.index]

            _, out_tree = tree_flatten(ctx.out_tree)
            return tree_unflatten(out_tree, map(returns, ctx.returns))

        def run_context(*args, **kwargs):
            make_ctx.args, make_ctx.kwargs = args, kwargs
            in_avals, _, _ = _abstractify(args, kwargs)
            ctx = make_ctx(tuple(in_avals))
            return ctx

        wrapper.run_context = run_context
        return wrapper

    if func is None:
        return decorator
    return decorator(func)
