from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
import functools
from typing import Callable, Optional
import jax
from jax.tree_util import tree_flatten
from jax._src import api
from jax.api_util import argnums_partial, donation_vector,flatten_fun_nokwargs
from jax._src.maps import FrozenDict
from jax import linear_util as lu
from geesibling.core.lib._graph import Device
from concurrent.futures import ThreadPoolExecutor, as_completed

from geesibling.adapters.jax.pipeline.primitive_def import mark_gradient
from geesibling.adapters.jax.model_parallelism import MakeScheduleContext
from geesibling.adapters.jax.pipeline.layer_construction import layer_level_transformation
from geesibling.adapters.jax.pipeline.stage_construction import compile_executable
from geesibling.adapters.jax.pipeline.util import auto_static_argnums, abstractify_with_aval
from geesibling.tools import log


def shard_parallel(pr,args,out_tree):
    devices = jax.devices()
    mesh = Mesh(devices, ['x'])
    def sharded_eval_jaxpr(*args):
        def eval_jaxpr(*args):
            return jax.core.eval_jaxpr(pr.jaxpr, pr.consts, *args)

        return shard_map(eval_jaxpr, mesh, in_specs=P(), out_specs=P(),check_rep=False)(*args)
    
    jitted_sharded_eval_jaxpr = jax.jit(sharded_eval_jaxpr)
    def evaluate_jaxpr_with_args(*args):
        return jitted_sharded_eval_jaxpr(*args)
    
    with ThreadPoolExecutor(max_workers=128) as executor:
        future = executor.submit(evaluate_jaxpr_with_args, *args)

    return future.result()
