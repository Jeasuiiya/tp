import logging
import functools
from typing import Callable, Optional
import jax
from jax.tree_util import tree_flatten
from jax.api_util import argnums_partial, donation_vector,flatten_fun_nokwargs
from jax._src import api
from geesibling.core.lib._graph import Device
from geesibling.adapters.jax.pipeline.devicecontext import get_global_virtual_physical_mesh
from geesibling.adapters.jax.model_parallelism import MakeScheduleContext
from geesibling.adapters.jax.pipeline.layer_construction import layer_level_transformation
from geesibling.adapters.jax.pipeline.primitive_def import mark_gradient
from geesibling.adapters.jax.pipeline.util import auto_static_argnums, abstractify_with_aval
from geesibling.tools import log
from jax import linear_util as lu
from jax._src.maps import FrozenDict
import ray
import time
from geesibling.adapters.jax.pipeline.instructions import PipelineInstType
__doc__ = """
parallelize api
Author: yiguangzheng
datetime: 2023.7.4
version: 1 2023.7.4 first commit
"""
DEVICE_MAP = {"": jax.devices("cpu")[0] if len(jax.devices("gpu")) == 0 else jax.devices("gpu")[0]}
layer_num=4
from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
def register_device():
    for i in jax.devices("cpu"):
        DEVICE_MAP[str(i)] = i
    for i in jax.devices("gpu"):
        DEVICE_MAP[str(i)] = i
    log.debug(DEVICE_MAP)


register_device()
logging.basicConfig(level=logging.WARNING)

def device_config(attrs):
    d = []
    for k, v in attrs.items():
        d.append(Device(v["type"], k, v["memory"], v["free_memory"], v["execute_time"]))
    return d

#抽象的数据转换
def _abstractify(args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    return map(jax.api_util.shaped_abstractify, flat_args), flat_args, in_tree#将其转化为含有形状和数据类型等信息的对象

def parallelize(func: Optional[Callable] = None, *, parallel_method=""):
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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):#代替被装饰的函数，先将参数以及关键字抽象化，构建上下文调度，然后按照拓扑的顺序，调度每一层中的所有块的执行，生成最终的输出
            if parallel_method.method=="PipeshardParallel":
                virtual_mesh=get_global_virtual_physical_mesh()
                f = lu.wrap_init(func)
                static_argnums = auto_static_argnums(args)
                if static_argnums:
                    dyn_argnums = [
                        i for i in range(len(args)) if i not in static_argnums
                    ]
                    # Freeze static dict to make it hashable
                    frozen_args = []
                    for i, arg in enumerate(args):
                        if i in static_argnums and isinstance(arg, dict):
                            frozen_args.append(FrozenDict(arg))
                        else:
                            frozen_args.append(arg)
                    f, dyn_args = argnums_partial(f, dyn_argnums, frozen_args)
                else:
                    dyn_args = args
                args_flat, in_tree = tree_flatten(dyn_args)
                f, out_tree = flatten_fun_nokwargs(f, in_tree)
                batch_invars = donation_vector((1,), dyn_args, kwargs)
#                abstract_args = map(abstractify_with_aval, args_flat)
#                abstract_args_micro = map(abstractify_with_aval, args_flat)
#                closed_jaxpr, out_tree= jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)
                #sentence = "123"
                #for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                #    for worker in physical_mesh.workers:
                #        ray.get(worker.get_func_to_pr.remote(sentence))

                list_instr={}
                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        if parallel_method.flag:
                            list_instr[mesh_idx]=ray.get(worker.get_data_to_split.remote(args_flat,parallel_method.num_microbatch))
                        else:
                            abstract_args = map(abstractify_with_aval, args_flat)
                            closed_jaxpr, out_tree= jax.make_jaxpr(func, return_shape=True)(*args, **kwargs) 
                            list_instr[mesh_idx]=ray.get(worker.get_stages_to_run.remote(f,parallel_method.policy, parallel_method.method, batch_invars, parallel_method.num_microbatch, args_flat,out_tree, parallel_method.layer_num, *abstract_args))

                parallel_method.flag = True
                def run_executable(worker,instructions):
                    for num,instruction in enumerate(instructions):
                        if instruction.opcode == PipelineInstType.RUN:
                            worker.run_model_parallelism.remote(num)
                        elif instruction.opcode == PipelineInstType.SEND:
                            worker.do_send_data.remote(num)
                        elif instruction.opcode == PipelineInstType.RECV:
                            worker.do_recv_data.remote(num)

                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        run_executable(worker,list_instr[mesh_idx])
                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        if mesh_idx==0:
                            result=ray.get(worker.return_result.remote())

                for mesh_idx, physical_mesh in enumerate(virtual_mesh.launched_physical_mesh_group):
                    for worker in physical_mesh.workers:
                        worker.free_buffers.remote()

            elif parallel_method.method=="ShardParallel":
                make_ctx = MakeScheduleContext(func, parallel_method.devices or (), parallel_method.policy or "fddps", parallel_method.method or "")
                make_ctx.args, make_ctx.kwargs = args, kwargs
                in_avals, flat_args, _= _abstractify(args, kwargs)

                pr, out_tree = jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)#生成jax语法树
                ctx = make_ctx([pr,parallel_method.devices])
                result = make_ctx.get_model_parallelism_result(ctx, flat_args, out_tree)

            return result

        def run_context(*args, **kwargs):#用于在不执行函数的情况下创建并返回调度上下文
            make_ctx.args, make_ctx.kwargs = args, kwargs
            in_avals, _, _ = _abstractify(args, kwargs)
            ctx = make_ctx(tuple(in_avals))
            return ctx

        wrapper.run_context = run_context
        return wrapper #添加属性，提供外界访问


    if func is None:
        return decorator
    return decorator(func)



def grad(*args, **kwargs):
    """This is the same as jax.grad, except that alpa inserts a
    gradient marker after the gradient computation.

    This function annotates all gradient tensors. This information is used to
    perform gradient accumulation transformation.
    If any auxiliary tensors are returned, they are averaged over mini batches
    in the same way as how th)e gradients are averaged.
    """
    def ret(*call_args, **call_kwargs):
        # Apply transformations (e.g., layer construction, rematerialization)
        # to the forward func
        global layer_num
        arg_list = list(args)
        arg_list[0] = layer_level_transformation(arg_list[0],layer_num)
        grad_func = api.grad(*arg_list, **kwargs)
        grads = grad_func(*call_args, **call_kwargs)
        return mark_gradient(grads)

    return ret


def value_and_grad(*args, **kwargs):
    """This is the same as jax.value_and_grad, except that alpa inserts a
    gradient marker after the gradient computation.


    This function annotates all gradient tensors. This information is used to
    perform gradient accumulation transformation.
    If any auxiliary tensors are returned, they are averaged over mini batches
    in the same way as how the gradients are averaged.
    """
    
    def ret(*call_args, **call_kwargs):
        # Apply transformations (e.g., layer construction, rematerialization)
        # to the forward func
        global layer_num
        arg_list = list(args)
        arg_list[0] = layer_level_transformation(arg_list[0],layer_num)
        grad_func = api.value_and_grad(*arg_list, **kwargs)
        val, grads = grad_func(*call_args, **call_kwargs)
        return mark_gradient((val, grads))

    return ret
