import jax
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P, Mesh
from jax.core import JaxprEqn, ClosedJaxpr, Jaxpr, new_jaxpr_eqn, Primitive
from functools import partial

devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((2, 2), devices[:4])
mesh = Mesh(device_mesh, axis_names=('x', 'y'))

dot_general_shard_primitive = Primitive('dot_general_shard')
add_shard_primitive = Primitive('add_shard')
merge_shard_primitive = Primitive('merge_shard')

def dot_general_shard_impl(*args, mesh, in_specs, out_specs,dimension_numbers):
    print(f"Args1 Device: {args[0].devices()}")
    print(f"Args2 Device: {args[1].devices()}")

    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    def parallel_dot_general(*args):
        result = jax.lax.dot_general(*args, dimension_numbers=dimension_numbers, precision=None)
        print(f"inside result shape {result.shape}")
        return result

    result = parallel_dot_general(*args)
    print(f"Result Device: {result.devices()}")

    print(f"Outside shard_map result shape: {result.shape}")

    return result

def get_dot_general_shard(dimension_numbers):
    flag = False

    dimension_numbers_str = str(dimension_numbers)
    # with open("/home/ai/ljj/tp/examples/gpt2/dimension_numbers_log.txt", "a") as f:
    #     f.write(dimension_numbers_str + "\n")
    dimension_numbers_map = {
        "(((1,), (0,)), ((), ()))": ((P('x', None), P(None, 'y')), P('x', None)),
        "(((2,), (0,)), ((), ()))": ((P(None, 'x', None), P(None, 'y')), P(None, 'x', 'y')),
        "(((3,), (3,)), ((0, 2), (0, 2)))": ((P(None, 'x', None, None), P(None, 'y', None, None)), P(None, None, 'x', 'y')),
        "(((1,), (3,)), ((0, 2), (0, 1)))": ((P(None, None, None,'y'), P(None, None, 'x',None)), P(None, None, 'x', 'y')),
    }
  
    if dimension_numbers_str in dimension_numbers_map:
        in_specs, out_specs = dimension_numbers_map[dimension_numbers_str]
    else:
        in_specs = (P(None, None), P(None, None))
        out_specs = P(None, None)
        flag = True
    
    return in_specs, out_specs, flag

dot_general_shard_primitive.def_impl(dot_general_shard_impl)

# 实现 add 的切分
def add_shard_impl(*args, mesh, in_specs, out_specs):
    print(f"Args1 Device: {args[0].devices()}")
    print(f"Args2 Device: {args[1].devices()}")
    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    def parallel_add(x, y):
        result  = jax.numpy.add(x, y)
        print(f"inside result shape {result.shape}")
        return result
    
    result = parallel_add(*args)
    print(f"Result Device: {result.devices()}")

    print(f"Outside shard_map result shape: {result.shape}")

    return result

def get_add_shard(params):
    in_specs = (P(None, 'y'), P(None, 'y'))
    out_specs = P(None, 'y')
    flag = False  # 设置为 False 表示需要进行切分
    return in_specs, out_specs, flag

add_shard_primitive.def_impl(add_shard_impl)

# 实现合并操作
def merge_shard_impl(sharded_result):
    # 这里假设 sharded_result 是一个 DeviceArray，直接放置到第一个设备
    merged = jax.device_put(sharded_result, devices[0])
    return merged

merge_shard_primitive.def_impl(merge_shard_impl)

def shard_parallel(jaxpr, params, out_tree):    
    print("jaxpr===============================")
    print(jaxpr)
    print("jaxprs===============================")

    new_eqns = []
    var_mapping = {}
    sharded_vars = set()
    
    for eqn in jaxpr.eqns:
        primitive = eqn.primitive
        if primitive.name == 'dot_general':
            # Shard dot_general
            dimension_numbers = eqn.params['dimension_numbers']
            in_specs, out_specs, flag = get_dot_general_shard(dimension_numbers)
            if not flag:
                new_eqn = new_jaxpr_eqn(
                    invars=eqn.invars,
                    outvars=eqn.outvars,
                    primitive=dot_general_shard_primitive,
                    params={
                        'mesh': mesh,
                        'in_specs': in_specs,
                        'out_specs': out_specs,
                        'dimension_numbers': dimension_numbers
                    },
                    effects=set()
                )
                new_eqns.append(new_eqn)
                sharded_vars.add(eqn.outvars[0])
            else:
                new_eqns.append(eqn)
        elif primitive.name == 'add':
            # Check if any input is sharded
            input_vars = eqn.invars
            sharded_inputs = [var for var in input_vars if var in sharded_vars]
            if sharded_inputs:
                # Only need to shard unsharded inputs
                unsharded_inputs = [var for var in input_vars if var not in sharded_vars]
                if len(unsharded_inputs) == 1:
                    # Shard the unsharded input to match sharded input
                    # to_shard_var = unsharded_inputs[0]
                    # Replace with sharded add
                    in_specs, out_specs, flag = get_add_shard(dimension_numbers)
                    new_eqn = new_jaxpr_eqn(
                        invars=input_vars,
                        outvars=eqn.outvars,
                        primitive=add_shard_primitive,
                        params={
                            'mesh': mesh,
                            'in_specs': in_specs,  # in_specs, out_specs
                            'out_specs': out_specs
                        },
                        effects=set()
                    )
                    new_eqns.append(new_eqn)
                    sharded_vars.add(eqn.outvars[0])
                    # Insert merge after add
                    # merge_eqn = new_jaxpr_eqn(
                    #     invars=(eqn.outvars[0],),
                    #     outvars=(f'merged_{eqn.outvars[0]}',),
                    #     primitive=merge_shard_primitive,
                    #     params={},
                    #     effects=set()
                    # )
                    # new_eqns.append(merge_eqn)
                    # var_mapping[eqn.outvars[0]] = f'merged_{eqn.outvars[0]}'
                else:
                    # Handle multiple unsharded inputs if necessary
                    new_eqns.append(eqn)
            else:
                # No sharding needed, keep original add
                new_eqns.append(eqn)
        else:
            # Other primitives, keep as is
            new_eqns.append(eqn)

    new_jaxpr_core = Jaxpr(jaxpr.jaxpr.constvars, jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars, new_eqns)
    new_jaxpr = ClosedJaxpr(new_jaxpr_core, jaxpr.consts)

    print("new_jaxpr===============================")
    print(new_jaxpr)
    print("new_jaxprs===============================")
    
    result = jax.core.eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.consts, *params)

    # for arg in result:
    #     print(f"Result Device: {arg.devices()}")
    result = jax.device_put(result, devices[0])
    # result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

    return result