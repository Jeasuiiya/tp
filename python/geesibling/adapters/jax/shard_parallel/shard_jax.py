import jax
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P, Mesh
from jax.core import JaxprEqn, ClosedJaxpr, Jaxpr, new_jaxpr_eqn, Primitive
from functools import partial
import sys

class Logger:
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def __del__(self):
        self.log.close()


# sys.stdout = Logger("/home/ai/ljj/tp/examples/gpt2/output_log.txt")

devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((2, 2), devices[:4])
mesh = Mesh(device_mesh, axis_names=('x', 'y'))

dot_general_shard_primitive = Primitive('dot_general_shard')

def dot_general_shard_impl(*args, mesh, in_specs, out_specs,dimension_numbers):
    # print(f"Input arguments shapes: {[arg.shape for arg in args]}")
    # print(f"Dimension numbers: {dimension_numbers}")
    # actually_output = jax.lax.dot_general(*args, dimension_numbers, precision=None)

    # print(f"Args1 Device: {args[0].devices()}")
    # print(f"Args2 Device: {args[1].devices()}")

    # @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    def parallel_fn(*args):
        result = jax.lax.dot_general(*args, dimension_numbers=dimension_numbers, precision=None)
        # print(f"inside result shape {result.shape}")
        return result

    result = parallel_fn(*args)
    # print(f"Result Device: {result.devices()}")

    # print(f"Outside shard_map result shape: {result.shape}")
    # print(f"Actual result shape: {actually_output.shape}")

    return result

dot_general_shard_primitive.def_impl(dot_general_shard_impl)

def shard_parallel(jaxpr, params, out_tree):    
    print("jaxpr===============================")
    print(jaxpr)
    print("jaxprs===============================")

    new_eqns = []

    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'dot_general':
            in_specs, out_specs, flag = get_dot_general_shard(eqn.params['dimension_numbers'])
            if flag:
                new_eqns.append(eqn)
                continue
            new_eqn = new_jaxpr_eqn(
                invars=eqn.invars,
                outvars=eqn.outvars,
                primitive=dot_general_shard_primitive,
                params={
                    'mesh': mesh,
                    'in_specs': in_specs,
                    'out_specs': out_specs,
                    'dimension_numbers': eqn.params['dimension_numbers']
                },
                effects=set()
            )
            new_eqns.append(new_eqn)
        else:
            new_eqns.append(eqn)

    new_jaxpr_core = Jaxpr(jaxpr.jaxpr.constvars, jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars, new_eqns)
    new_jaxpr = ClosedJaxpr(new_jaxpr_core, jaxpr.consts)

    # print("new_jaxpr===============================")
    # print(new_jaxpr)
    # print("new_jaxprs===============================")
    
    result = jax.core.eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.consts, *params)

    # for arg in result:
    #     print(f"Result Device: {arg.devices()}")
    result = jax.device_put(result, devices[0])
    # result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

    return result

def get_dot_general_shard(dimension_numbers):
    flag = False

    dimension_numbers_str = str(dimension_numbers)
    # with open("/home/ai/ljj/tp/examples/gpt2/dimension_numbers_log.txt", "a") as f:
    #     f.write(dimension_numbers_str + "\n")
    dimension_numbers_map = {
        "(((1,), (0,)), ((), ()))": ((P('x', None), P(None, 'y')), P('x', 'y')),
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