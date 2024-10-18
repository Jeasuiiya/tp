import jax
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P, Mesh
from jax.core import JaxprEqn, ClosedJaxpr, Jaxpr, new_jaxpr_eqn, Primitive
from functools import partial
from jax import debug

dot_general_shard_primitive = Primitive('dot_general_shard')

def dot_general_shard_impl(*args, mesh, in_specs, out_specs):
    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    def parallel_fn(*args):
        result = jax.lax.dot_general(*args, dimension_numbers=(((1,), (0,)), ((), ())), precision=None)
        return result
    return parallel_fn(*args)

dot_general_shard_primitive.def_impl(dot_general_shard_impl)

def shard_parallel(jaxpr, params, out_tree):    
    # print("jaxpr===============================")
    # print(jaxpr)
    # print("jaxprs===============================")
  
    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh((2, 2), devices[:4])
    mesh = Mesh(device_mesh, axis_names=('x', 'y'))

    new_eqns = []

    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'dot_general':
            if eqn.params['dimension_numbers'] == (((1,), (0,)), ((), ())):
                new_eqn = new_jaxpr_eqn(
                    invars=eqn.invars,
                    outvars=eqn.outvars,
                    primitive=dot_general_shard_primitive,
                    params={
                        'mesh': mesh,
                        'in_specs': (P('x', None), P(None, 'y')),
                        'out_specs': P('x', 'y')
                    },
                    effects=set()
                )
                new_eqns.append(new_eqn)
            else:
                new_eqns.append(eqn)
        else:
            new_eqns.append(eqn)

    new_jaxpr_core = Jaxpr(jaxpr.jaxpr.constvars, jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars, new_eqns)
    new_jaxpr = ClosedJaxpr(new_jaxpr_core, jaxpr.consts)
    # print("new_jaxpr===============================")
    # print(new_jaxpr)
    # print("new_jaxprs===============================")
    
    result = jax.core.eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.consts, *params)

    result = jax.device_put(result, devices[0])
    # result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

    return result
