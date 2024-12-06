from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from jax.experimental import mesh_utils
import jax
from collections import defaultdict
import networkx as nx

devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((4, ), devices[:4])
mesh = Mesh(device_mesh, axis_names=('x',))


def shard_parallel(jaxpr,args,out_tree):
    print("jaxpr===============================")
    print(jaxpr)
    print("jaxprs===============================")

    nodes, edge_operations = jaxpr2graph(jaxpr.jaxpr)
    print("Nodes:", nodes)
    print("Edge Operations:", edge_operations)

    G = build_graph(nodes, edge_operations)




    # in_specs = []
    # out_specs = []
    # for var in jaxpr.jaxpr.invars:
    #     if str(var) == start_node:
    #         # 对 start_node 对应的输入进行切分，仅切分第一个维度
    #         partition_spec = P('x', *([None] * (len(var.aval.shape) - 1)))
    #     else:
    #         # 对其他输入进行全复制
    #         partition_spec = P(*([None] * len(var.aval.shape)))
    #     in_specs.append(partition_spec)

    # print(f"输入分片规范: {in_specs}")

    # for var in jaxpr.jaxpr.outvars:
    #     # 根据输出变量的形状动态生成分片规范
    #     if hasattr(var.aval, 'shape'):
    #         partition_spec = P('x', *([None] * (len(var.aval.shape) - 1)))
    #     else:
    #         partition_spec = P()  # 如果没有形状信息，保持全复制
    #     out_specs.append(partition_spec)

    # print(f"输出分片规范: {out_specs}")

    # # 定义评估函数
    # def eval_jaxpr(*args):
    #     sharded_results= jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
    #     for i in jax.tree_leaves(sharded_results):
    #         print(f"sharded_results shape {i.shape}")
    #     return sharded_results

    # # 使用 shard_map 进行并行计算
    # sharded_eval_jaxpr = shard_map(
    #     eval_jaxpr,
    #     mesh,
    #     in_specs=tuple(in_specs),
    #     out_specs=out_specs
    # )

    # # JIT 编译
    # jitted_sharded_eval_jaxpr = jax.jit(sharded_eval_jaxpr)

    # # 执行评估
    # sharded_results = jitted_sharded_eval_jaxpr(*args)

    # merged_results = sharded_results

    # for i in jax.tree_leaves(merged_results):
    #     print(f"merged_results shape {i.shape}")
    #     print(i)

    # return merged_results

class Node:
    def __init__(self, node_type, dtype, shape, predicted_shape=None):
        self.node_type = node_type
        self.dtype = dtype
        self.shape = shape
        self.predicted_shape = predicted_shape

    def __repr__(self):
        return (f"Node(node_type='{self.node_type}', dtype='{self.dtype}', "
                f"shape={self.shape}, predicted_shape={self.predicted_shape})")

def jaxpr2graph(jaxpr):
    nodes = {}
    edge_operations = {}
    NO_PARAMS_OPERATIONS = {'custom_jvp_call', 'convert_element_type'}
    
    INPUT_TYPE = "input"
    INTERMEDIATE_TYPE = "intermediate"
    OUTPUT_TYPE = "output"

    for var in jaxpr.invars:
        var_name = str(var)
        node = Node(
            node_type=INPUT_TYPE,
            dtype=var.aval.dtype,
            shape=var.aval.shape,
        )
        nodes[var_name] = node
    
    for eqn in jaxpr.eqns:
        inputs = eqn.invars
        outputs = eqn.outvars
        operation = eqn.primitive.name
        params = eqn.params if operation not in NO_PARAMS_OPERATIONS else {}
        
        for output in outputs:
            output_name = str(output)
            if output_name == "_":
                continue
            node = Node(
                node_type=INTERMEDIATE_TYPE,
                dtype=output.aval.dtype,
                shape=output.aval.shape
            )
            nodes[output_name] = node
        
            for input_var in inputs:
                input_name = str(input_var)
                if input_name not in nodes:
                    continue 
                edge_operations[(input_name, output_name)] = {
                    'operation': operation,
                    'params': params
                }
    
    for var in jaxpr.outvars:
        var_name = str(var)
        nodes[var_name].node_type = OUTPUT_TYPE
    return nodes, edge_operations

def build_graph(nodes, edge_operations):
    G = nx.DiGraph()
    for node_name, node in nodes.items():
        G.add_node(node_name, node=node)
    
    for (src, dst), attrs in edge_operations.items():
        G.add_edge(src, dst, **attrs)  #待修改

    return G