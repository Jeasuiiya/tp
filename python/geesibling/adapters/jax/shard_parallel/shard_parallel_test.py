# from jax.experimental.shard_map import shard_map
# from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
# from jax.experimental import mesh_utils
# import jax
from collections import defaultdict
import networkx as nx

def shard_parallel():
    # 构建 nodes 字典
    nodes = {'ci': Node(node_type='input', dtype='float32', shape=(64, 784), predicted_shape=None, sharding_shape=None), 'a': Node(node_type='input', dtype='float64', shape=(784, 1024), predicted_shape=None, sharding_shape=None), 'c': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None, sharding_shape=None), 'e': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None, sharding_shape=None), 'g': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None, sharding_shape=None), 'i': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None, sharding_shape=None), 'k': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None, sharding_shape=None), 'm': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None, sharding_shape=None), 'o': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None, sharding_shape=None), 'q': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None, sharding_shape=None), 'zb': Node(node_type='output', dtype='float64', shape=(64, 784), predicted_shape=None, sharding_shape=None), 'cv': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'cw': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None, sharding_shape=None), 'cx': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yz': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'za': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'da': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'db': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None, sharding_shape=None), 'dc': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yx': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yy': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'df': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'dg': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None, sharding_shape=None), 'dh': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yv': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yw': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'dk': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'dl': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None, sharding_shape=None), 'dm': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yt': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'yu': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None, sharding_shape=None), 'dp': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None, sharding_shape=None)}
    # 构建 edge_operations 字典
    edge_operations = {('ci', 'zb'): {'operation': 'convert_element_type', 'params': {}}, ('zb', 'cv'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('a', 'cv'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('c', 'cw'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('cv', 'cx'): {'operation': 'add', 'params': {}}, ('cw', 'cx'): {'operation': 'add', 'params': {}}, ('cx', 'yz'): {'operation': 'custom_jvp_call', 'params': {}}, ('cx', 'za'): {'operation': 'gt', 'params': {}}, ('yz', 'da'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('e', 'da'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('g', 'db'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('da', 'dc'): {'operation': 'add', 'params': {}}, ('db', 'dc'): {'operation': 'add', 'params': {}}, ('dc', 'yx'): {'operation': 'custom_jvp_call', 'params': {}}, ('dc', 'yy'): {'operation': 'gt', 'params': {}}, ('yx', 'df'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('i', 'df'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('k', 'dg'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('df', 'dh'): {'operation': 'add', 'params': {}}, ('dg', 'dh'): {'operation': 'add', 'params': {}}, ('dh', 'yv'): {'operation': 'custom_jvp_call', 'params': {}}, ('dh', 'yw'): {'operation': 'gt', 'params': {}}, ('yv', 'dk'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('m', 'dk'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('o', 'dl'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('dk', 'dm'): {'operation': 'add', 'params': {}}, ('dl', 'dm'): {'operation': 'add', 'params': {}}, ('dm', 'yt'): {'operation': 'custom_jvp_call', 'params': {}}, ('dm', 'yu'): {'operation': 'gt', 'params': {}}, ('yt', 'dp'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('q', 'dp'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}}

    G = build_graph(nodes, edge_operations)
    
    # define_sharding_specs_single_dim

    infer_all_shapes(G)

    for node_name, node in nodes.items():
        if node.shape == node.predicted_shape:
            print(f"Node {node_name}: shape={node.shape}, predicted_shape={node.predicted_shape}")
        else: 
            print(f"Node {node_name} predict wrong!!!!!!!!")

class Node:
    def __init__(self, node_type, dtype, shape, predicted_shape=None, sharding_shape=None):
        self.node_type = node_type
        self.dtype = dtype
        self.shape = shape
        self.predicted_shape = predicted_shape
        self.sharding_shape = sharding_shape

    def __repr__(self):
        return (f"Node(node_type='{self.node_type}', dtype='{self.dtype}', "
                f"shape={self.shape}, predicted_shape={self.predicted_shape}, "
                f"sharding_shape={self.sharding_shape})")

def build_graph(nodes, edge_operations):
    G = nx.DiGraph()
    for node_name, node in nodes.items():
        G.add_node(node_name, node=node)
    
    for (src, dst), attrs in edge_operations.items():
        G.add_edge(src, dst, operation=attrs['operation'], params=attrs.get('params', {}))

    return G

def find_longest_paths(G, num_paths=3):
    all_paths = list(nx.all_simple_paths(G, source=list(G.nodes())[0], target=list(G.nodes())[-1]))
    path_lengths = [(path, len(path)) for path in all_paths]
    path_lengths.sort(key=lambda x: -x[1])
    longest_paths = [path for path, _ in path_lengths[:num_paths]]
    return longest_paths

def define_sharding_specs_from_graph(G, mesh_axis_names, num_paths=3):
    # 找到三条最长路径
    longest_paths = find_longest_paths(G, num_paths)
    print(f"找到的最长路径: {longest_paths}")

    # 定义切分策略
    sharding_dict = {}
    mesh_axis = mesh_axis_names[0] if mesh_axis_names else None

    # 遍历三条最长路径
    for path in longest_paths:
        try:
            # 对路径的起始节点进行切分
            main_var = path[0]
            print(f"尝试对路径 {path} 进行切分，选择变量 '{main_var}' 为切分基准")

            # 对路径的起始节点进行第一个维度切分
            sharding_dict[main_var] = P(mesh_axis, *([None] * (len(G.nodes[main_var]['shape']) - 1)))

            # 按拓扑顺序传播切分策略
            for node in nx.topological_sort(G):
                if node in sharding_dict:
                    continue  # 已处理的节点

                preds = list(G.predecessors(node))
                if not preds:
                    # 没有前驱的节点（输入变量），默认复制
                    sharding_dict[node] = P()
                    continue

                # 动态推导切分策略
                input_shardings = [sharding_dict.get(pred, P()) for pred in preds]
                op = G.edges[preds[0], node]['operation']

                if op == 'dot_general':
                    # 矩阵乘法：如果 lhs 被切分，则输出沿第一个维度切分
                    lhs, rhs = preds[:2]
                    if sharding_dict.get(lhs, P())[0] == mesh_axis:
                        sharding_dict[node] = P(mesh_axis, *([None] * (len(G.nodes[node]['shape']) - 1)))
                    else:
                        sharding_dict[node] = P()
                elif op == 'broadcast_in_dim':
                    # 广播：默认不切分
                    sharding_dict[node] = P()
                else:
                    # 其他操作，默认复制
                    sharding_dict[node] = P()

            # 如果成功，跳出路径循环
            print(f"路径 {path} 切分成功")
            break  # 成功后退出路径遍历
        except Exception as e:
            print(f"路径 {path} 切分失败: {e}")
            # 尝试下一条路径
            continue
    else:
        print("所有路径切分失败。")
        return None  # 所有路径切分失败

    return sharding_dict

def infer_shape(operation, params, input_shapes):
    if operation in ['add', 'sub', 'mul', 'div', 'max', 'min', 'eq', 'ne', 'gt', 'and']:
        # 这些操作通常是逐元素操作，输出形状与输入形状相同（假设广播规则满足）
        # 这里简化处理，实际情况需要实现广播规则(待补充)
        return input_shapes[0]
    elif operation == 'convert_element_type':
        # 类型转换操作，输出形状与输入形状相同
        return input_shapes[0]
    elif operation == 'custom_jvp_call':
        # 类型转换操作，输出形状与输入形状相同
        return input_shapes[0]
    elif operation == 'integer_pow':
        # 整数幂操作，输出形状与输入形状相同
        return input_shapes[0]
    elif operation in ['sqrt', 'rsqrt', 'exp', 'tanh']:
        # 数学函数，输出形状与输入形状相同
        return input_shapes[0]
    elif operation == 'dot_general':
        dimension_numbers=params['dimension_numbers']
        # (lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims) = dimension_numbers

        lhs_shape = input_shapes[0]
        rhs_shape = input_shapes[1]
        
        # for lhs_cdim, rhs_cdim in zip(lhs_contracting_dims, rhs_contracting_dims):
        #     if lhs_shape[lhs_cdim] != rhs_shape[rhs_cdim]:
        #         raise ValueError(
        #             f"收缩维度不匹配: lhs dim {lhs_cdim} size {lhs_shape[lhs_cdim]} != rhs dim {rhs_cdim} size {rhs_shape[rhs_cdim]}"
        #         )

        # for lhs_bdim, rhs_bdim in zip(lhs_batch_dims, rhs_batch_dims):
        #     if lhs_shape[lhs_bdim] != rhs_shape[rhs_bdim]:
        #         raise ValueError(
        #             f"批处理维度不匹配: lhs dim {lhs_bdim} size {lhs_shape[lhs_bdim]} != rhs dim {rhs_bdim} size {rhs_shape[rhs_bdim]}"
        #         )

        dimension_output_map = {
            # 2D × 2D
            (((1,), (0,)), ((), ())): lambda lhs, rhs: (lhs[0], rhs[1]),
            (((0,), (0,)), ((), ())): lambda lhs, rhs: (lhs[1], rhs[1]),
            (((1,), (1,)), ((), ())): lambda lhs, rhs: (lhs[0], rhs[0]),
            # 3D × 2D
            (((2,), (0,)), ((), ())): lambda lhs, rhs: (lhs[0], lhs[1], rhs[1]),
            (((2,), (1,)), ((), ())): lambda lhs, rhs: (lhs[0], lhs[1], rhs[0]),
            # 4D × 4D
            (((3,), (3,)), ((0, 2), (0, 2))): lambda lhs, rhs: (lhs[0], lhs[1], lhs[2], rhs[1]),
            (((1,), (3,)), ((0, 2), (0, 1))): lambda lhs, rhs: (lhs[0], lhs[2], lhs[1], rhs[3]),
            (((2,), (3,)), ((0, 1), (0, 2))): lambda lhs, rhs: (lhs[0], lhs[1], lhs[3], rhs[2]),
            (((3,), (1,)), ((0, 1), (0, 2))): lambda lhs, rhs: (lhs[0], lhs[1], lhs[2], rhs[2]),
            # 特殊的3D × 3D → 2D
            (((0,1), (0,1)), ((), ())): lambda lhs, rhs: (lhs[2], rhs[2]),
        }
        
        output_shape = dimension_output_map[dimension_numbers](lhs_shape, rhs_shape)
        return output_shape
    elif operation == 'reshape':
        return params['new_sizes']
    elif operation == 'broadcast_in_dim':
        return params['shape']
    elif operation in ['reduce_sum', 'reduce_max']:
        # 归约操作，根据 'axes' 参数移除相应的维度
        axes = params['axes']
        input_shape = input_shapes[0]
        output_shape = tuple(dim for idx, dim in enumerate(input_shape) if idx not in axes)
        return output_shape
    elif operation == 'select_n': 
        # 条件选择操作，输出形状与选择的分支形状相同
        # 假设所有分支的形状相同
        # 输入_shapes[0]: condition
        # 输入_shapes[1]: true_branch
        # 输入_shapes[2]: false_branch
        return input_shapes[1]
    elif operation == 'gather':
        return params['slice_sizes']
    elif operation == 'slice': 
        # 切片操作，根据 start_indices, limit_indices, strides 计算输出形状
        start_indices = params['start_indices']
        limit_indices = params['limit_indices']
        strides = params['strides'] or (1,) * len(start_indices)  # 默认为1
        input_shape = input_shapes[0]
        output_shape = tuple(
            (limit - start + stride - 1) // stride if stride != 0 else 1
            for start, limit, stride in zip(start_indices, limit_indices, strides)
        )
        return output_shape
    elif operation == 'concatenate':
        # 连接操作，根据 'dimension' 参数和所有输入的形状计算输出形状
        concat_dim = params['dimension']
        input_shapes_rest = [shape for shape in input_shapes]
        total_concat_dim = sum(shape[concat_dim] for shape in input_shapes_rest)
        # 其他维度保持一致
        output_shape = list(input_shapes_rest[0])
        output_shape[concat_dim] = total_concat_dim
        return tuple(output_shape)
    elif operation == 'transpose':
        # 转置操作，根据 'permutation' 参数重新排列维度
        permutation = params['permutation']
        input_shape = input_shapes[0]
        output_shape = tuple(input_shape[i] for i in permutation)
        return output_shape
    elif operation == 'stop_gradient':
        # 停止梯度传播，形状与输入相同
        return input_shapes[0]
    elif operation == 'iota':
        # 生成序列操作，输出形状由 'shape' 参数决定
        return params['shape']
    else:
        raise NotImplementedError(f"未实现操作类型 '{operation}' 的形状推断。")

def infer_all_shapes(G):
    for node_name, node_data in G.nodes(data=True):
        node = node_data['node']
        if node.node_type == 'input':
            node.predicted_shape = node.shape
            print(f"input {node_name} shape: {node.predicted_shape}")
    
    for node in nx.topological_sort(G):
        node_data = G.nodes[node]['node']
        if node_data.node_type == 'input':
            continue

        preds = list(G.predecessors(node))
        operation = G.edges[preds[0], node]['operation']
        params = G.edges[preds[0], node]['params']
        input_shapes = [G.nodes[p]['node'].predicted_shape for p in preds]
        
        try:
            node_data.predicted_shape = infer_shape(operation, params, input_shapes)
        except (NotImplementedError, ValueError) as e:
            print(f"EEEEEEEEEEEENode {node} failed: {e}")
            node_data.predicted_shape = None

shard_parallel()