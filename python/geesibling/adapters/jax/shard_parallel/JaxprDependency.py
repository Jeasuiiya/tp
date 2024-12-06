import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pydot

# 定义 Node 类
class Node:
    def __init__(self, node_type, dtype, shape, predicted_shape=None):
        self.node_type = node_type
        self.dtype = dtype
        self.shape = shape
        self.predicted_shape = predicted_shape

    def __repr__(self):
        return (f"Node(node_type='{self.node_type}', dtype='{self.dtype}', "
                f"shape={self.shape}, predicted_shape={self.predicted_shape})")

# 构建 nodes 字典
nodes = {'ci': Node(node_type='input', dtype='float32', shape=(64, 784), predicted_shape=None), 'a': Node(node_type='input', dtype='float64', shape=(784, 1024), predicted_shape=None), 'c': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None), 'e': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None), 'g': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None), 'i': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None), 'k': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None), 'm': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None), 'o': Node(node_type='input', dtype='float64', shape=(1024,), predicted_shape=None), 'q': Node(node_type='input', dtype='float64', shape=(1024, 1024), predicted_shape=None), 'zb': Node(node_type='output', dtype='float64', shape=(64, 784), predicted_shape=None), 'cv': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'cw': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None), 'cx': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yz': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None), 'za': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None), 'da': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'db': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None), 'dc': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yx': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yy': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None), 'df': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'dg': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None), 'dh': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yv': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yw': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None), 'dk': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'dl': Node(node_type='intermediate', dtype='float64', shape=(1, 1024), predicted_shape=None), 'dm': Node(node_type='intermediate', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yt': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None), 'yu': Node(node_type='output', dtype='bool', shape=(64, 1024), predicted_shape=None), 'dp': Node(node_type='output', dtype='float64', shape=(64, 1024), predicted_shape=None)}
# 构建 edge_operations 字典
edge_operations = {('ci', 'zb'): {'operation': 'convert_element_type', 'params': {}}, ('zb', 'cv'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('a', 'cv'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('c', 'cw'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('cv', 'cx'): {'operation': 'add', 'params': {}}, ('cw', 'cx'): {'operation': 'add', 'params': {}}, ('cx', 'yz'): {'operation': 'custom_jvp_call', 'params': {}}, ('cx', 'za'): {'operation': 'gt', 'params': {}}, ('yz', 'da'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('e', 'da'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('g', 'db'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('da', 'dc'): {'operation': 'add', 'params': {}}, ('db', 'dc'): {'operation': 'add', 'params': {}}, ('dc', 'yx'): {'operation': 'custom_jvp_call', 'params': {}}, ('dc', 'yy'): {'operation': 'gt', 'params': {}}, ('yx', 'df'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('i', 'df'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('k', 'dg'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('df', 'dh'): {'operation': 'add', 'params': {}}, ('dg', 'dh'): {'operation': 'add', 'params': {}}, ('dh', 'yv'): {'operation': 'custom_jvp_call', 'params': {}}, ('dh', 'yw'): {'operation': 'gt', 'params': {}}, ('yv', 'dk'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('m', 'dk'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('o', 'dl'): {'operation': 'broadcast_in_dim', 'params': {'shape': (1, 1024), 'broadcast_dimensions': (1,)}}, ('dk', 'dm'): {'operation': 'add', 'params': {}}, ('dl', 'dm'): {'operation': 'add', 'params': {}}, ('dm', 'yt'): {'operation': 'custom_jvp_call', 'params': {}}, ('dm', 'yu'): {'operation': 'gt', 'params': {}}, ('yt', 'dp'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}, ('q', 'dp'): {'operation': 'dot_general', 'params': {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': None}}}
# 初始化有向图并添加节点和边
G_detailed = nx.DiGraph()

# 添加节点
for node_id, node in nodes.items():
    G_detailed.add_node(node_id, 
                        node_type=node.node_type, 
                        dtype=node.dtype, 
                        shape=node.shape)

# 添加边
for (src, dst), attrs in edge_operations.items():
    print(attrs['operation'])
    G_detailed.add_edge(src, dst, operation=attrs['operation'])

# 根据 node_type 定义不同的形状
shape_map = {
    'input': 'o',          # 圆形
    'output': 's',         # 方形
    'intermediate': 'D'    # 菱形
}


# 定义 visualize_graph 方法
def visualize_graph(G, shape_map):
    # 为绘图准备节点形状和标签
    node_shapes = {node_id: shape_map.get(attrs['node_type'], 'o') 
                   for node_id, attrs in G.nodes(data=True)}
    
    # 准备节点标签，包含 dtype 和 shape
    node_labels = {node_id: f"{node_id}\n{attrs['dtype']}\n{attrs['shape']}" 
                   for node_id, attrs in G.nodes(data=True)}

    # 设置自上而下的层次结构布局
    pos_hierarchy = nx.nx_pydot.pydot_layout(G, prog="dot")

    # 绘制图形
    plt.figure(figsize=(15, 30))

    # 获取所有不同的形状
    unique_shapes = set(node_shapes.values())

    for shape in unique_shapes:
        # 获取具有当前形状的节点
        shaped_nodes = [node for node, shp in node_shapes.items() if shp == shape]
        nx.draw_networkx_nodes(
            G, pos=pos_hierarchy, nodelist=shaped_nodes, 
            node_shape=shape, node_size=1000,
            node_color='lightblue' if shape == 'o' else ('lightgreen' if shape == 's' else 'lightcoral'),
            edgecolors='black'
        )

    # 绘制边
    nx.draw_networkx_edges(G, pos=pos_hierarchy, arrows=True, arrowstyle='->', arrowsize=20)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos=pos_hierarchy, labels=node_labels, font_size=6, font_weight="bold")

    # 添加边标签表示操作
    edge_labels = nx.get_edge_attributes(G, "operation")
    nx.draw_networkx_edge_labels(G, pos=pos_hierarchy, edge_labels=edge_labels, font_size=8)

    plt.title("Top-Down Dependency Graph of JAXpr Operations with Node Shapes, Data Types, and Edge Operations", fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


visualize_graph(G_detailed, shape_map)

def find_longest_path(G):
    longest_path = nx.dag_longest_path(G)
    return longest_path

def get_longest_path_endpoints(longest_path):
    if not longest_path:
        return None, None
    start_node = longest_path[0]
    end_node = longest_path[-1]
    return start_node, end_node


longest_path = find_longest_path(G_detailed)
print("最长路径:", " -> ".join(longest_path))

# 提取起始和结束节点
start_node, end_node = get_longest_path_endpoints(longest_path)
print(f"最长路径的起始节点: {start_node}")
print(f"最长路径的结束节点: {end_node}")
