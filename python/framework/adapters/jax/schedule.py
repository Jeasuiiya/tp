from typing import Any, Dict, List, Tuple
from framework.core._graph import (
    SubGraph,
    Block,
    DataType,
)

__doc__ = "schedule blocks"

NodeName = str
GraphPortRef = Tuple[NodeName, int]
BlockPortRef = Tuple[Block, int]
BlockId = int
PortIndex = int
BlockInputPort = Tuple[PortIndex, BlockId, int, DataType, List]
BlockOutPort = Tuple[PortIndex, DataType, List]


class ScheduleContext:
    """
    schedule context status
    """

    graph2block = Dict[SubGraph, Block]
    block2graph = Dict[Block, SubGraph]
    entry_blocks = List[Block]
    invars = List[GraphPortRef]
    returns = List[GraphPortRef]
    block_input_var = List[BlockPortRef]
    block_outputs = Dict[BlockId, List]
    nodeoutput_blockoutput = Dict[GraphPortRef, BlockPortRef]
    blockoutput_nodeoutput = Dict[BlockPortRef, GraphPortRef]
    execute_successor = Dict[Block, List[Block]]
    node_output_type = Dict[GraphPortRef, Any]

    def __init__(self, invars, returns, node_output_type, out_tree, cache_block_executable):
        self.invars = invars
        self.returns = returns
        self.node_output_type = node_output_type
        self.graph2block = {}
        self.block2graph = {}
        self.entry_blocks = []
        self.block_outputs = {}
        self.block_input_var = [None] * len(invars)
        self.nodeoutput_blockoutput = {}
        self.blockoutput_nodeoutput = {}
        self.execute_successor = {}
        self.out_tree = out_tree
        self.cache_block_executable = cache_block_executable

    def order(self):
        """
        return blocks with topo order
        """
        queue = []
        ready = set({(0, i) for i in range(len(self.invars))})

        def can_enqueue(b):
            if b.inputports_size == 0 and b.outputports_size > 0:
                return True
            if b.inputports_size == 0 and b.outputports_size == 0:
                return False
            complete_flag = 0
            in_ports = b.inputports
            for i in map(lambda x: (x[1], x[2]), in_ports):
                # print(i)
                if i in ready:
                    complete_flag = complete_flag + 1
            if complete_flag == len(in_ports):
                return True
            return False

        def enqueue(block, callback=None):
            if callback is None:
                callback = queue.append
            if can_enqueue(block):
                callback(b)

        queue_level = []
        for b in self.block2graph:
            enqueue(b, queue_level.append)
        queue.append(queue_level)
        ready = set({(0, i) for i in range(len(self.invars))})
        while len(queue) != 0:
            queue_level = queue[0]
            queue.pop(0)
            yield queue_level
            for current_block in queue_level:
                # yield current_block
                for i in current_block.outputports:
                    ready.add((current_block.id, i[0]))
            next_level = []
            for current_block in queue_level:
                for b in self.execute_successor.get(current_block, ()):
                    enqueue(b, next_level.append)
            if len(next_level) > 0:
                queue.append(next_level)

    def block(self, graph: SubGraph):
        b = Block(graph)
        # record ref
        self.graph2block[graph] = b
        self.block2graph[b] = graph
        return b

    def blocks(self, graphs: List[SubGraph]):
        """
        register subgraphs as blocks
        """
        for i in graphs:
            self.block(i)

    def _prepare_outputs(self):
        return_node_names = []
        if self.returns is not None and len(self.returns) != 0:
            return_node_names = list(zip(*self.returns))[0]
        for i in self.graph2block.values():
            graph = i.graph
            for j in range(graph.nodes_num):
                node = graph.get_node(j)
                # print(j, node)
                if node.name in return_node_names:
                    for k in node.output_indexes():
                        if node.output_name(k) in self.returns:
                            i.add_outputport(node.output_type(k), node.output_shape(k))
                            block_ref = (i.id, i.outputports[-1][0])
                            node_ref = node.output_name(k)
                            self.nodeoutput_blockoutput[node_ref] = block_ref
                            self.blockoutput_nodeoutput[block_ref] = node_ref

            for output_maps in graph.outputs:
                for k in output_maps:
                    node = graph.get_node(k[0][0])
                    node_ref = node.output_name(k[0][1])
                    if self.nodeoutput_blockoutput.get(node_ref, None) is None:
                        i.add_outputport(node.output_type(k[0][1]), node.output_shape(k[0][1]))
                        block_ref = (i.id, i.outputports[-1][0])
                        self.nodeoutput_blockoutput[node_ref] = block_ref
                        self.blockoutput_nodeoutput[block_ref] = node_ref

    def _prepare_inputs(self):
        input_node_names = []
        if self.invars is not None and len(self.invars) != 0:
            input_node_names = list(zip(*self.invars))[0]
        for i in self.graph2block.values():
            graph = i.graph
            # external input first
            for j in range(graph.nodes_num):
                node = graph.get_node(j)
                if node.op == "Input" and node.name in input_node_names:
                    if i not in self.entry_blocks:
                        self.entry_blocks.append(i)
                    global_input_index = input_node_names.index(node.name)
                    # 0 block means external input. Input Op only has 1 output, type and shape index is 0
                    i.add_inputport(0, global_input_index, node.output_type(0), node.output_shape(0))
                    # record global input to current block port

                    self.block_input_var[global_input_index] = (i.id, len(i.inputports) - 1)
                    self.blockoutput_nodeoutput[(0, global_input_index)] = node.output_name(0)
            for g, input_maps in zip(graph.input_graphs, graph.inputs):
                ref_block = self.graph2block[g]
                ref_block_id = ref_block.id
                if self.execute_successor.get(ref_block, None) is None:
                    self.execute_successor[ref_block] = [i]
                else:
                    self.execute_successor[ref_block].append(i)
                node_ref_set = set()
                for k in input_maps:  #   input_maps: List[Tuple[GraphPortRef, GraphPortRef]]
                    if k[0] in node_ref_set:
                        continue
                    node_ref_set.add(k[0])
                    node = g.get_node(k[0][0])
                    i.add_inputport(
                        ref_block_id,
                        self.nodeoutput_blockoutput[node.output_name(k[0][1])][1],
                        node.output_type(k[0][1]),
                        node.output_shape(k[0][1]),
                    )

    def regular_blocks(self):
        self._prepare_outputs()
        self._prepare_inputs()
