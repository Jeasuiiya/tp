import collections
from typing import Any, DefaultDict, Dict, Sequence, Tuple
import itertools as it
import jax
import jax.numpy as jnp
import jax.core as jcore
from framework.core._graph import Graph, Node

__doc__ = """
convert jaxpr to core graph
Author: yiguangzheng
datetime: 2023.1.9
version: 1 2023.1.9 first commit
"""


class ConvertContext:
    """
    hold some map using in coversion and provide someo helper for conversion
    """

    var_ids: DefaultDict[jcore.Var, int]
    id_vars: Dict[int, jcore.Var]
    var_outputs: Dict[jcore.Atom, Tuple[Node, int]]
    literal_inputs = Dict[str, Node]
    node_name_record = {}

    def __init__(self):
        self.var_ids = collections.defaultdict(it.count().__next__, {})
        self.id_vars = {}
        self.var_outputs = {}
        self.literal_inputs = {}
        self.node_name_record = {}

    def register_var(self, var: jcore.Var):
        """
        register var and id such as input, output, const and SSA var in maps
        """
        var_id = self.var_ids[var]
        self.id_vars[var_id] = var

    def register_output(self, var, node, index):
        """
        register node's output.
        an output of a node can be search by a var.
        """
        self.var_outputs[var] = (node, index)

    def gen_name(self, prefix):
        """
        generate node name
        """
        if self.node_name_record.get(prefix, None) is None:
            index = 0
        else:
            index = self.node_name_record[prefix] + 1
        self.node_name_record[prefix] = index
        return f"{prefix}_{index}"


def process_var(v: jcore.Var, context: ConvertContext, name_prifix=None):
    """
    process vars of jaxpr.
    the var will be register as var and output in context.
    """
    if isinstance(v, (jcore.Literal, jcore.DropVar)):
        if context.literal_node.get(v.val) is None:
            node = Node(context.gen_name("Literal"), "Literal")
            node.attrs["value"] = v.val

            return node
        return context.literal_node[v.val]
    if context.var_outputs.get(v) is None:
        context.register_var(v)
        prefix = name_prifix + "_" if name_prifix is not None else ""
        node = Node(f"{prefix}{context.var_ids[v]}{v.suffix}", "Input")
        context.register_output(v, node, 0)
        return node
    return context.var_outputs[v]


def process_vars(vs: Sequence[Any], context: ConvertContext, name_prifix=None):
    """
    register vars
    """
    return [process_var(i, context, name_prifix) for i in vs]


def process_eqn_invars(invars, in_node, context: ConvertContext):
    """
    process eqn input vars.
    """
    for i, v in enumerate(invars):
        if isinstance(v, (jcore.Literal, jcore.DropVar)):
            node = Node(context.gen_name("Literal"), "Literal")
            node.attrs["value"] = v.val
            context.literal_inputs[(in_node, i)] = node


def process_eqn(eqn, context: ConvertContext):
    """
    process eqn.
    every eqn will be register as a node in graph.
    """
    _ = [context.register_var(var) for var in eqn.outvars]
    node = Node(context.gen_name(eqn.primitive.name), eqn.primitive.name)
    _ = [context.register_output(var, node, i) for i, var in enumerate(eqn.outvars)]
    node.outputs = [f"{node.name}:{context.var_outputs[var][1]}" for var in eqn.outvars]
    process_eqn_invars(eqn.invars, node, context)

    def input_name(index, var):
        if isinstance(var, jcore.Literal):
            node_name = context.literal_inputs[(node, index)].name
            index = 0
        else:
            node_name = context.var_outputs[var][0].name
            index = context.var_outputs[var][1]
        return f"{node_name}:{index}"

    node.inputs = [input_name(i, v) for i, v in enumerate(eqn.invars)]
    return node


def process_eqns(eqns, context: ConvertContext):
    """
    process eqns
    """
    return [process_eqn(i, context) for i in eqns]


def process_output(outvar, context: ConvertContext):
    """
    process outputs of jaxpr.
    """
    node_name = context.var_outputs[outvar][0].name
    index = context.var_outputs[outvar][1]
    return f"{node_name}:{index}"


def process_outputs(outvars, context: ConvertContext):
    """
    process outputs.
    """
    return [process_output(i, context) for i in outvars]


def jaxpr2graph(jaxpr: jcore.ClosedJaxpr):
    """
    convert jax to framework graph
    Args:
        jaxpr: a ClosedJaxpr that will be converted
    """
    context = ConvertContext()
    graph = Graph()
    input_nodes = process_vars(jaxpr.jaxpr.invars, context, "input")
    eqn_nodes = process_eqns(jaxpr.jaxpr.eqns, context)
    # outputs = process_outputs(jaxpr.jaxpr.outvars, context)

    _ = [graph.add_node(n) for n in input_nodes]
    _ = [graph.add_node(n) for n in context.literal_inputs.values()]
    _ = [graph.add_node(n) for n in eqn_nodes]
    # graph.outputs = outputs
    return graph


def add(x, y):
    a = x + 1
    return a + y


def main():
    x = jnp.arange(10)
    y = jnp.arange(10)
    pr = jax.make_jaxpr(add)(x, y)
    print(jaxpr2graph(pr))


if __name__ == "__main__":
    main()
