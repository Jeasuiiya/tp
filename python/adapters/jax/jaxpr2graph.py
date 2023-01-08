import collections
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple
import jax
import jax.numpy as jnp
import jax.core as jcore
import itertools as it
import sys
from framework.core._graph import Graph, Node

NODE_NAME_RECORE = dict()


def gen_name(prefix):
    if NODE_NAME_RECORE.get(prefix, None) is None:
        index = 0
    else:
        index = NODE_NAME_RECORE[prefix] + 1
    NODE_NAME_RECORE[prefix] = index
    return "{}_{}".format(prefix, index)


class ConvertContext:
    var_ids: DefaultDict[jcore.Var, int]
    id_vars: Dict[int, jcore.Var]
    var_outputs: Dict[jcore.Atom, Tuple[Node, int]]
    literal_inputs = Dict[str, Node]

    def __init__(self):
        self.var_ids = collections.defaultdict(it.count().__next__, {})
        self.id_vars = dict()
        self.var_outputs = dict()
        self.literal_inputs = dict()

    def register_var(self, var: jcore.Var):
        var_id = self.var_ids[var]
        self.id_vars[var_id] = var

    def register_output(self, var, node, index):
        self.var_outputs[var] = (node, index)


def process_var(v: jcore.Var, context: ConvertContext, name_prifix=None):
    if isinstance(v, (jcore.Literal, jcore.DropVar)):
        if context.literal_node.get(v.val) is None:
            node = Node(gen_name("Literal"), "Literal")
            node.attrs["value"] = v.val

            return node
        else:
            return context.literal_node[v.val]
    if context.var_outputs.get(v) is None:
        context.register_var(v)
        prefix = name_prifix + "_" if name_prifix is not None else ""
        node = Node(f"{prefix}{context.var_ids[v]}{v.suffix}", "Input")
        context.register_output(v, node, 0)
        return node
    else:
        return context.var_outputs[v]


def process_vars(vs: Sequence[Any], context: ConvertContext, name_prifix=None):
    return [process_var(i, context, name_prifix) for i in vs]


def process_eqn_invars(invars, in_node, context: ConvertContext):
    for i, v in enumerate(invars):
        if isinstance(v, (jcore.Literal, jcore.DropVar)):
            node = Node(gen_name("Literal"), "Literal")
            # node.attrs["value"] = v.val
            context.literal_inputs[(in_node, i)] = node


def process_eqn(eqn, context: ConvertContext):

    [context.register_var(var) for var in eqn.outvars]
    node = Node(gen_name(eqn.primitive.name), eqn.primitive.name)
    [context.register_output(var, node, i)
     for i, var in enumerate(eqn.outvars)]
    node.outputs = ["{}:{}".format(
        node.name, context.var_outputs[var][1]) for var in eqn.outvars]
    process_eqn_invars(eqn.invars, node, context)

    def input_name(index, var):
        if isinstance(var, jcore.Literal):
            node_name = context.literal_inputs[(node, index)].name
            index = 0
        else:
            node_name = context.var_outputs[var][0].name
            index = context.var_outputs[var][1]
        return "{}:{}".format(node_name, index)
    node.inputs = [input_name(i, v) for i, v in enumerate(eqn.invars)]
    return node


def process_eqns(eqns, context: ConvertContext):
    return [process_eqn(i, context) for i in eqns]


def process_output(outvar, context: ConvertContext):
    node_name = context.var_outputs[outvar][0].name
    index = context.var_outputs[outvar][1]
    return "{}:{}".format(node_name, index)


def process_outputs(outvars, context: ConvertContext):
    return [process_output(i, context) for i in outvars]


def jaxpr2graph(jaxpr: jcore.ClosedJaxpr):
    context = ConvertContext()
    graph = Graph()
    input_nodes = process_vars(jaxpr.jaxpr.invars, context, "input")
    eqn_nodes = process_eqns(jaxpr.jaxpr.eqns, context)
    ouputs = process_outputs(jaxpr.jaxpr.outvars, context)

    [graph.add_node(n) for n in input_nodes]
    [graph.add_node(n) for n in context.literal_inputs.values()]
    [graph.add_node(n) for n in eqn_nodes]
    # graph.outputs = ouputs
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
