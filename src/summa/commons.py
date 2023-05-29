from .graph import Graph
import random

def build_graph(sequence):
    graph = Graph()
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
        else:
            item = item + str(random.randint(0,1000000))
            graph.add_node(item)
    return graph

def remove_unreachable_nodes(graph):
    for node in graph.nodes():
        if sum(graph.edge_weight((node, other)) for other in graph.neighbors(node)) == 0:
            graph.del_node(node)
