import numpy as np
import networkx as nx
from typing import Union, Tuple, List


def deg(node: int, graph: Union[nx.Graph, nx.DiGraph]) -> Union[int, Tuple[int, int]]:
    if isinstance(graph, nx.DiGraph):
        in_deg = len(list(graph.predecessors(node)))
        out_deg = len(list(graph.successors(node)))
        return in_deg, out_deg
    elif isinstance(graph, nx.Graph):
        return len(graph[node])
    else:
        raise TypeError(f'not supported for the input types: {type(graph)}')
        
def calculate_alpha(lamba, active_num, c1, infected_num, c2):
    alpha = ((lamba * active_num )+ c1) / (infected_num + c2)
    return alpha

def graph2adj_matrix(graph: Union[nx.Graph, nx.DiGraph], alpha: float = 0.1) -> np.array:
    num_nodes = len(graph.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes)).astype(np.float)

    if isinstance(graph, nx.DiGraph):
        for a, b in graph.edges():
            deg_a = deg(a, graph)
            deg_b = deg(b, graph)
            adj_matrix[a][b] = alpha * np.log(deg_a[1]+1) / (alpha *np.log(deg_a[1]+1) + (1-alpha)*np.log(deg_b[0]+1))          
    elif isinstance(graph, nx.Graph):
        for a, b in graph.edges():
            deg_a = deg(a, graph)
            deg_b = deg(b, graph)
            adj_matrix[a][b] = alpha * np.log2(deg_a+1) / (alpha * np.log2(deg_a+1) +(1-alpha)*np.log2(deg_b+1))
            adj_matrix[b][a] = alpha * np.log2(deg_b+1) / (alpha *np.log2(deg_b+1) +(1-alpha)* np.log2(deg_a+1))
    else:
        raise TypeError(f'not supported for the input types: {type(graph)}')

    return adj_matrix


def one_hot2idx(array: np.ndarray) -> List:
    return list(np.arange(len(array))[array])


def n_depth_neighbors(graph, source, depth=1):
    neighbors = []
    for neighbor in dict(nx.bfs_successors(graph, source, depth)).values():
        neighbors = neighbors + neighbor
    return neighbors

def read_graph_from_file(file_path, directed):
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            if len(nodes) == 2:
                node1, node2 = nodes
                node1=int(node1)
                node2=int(node2)
                graph.add_edge(node1, node2)
    return graph
