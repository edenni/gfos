from collections import defaultdict, deque

import numpy as np


def get_config_graph(
    origin_edges: np.array,
    config_node_ids: np.array,
    return_distance: bool = False,
):
    g = Graph()

    for src, tgt in origin_edges:
        g.add_edge(src, tgt)

    trimmed_graph = g.trim(config_node_ids.tolist(), return_distance)
    if return_distance:
        trimmed_graph, distances = trimmed_graph

    config_node_index = []
    edge_mapping = {node: i for i, node in enumerate(config_node_ids)}

    for src, tgts in trimmed_graph.items():
        if not tgts:
            continue
        for tgt in tgts:
            config_node_index.append([edge_mapping[src], edge_mapping[tgt]])

    config_node_index = np.array(config_node_index)

    if return_distance:
        edge_weights = [distances[src][tgt] for src, tgt in config_node_index]
        edge_weights = np.array(edge_weights)
        edge_weights = np.max(edge_weights) / edge_weights

    return (
        (config_node_index, edge_weights)
        if return_distance
        else config_node_index
    )


class Graph:
    def __init__(self):
        self.graph = defaultdict(set)

    def add_edge(self, u, v):
        self.graph[u].add(v)

    def trim(self, specified_nodes: list, return_distance: bool):
        """Get graph of specified nodes and their neighbors.
        Trim those nodes that are not specified.
        If there is a path a->b->c, then c will not be a's neighbor.
        """
        trimmed_graph = defaultdict(set)
        visited_global = set()  # to keep track of globally visited nodes
        specified_nodes = set(specified_nodes)
        if return_distance:
            distance_between_nodes = defaultdict(lambda: defaultdict(int))

        for src in specified_nodes:
            if src in visited_global:  # skip already visited nodes
                continue

            visited = set([src])
            if return_distance:
                queue = deque([(src, 1)])
            else:
                queue = deque([src])

            while queue:
                if return_distance:
                    node, distance = queue.popleft()
                else:
                    node = queue.popleft()
                visited_global.add(node)
                for neighbor in self.graph[node]:
                    if neighbor in specified_nodes:
                        trimmed_graph[src].add(neighbor)
                        if return_distance:
                            distance_between_nodes[src][neighbor] = (
                                distance + 1
                            )
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        if return_distance:
                            queue.append((neighbor, distance + 1))
                        else:
                            queue.append(neighbor)

        if return_distance:
            return trimmed_graph, distance_between_nodes
        else:
            return trimmed_graph
