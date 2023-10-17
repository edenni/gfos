from collections import defaultdict, deque

import numpy as np


def get_config_graph(
    origin_edges: np.array,
    config_node_ids: np.array,
    full_connection: bool = False,
    return_distance: bool = False,
):
    g = Graph()

    for src, tgt in origin_edges:
        g.add_edge(src, tgt)

    trimmed_graph = g.trim_and_merge(config_node_ids.tolist(), return_distance)
    if return_distance:
        trimmed_graph, distances = trimmed_graph

    trimmed_edges = []

    for src, tgts in trimmed_graph.items():
        if not tgts:
            continue
        for tgt in tgts:
            trimmed_edges.append([src, tgt])

    trimmed_edges = np.array(trimmed_edges)

    if return_distance:
        weights = [distances[src][tgt] for src, tgt in trimmed_edges]
        weights = np.array(weights)
        weights = weights.max() / weights

    return (trimmed_edges, weights) if return_distance else trimmed_edges


class Graph:
    def __init__(self):
        self.graph = defaultdict(set)

    def add_edge(self, u, v):
        self.graph[u].add(v)

    def trim_and_merge(
        self, specified_nodes: list, return_distance: bool = False
    ):
        """Get graph of specified nodes and their neighbors.
        Trim those nodes that are not specified.
        If there is a path a->b->c, then c will not be a's neighbor.
        """
        trimmed_graph = defaultdict(set)
        visited_global = set()  # to keep track of globally visited nodes
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

    def trim_and_merge_full_connection(self, specified_nodes: list):
        """Get graph of specified nodes and their neighbors.
        If there is a path a->b->c, both b and c will be a's neighbor.
        """
        trimmed_graph = defaultdict(set)

        specified_nodes = set(specified_nodes)

        for src in specified_nodes:
            other_specified_nodes = specified_nodes - {src}

            visited = set()
            queue = deque([src])
            while queue:
                node = queue.popleft()
                if node in other_specified_nodes:
                    trimmed_graph[src].add(node)
                visited.add(node)
                for neighbor in self.graph[node]:
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)

        # Ensure all specified nodes are in the graph even if they are isolated
        for node in specified_nodes:
            if node not in trimmed_graph:
                trimmed_graph[node] = set()

        return dict(trimmed_graph)
