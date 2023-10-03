from collections import defaultdict, deque

import numpy as np


def get_config_graph(origin_edges: np.array, config_node_ids: np.array):
    g = Graph()

    for src, tgt in origin_edges:
        g.add_edge(src, tgt)

    trimmed_graph = g.trim_and_merge(config_node_ids.tolist())

    trimmed_edges = []
    edge_mapping = {node: i for i, node in enumerate(config_node_ids)}

    for src, tgts in trimmed_graph.items():
        if not tgts:
            continue
        for tgt in tgts:
            trimmed_edges.append([edge_mapping[src], edge_mapping[tgt]])

    trimmed_edges = np.array(trimmed_edges)

    return trimmed_edges


class Graph:
    def __init__(self):
        self.graph = defaultdict(set)

    def add_edge(self, u, v):
        self.graph[u].add(v)

    def trim_and_merge(self, specified_nodes: list):
        trimmed_graph = defaultdict(set)
        visited_global = set()  # to keep track of globally visited nodes
        specified_nodes = set(specified_nodes)

        for src in specified_nodes:
            if src in visited_global:  # skip already visited nodes
                continue

            visited = set([src])
            queue = deque([src])

            while queue:
                node = queue.popleft()
                visited_global.add(node)
                for neighbor in self.graph[node]:
                    if neighbor in specified_nodes:
                        trimmed_graph[src].add(neighbor)
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return trimmed_graph

    def trim_and_merge_full_connection(self, specified_nodes: list):
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
