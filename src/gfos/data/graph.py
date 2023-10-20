import logging
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)


def get_config_graph(
    origin_edges: np.array,
    config_node_ids: np.array,
):
    g = Graph()

    for src, tgt in origin_edges:
        g.add_edge(src, tgt)

    trimmed_graph, distances, paths = g.trim(config_node_ids.tolist())

    config_node_index = []
    for src, tgts in trimmed_graph.items():
        if not tgts:
            continue
        for tgt in tgts:
            config_node_index.append([src, tgt])

    edge_weights = [distances[src][tgt] for src, tgt in config_node_index]
    edge_weights = np.array(edge_weights)
    if edge_weights.min() == 0:
        logger.warning("Zero distance found in graph")
    edge_weights = np.where(edge_weights == 0, 1, edge_weights)
    edge_weights = np.max(edge_weights) / edge_weights

    # map node ids to index in paths
    mapped_paths = []
    for src, tgt in config_node_index:
        # path still use original node ids
        # which will be used as index in node_feat
        path = paths[src][tgt]
        if len(path) == 0:
            path = [src, tgt]
        mapped_paths.append(path)

    # map node ids to index
    edge_mapping = {node: i for i, node in enumerate(config_node_ids)}
    config_node_index = [
        [edge_mapping[src], edge_mapping[tgt]]
        for src, tgt in config_node_index
    ]
    config_node_index = np.array(config_node_index)

    return config_node_index, edge_weights, mapped_paths


class Graph:
    def __init__(self):
        self.graph = defaultdict(set)

    def add_edge(self, u, v):
        self.graph[u].add(v)

    def trim(self, specified_nodes: list):
        """Get graph of specified nodes and their neighbors.
        Trim those nodes that are not specified.
        """
        trimmed_graph = defaultdict(set)
        visited_global = set()  # to keep track of globally visited nodes
        specified_nodes = set(specified_nodes)

        distance_between_nodes = defaultdict(lambda: defaultdict(int))
        node_path = defaultdict(dict)

        for src in specified_nodes:
            if src in visited_global:  # skip already visited nodes
                continue

            visited = set([src])
            queue = deque([(src, 1, [])])

            while queue:
                node, distance, path = queue.popleft()
                visited_global.add(node)

                for neighbor in self.graph[node]:
                    if neighbor in specified_nodes:
                        trimmed_graph[src].add(neighbor)
                        distance_between_nodes[src][neighbor] = distance + 1
                        node_path[src][neighbor] = path
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(
                            (neighbor, distance + 1, path + [neighbor])
                        )

        return trimmed_graph, distance_between_nodes, node_path

    def trim_return_path(self, specified_nodes: list, return_path: bool):
        """Get graph of specified nodes and their neighbors.
        Trim those nodes that are not specified.
        If there is a path a->b->c, then c will not be a's neighbor.
        """
        trimmed_graph = defaultdict(set)
        visited_global = set()  # to keep track of globally visited nodes
        specified_nodes = set(specified_nodes)
        if return_path:
            distance_between_nodes = defaultdict(dict)

        for src in specified_nodes:
            if src in visited_global:  # skip already visited nodes
                continue

            visited = set([src])
            if return_path:
                queue = deque([(src, list())])
            else:
                queue = deque([src])

            while queue:
                if return_path:
                    node, path = queue.popleft()
                else:
                    node = queue.popleft()
                visited_global.add(node)
                for neighbor in self.graph[node]:
                    if neighbor in specified_nodes:
                        trimmed_graph[src].add(neighbor)
                        if return_path:
                            distance_between_nodes[src][neighbor] = path
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        if return_path:
                            queue.append((neighbor, path + [neighbor]))
                        else:
                            queue.append(neighbor)

        if return_path:
            return trimmed_graph, distance_between_nodes
        else:
            return trimmed_graph
