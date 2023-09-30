import os
from collections import defaultdict

import torch


def load_layout(
    base_dir: str,
    compile_type: str | None = None,
    model_type: str | None = None,
):
    if model_type is not None:
        assert model_type in (
            "nlp",
            "xla",
        ), f"model_type must be nlp or xla but got {model_type}"

    if compile_type is not None:
        assert compile_type in (
            "default",
            "random",
        ), f"compile_type must be default or random but got {compile_type}"

    dfs = defaultdict(list)

    if model_type is None:
        model_types = ("nlp", "xla")
    else:
        model_types = (model_type,)

    if compile_type is None:
        compile_types = ("default", "random")
    else:
        compile_types = (compile_type,)

    dirs = [
        os.path.join(base_dir, model_type, compile_type, training)
        for model_type in model_types
        for compile_type in compile_types
        for training in ["train", "valid", "test"]
    ]

    for path in dirs:
        split = path.split("\\")[-1]
        files = os.listdir(path)

        dfs[split] += [os.path.join(path, file) for file in files]

    return dfs


def edges_adjacency(edges: torch.Tensor, add_diagonal=True) -> torch.Tensor:
    """
    Generate an adjacency matrix from the edges
    Args:
        edges: Tensor of shape (num_edges, 2) with the edges
        add_diagonal: Boolean indicating if the diagonal should be added to the adjacency matrix
    Returns:
        adjacency_matrix: Tensor of shape (num_nodes, num_nodes) with the adjacency matrix
    """
    adjacency_matrix = torch.zeros(
        (edges.max() + 1, edges.max() + 1), device=edges.device
    )
    adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
    if add_diagonal:
        diag_idx = torch.arange(adjacency_matrix.shape[0])
        adjacency_matrix[diag_idx, diag_idx] = 1
    return adjacency_matrix


def aggregate_neighbors(node_feat: torch.Tensor, edge_index: torch.Tensor):
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    in_degree_features = torch.zeros_like(node_feat, device=node_feat.device)
    out_degree_features = torch.zeros_like(node_feat, device=node_feat.device)

    source_node_features = node_feat[source_nodes]
    target_node_features = node_feat[target_nodes]

    in_degree_features.scatter_reduce_(
        0,
        target_nodes.unsqueeze(-1).expand_as(source_node_features),
        source_node_features,
        reduce="mean",
    )

    out_degree_features.scatter_reduce_(
        0,
        source_nodes.unsqueeze(-1).expand_as(target_node_features),
        target_node_features,
        reduce="mean",
    )

    return out_degree_features - in_degree_features
