import torch
from torch_geometric.data import Batch
from torch_sparse import SparseTensor


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


def get_adj(batch):
    batch_list = batch.to_data_list()
    processed_batch_list = []

    for g in batch_list:
        g.adj = SparseTensor(
            row=g.edge_index[0],
            col=g.edge_index[1],
            sparse_sizes=(g.num_nodes, g.num_nodes),
        )
        g.cadj = SparseTensor(
            row=g.config_edge_index[0],
            col=g.config_edge_index[1],
            sparse_sizes=(g.num_config_nodes, g.num_config_nodes),
        )

        processed_batch_list.append(g)

    return Batch.from_data_list(processed_batch_list)
