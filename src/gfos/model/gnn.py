from typing import Literal

import torch
import torch.nn as nn
import torch_geometric.nn as geonn
from torch_geometric.data import Batch, Data

from .utils import aggregate_neighbors


class LayoutModel(torch.nn.Module):
    def __init__(
        self,
        conv_layer: Literal["GATConv", "GCNConv", "SAGEConv"],
        op_embedding_dim: int = 32,
        config_dim: int = 64,
        graph_dim: int = 64,
        node_feat_dim: int = 140,
        node_config_dim: int = 18,
    ):
        super().__init__()

        NUM_OPCODE = 120

        conv_layer = getattr(geonn, conv_layer)

        merged_node_dim = 2 * graph_dim + config_dim

        self.embedding = torch.nn.Embedding(
            NUM_OPCODE,
            op_embedding_dim,
        )
        in_channels = op_embedding_dim + node_feat_dim

        # TODO: create skip connection conv module
        self.model_gnn = geonn.Sequential(
            "x, edge_index",
            [
                (conv_layer(in_channels, graph_dim), "x, edge_index -> x1"),
                nn.LeakyReLU(inplace=True),
                (conv_layer(graph_dim, graph_dim), "x1, edge_index -> x2"),
                (lambda x1, x2: x1 + x2, "x1, x2 -> x3"),
                nn.LeakyReLU(inplace=True),
                (conv_layer(graph_dim, graph_dim), "x3, edge_index -> x4"),
                nn.LeakyReLU(inplace=True),
                (conv_layer(graph_dim, graph_dim), "x4, edge_index -> x5"),
                (lambda x4, x5: x4 + x5, "x4, x5 -> x6"),
                nn.LeakyReLU(inplace=True),
                # (conv_layer(graph_dim, graph_dim), "x5, edge_index -> x6"),
                # nn.LeakyReLU(inplace=True),
                # (conv_layer(graph_dim, graph_dim), "x6, edge_index -> x7"),
                # (lambda x6, x7: x6 + x7, "x6, x7 -> x8"),
                # nn.LeakyReLU(inplace=True),
                # (conv_layer(graph_dim, graph_dim), "x8, edge_index -> x9"),
                # nn.LeakyReLU(inplace=True),
                # (conv_layer(graph_dim, graph_dim), "x9, edge_index -> x10"),
                # (lambda x9, x10: x9 + x10, "x9, x10 -> x11"),
                # nn.LeakyReLU(inplace=True),
            ],
        )

        self.config_mp = geonn.Sequential(
            "x, edge_index",
            [
                (geonn.GATConv(graph_dim, graph_dim), "x, edge_index -> x1"),
                nn.LeakyReLU(inplace=True),
                (geonn.GATConv(graph_dim, graph_dim), "x1, edge_index -> x2"),
                (lambda x1, x2: x1 + x2, "x1, x2 -> x3"),
                nn.LeakyReLU(inplace=True),
            ],
        )

        self.config_gnn = geonn.Sequential(
            "x, edge_index",
            [
                (nn.Dropout(p=0.2), "x -> x"),
                (
                    conv_layer(merged_node_dim, config_dim),
                    "x, edge_index -> x1",
                ),
                nn.LeakyReLU(inplace=True),
                (conv_layer(config_dim, config_dim), "x1, edge_index -> x2"),
                (lambda x1, x2: x1 + x2, "x1, x2 -> x3"),
                nn.LeakyReLU(inplace=True),
                (conv_layer(config_dim, config_dim), "x3, edge_index -> x4"),
                nn.LeakyReLU(inplace=True),
                (
                    conv_layer(config_dim, config_dim),
                    "x4, edge_index -> x5",
                ),
                (lambda x4, x5: x4 + x5, "x4, x5 -> x6"),
                nn.LeakyReLU(inplace=True),
            ],
        )

        self.config_prj = nn.Sequential(
            nn.Linear(node_config_dim, config_dim),
            nn.LeakyReLU(),
        )

        # self.deg_prj = nn.Sequential(
        #     nn.Linear(hidden_channels[-1], merged_node_dim, bias=False),
        #     nn.LayerNorm(merged_node_dim),
        #     nn.LeakyReLU(),
        # )

        self.dense = torch.nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config_dim, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        node_opcode: torch.Tensor,
        edge_index: torch.Tensor,
        node_config_feat: torch.Tensor,
        node_config_ids: torch.Tensor,
        config_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Get graph features
        c = node_config_feat.size(0)

        x = torch.cat([node_feat, self.embedding(node_opcode)], dim=1)

        # Get graph features
        x = self.model_gnn(x, edge_index)

        config_neighbors = aggregate_neighbors(x, edge_index)[node_config_ids]
        config_neighbors = nn.functional.normalize(config_neighbors, dim=-1)
        config_neighbors = self.config_mp(config_neighbors, config_edge_index)

        # (N, graph_out) -> (NC, graph_out)
        x = x[node_config_ids]
        # x += config_neighbors

        # Merge graph features with config features
        # (C, NC, 18) -> (C, NC, config_dim)
        node_config_feat = self.config_prj(node_config_feat)
        # pos_embedding = self.deg_prj(neighbor_feat)

        # (C, NC, 2*graph_out + config_dim)
        x = torch.cat(
            [
                config_neighbors.repeat((c, 1, 1)),
                x.repeat((c, 1, 1)),
                node_config_feat,
            ],
            dim=-1,
        )
        # x += pos_embedding
        x = nn.functional.normalize(x, dim=-1)

        datas = [
            Data(x=x[i], edge_index=config_edge_index)
            for i in range(x.shape[0])
        ]
        batch = Batch.from_data_list(datas)

        x = self.config_gnn(batch.x, batch.edge_index)
        x = geonn.pool.global_mean_pool(x, batch.batch)

        x = self.dense(x).flatten()

        return x
