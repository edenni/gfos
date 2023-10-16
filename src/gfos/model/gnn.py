from typing import Literal

import torch
import torch.nn as nn
import torch_geometric.nn as geonn
from torch_geometric.data import Batch, Data

from .utils import aggregate_neighbors


class LayoutModel(torch.nn.Module):
    def __init__(
        self,
        num_opcodes: int = 120,
        node_feat_dim: int = 140,
        node_config_dim: int = 18,
        op_embedding_dim: int = 32,
        node_layer: Literal["GATConv", "GCNConv", "SAGEConv"] = "SAGEConv",
        num_node_layers: int = 3,
        node_dim: int = 64,
        config_neighbor_layer: Literal[
            "GATConv", "GCNConv", "SAGEConv"
        ] = "SAGEConv",
        num_config_neighbor_layers: int = 3,
        config_neighbor_dim: int = 64,
        config_layer: Literal["GATConv", "GCNConv", "SAGEConv"] = "SAGEConv",
        num_config_layers: int = 3,
        config_dim: int = 64,
        head_dim: int = 64,
        dropout: float = 0.0,
        activation: str = "LeakyReLU",
    ):
        super(LayoutModel, self).__init__()

        merged_node_dim = 2 * node_dim + config_dim

        self.embedding = torch.nn.Embedding(
            num_opcodes,
            op_embedding_dim,
        )
        in_channels = op_embedding_dim + node_feat_dim

        self.node_gnn = self._create_conv_module(
            conv_layer=node_layer,
            num_layers=num_node_layers,
            in_channels=in_channels,
            hidden_channels=node_dim,
            out_channels=node_dim,
            activation=activation,
        )

        self.config_neighbor_gnn = self._create_conv_module(
            conv_layer=config_neighbor_layer,
            num_layers=num_config_neighbor_layers,
            in_channels=node_dim,
            hidden_channels=config_neighbor_dim,
            out_channels=config_neighbor_dim,
            activation=activation,
        )

        self.config_gnn = self._create_conv_module(
            conv_layer=config_layer,
            num_layers=num_config_layers,
            in_channels=merged_node_dim,
            hidden_channels=config_dim,
            out_channels=config_dim,
            activation=activation,
            dropout=dropout,
        )

        self.config_prj = nn.Sequential(
            nn.Linear(node_config_dim, config_dim),
            nn.LeakyReLU(),
        )

        self.dense = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config_dim, head_dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(head_dim, head_dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(head_dim, 1, bias=False),
        )

    def _create_conv_module(
        self,
        conv_layer: str,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: str,
        dropout: float = 0.0,
        **conv_kwargs: dict,
    ) -> nn.Module:
        """
        Create a message passing layer with activation function.

        Args:
            conv_layer(str): name of the convolution layer
            num_layers(int): number of layers
            in_channels(int): number of input channels
            hidden_channels(int): number of hidden channels
            out_channels(int): number of output channels
            activation(str): name of the activation function
            **conv_kwargs(dict): keyword arguments for convolution layer
        Returns:
            nn.Module: a sequential layer with convolution and activation
        """
        assert activation in [
            "ReLU",
            "LeakyReLU",
        ], f"Invalid activation: {activation}"
        assert conv_layer in [
            "GATConv",
            "GCNConv",
            "SAGEConv",
        ], f"Invalid conv layer: {conv_layer}"
        assert (
            num_layers > 1
        ), f"num_layers must be greater than 1 but got {num_layers}"

        conv_layer = getattr(geonn, conv_layer)
        activation = getattr(nn, activation)

        channels = (
            [in_channels]
            + [hidden_channels] * (num_layers - 1)
            + [out_channels]
        )

        conv_layers = []
        if dropout > 0:
            conv_layers.append((nn.Dropout(p=dropout), "x -> x"))

        for in_plane, out_plane in zip(channels[:-1], channels[1:]):
            conv_layers.extend(
                [
                    (
                        conv_layer(in_plane, out_plane, **conv_kwargs),
                        "x, edge_index -> x",
                    ),
                    activation(inplace=True),
                ]
            )

        return geonn.Sequential(
            "x, edge_index",
            conv_layers,
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
        c = node_config_feat.size(0)

        x = torch.cat([node_feat, self.embedding(node_opcode)], dim=1)

        # (N, in_channels) -> (N, node_dim)
        x = self.node_gnn(x, edge_index)

        # (N, node_dim) -> (NC, node_dim)
        config_neighbors = aggregate_neighbors(x, edge_index)[node_config_ids]
        config_neighbors = self.config_neighbor_gnn(
            config_neighbors, config_edge_index
        )

        # (N, node_dim) -> (NC, node_dim)
        x = x[node_config_ids]

        # (C, NC, node_config_dim) -> (C, NC, config_dim)
        node_config_feat = self.config_prj(node_config_feat)

        # (C, NC, merged_node_dim)
        x = torch.cat(
            [
                config_neighbors.repeat((c, 1, 1)),
                x.repeat((c, 1, 1)),
                node_config_feat,
            ],
            dim=-1,
        )
        x = nn.functional.normalize(x, dim=-1)

        datas = [
            Data(x=x[i], edge_index=config_edge_index)
            for i in range(x.shape[0])
        ]
        batch = Batch.from_data_list(datas)

        # (C, NC, merged_node_dim) -> (C, NC, config_dim)
        x = self.config_gnn(batch.x, batch.edge_index)

        # (C, NC, config_dim) -> (C, config_dim)
        x = geonn.pool.global_mean_pool(x, batch.batch)

        # (C, config_dim) -> (C,)
        x = self.dense(x).flatten()

        return x
