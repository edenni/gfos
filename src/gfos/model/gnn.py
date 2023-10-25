import logging
from typing import Literal

import torch
import torch.nn as nn
import torch_geometric.nn as geonn
from torch_geometric.data import Batch, Data

from .utils import aggregate_neighbors

logger = logging.getLogger(__name__)


class LayoutModel(torch.nn.Module):
    def __init__(
        self,
        num_opcodes: int = 120,
        node_feat_dim: int = 140,
        node_config_dim: int = 18,
        op_embedding_dim: int = 32,
        node_layer: Literal["GATConv", "GCNConv", "SAGEConv"] = "SAGEConv",
        num_node_layers: int = 4,
        node_dim: int = 64,
        node_dropout_between_layers: float = 0.0,
        node_conv_kwargs: dict = {},
        config_neighbor_layer: Literal[
            "GATConv", "GCNConv", "SAGEConv"
        ] = "SAGEConv",
        num_config_neighbor_layers: int = 2,
        config_neighbor_dim: int = 64,
        config_neighbor_dropout_between_layers: float = 0.0,
        config_neighbor_conv_kwargs: dict = {},
        config_layer: Literal["GATConv", "GCNConv", "SAGEConv"] = "SAGEConv",
        num_config_layers: int = 4,
        config_dim: int = 64,
        config_dropout_between_layers: float = 0.0,
        use_config_edge_weight: bool = False,
        use_config_edge_attr: bool = False,
        edge_dim: int = 32,
        jk_mode: Literal["cat", "max", "lstm"] = None,
        jk_kwargs: dict = {},
        config_conv_kwargs: dict = {},
        head_dim: int = 64,
        dropout: float = 0.0,
        activation: str = "LeakyReLU",
    ):
        super(LayoutModel, self).__init__()
        self.use_weight = use_config_edge_weight
        self.use_attr = use_config_edge_attr

        if not self.use_weight:
            logger.warning("Disable config edge weight")

        merged_node_dim = node_dim + config_neighbor_dim + config_dim

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
            inner_dropout=node_dropout_between_layers,
            use_edge_weight=False,
            jk_mode=jk_mode,
            jk_kwargs=jk_kwargs,
            **node_conv_kwargs,
        )

        if use_config_edge_attr:
            self.node2edge = nn.Sequential(
                nn.Linear(node_dim, edge_dim),
                nn.LeakyReLU(),
            )

        self.config_neighbor_gnn = self._create_conv_module(
            conv_layer=config_neighbor_layer,
            num_layers=num_config_neighbor_layers,
            in_channels=node_dim,
            hidden_channels=config_neighbor_dim,
            out_channels=config_neighbor_dim,
            activation=activation,
            inner_dropout=config_neighbor_dropout_between_layers,
            use_edge_weight=False,
            **config_neighbor_conv_kwargs,
        )

        self.config_gnn = self._create_conv_module(
            conv_layer=config_layer,
            num_layers=num_config_layers,
            in_channels=merged_node_dim,
            hidden_channels=config_dim,
            out_channels=config_dim,
            activation=activation,
            inner_dropout=config_dropout_between_layers,
            dropout=dropout,
            use_edge_weight=self.use_weight or self.use_attr,
            **config_conv_kwargs,
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
        inner_dropout: float = 0.0,
        use_edge_weight: bool = False,
        jk_mode: str = None,
        jk_kwargs: dict = {},
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

        if conv_kwargs:
            logger.info(f"Set kwargs: {conv_kwargs} for {conv_layer}")

        conv_layers = []
        if dropout > 0:
            conv_layers.append((nn.Dropout(p=dropout), "x0 -> x0"))

        for i, (in_plane, out_plane) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            # Not added at the start of the module
            if inner_dropout > 0 and len(conv_layers) > 0:
                if not (
                    isinstance(conv_layers[-1], tuple)
                    and len(conv_layers[-1]) > 0
                    and isinstance(conv_layers[-1][0], nn.Dropout)
                ):
                    conv_layers.append(
                        (nn.Dropout(p=inner_dropout), f"x{i} -> x{i}")
                    )
            conv_layers.extend(
                [
                    (
                        conv_layer(in_plane, out_plane, **conv_kwargs),
                        f"x{i}, edge_index, edge_weight -> x{i+1}"
                        if use_edge_weight
                        else f"x{i}, edge_index -> x{i+1}",
                    ),
                    activation(inplace=True),
                ]
            )

        xs = ",".join([f"x{i+1}" for i in range(num_layers)])

        if jk_mode is not None:
            conv_layers.extend(
                [
                    (lambda *xs: list(xs), f"{xs} -> xs"),
                    (
                        geonn.models.JumpingKnowledge(jk_mode, **jk_kwargs),
                        "xs -> x",
                    ),
                ]
            )

        return geonn.Sequential(
            "x0, edge_index, edge_weight"
            if use_edge_weight
            else "x0, edge_index",
            conv_layers,
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        node_opcode: torch.Tensor,
        edge_index: torch.Tensor,
        node_config_feat: torch.Tensor,
        node_config_ids: torch.Tensor,
        config_edge_index: torch.Tensor = None,
        config_edge_weight: torch.Tensor = None,
        # config_edge_mask: torch.Tensor = None,
        # config_edge_path_len: torch.Tensor = None,
        # config_edge_path: list[list[int]] = None,
    ) -> torch.Tensor:
        c = node_config_feat.size(0)

        x = torch.cat([node_feat, self.embedding(node_opcode)], dim=1)

        # (N, in_channels) -> (N, node_dim)
        x = self.node_gnn(x, edge_index)

        # # if self.use_attr:
        #     # config_edge_attr = torch.stack(
        #     #     [x[path].mean(dim=0) for path in config_edge_path]
        #     # )
        #     # config_edge_attr = self.reduce_node_to_edge(x, config_paths)
        #     # config_edge_attr = (
        #     #     x.expand_as(config_edge_mask) * config_edge_mask
        #     # ).sum(dim=1) / config_edge_path_len
        #     config_edge_attr = self.node2edge(
        #         config_edge_attr
        #     )  # (NC, edge_dim)
        # else:
        #     config_edge_attr = None

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
            Data(
                x=x[i],
                edge_index=config_edge_index,
                # edge_weight=config_edge_weight,
                # edge_attr=config_edge_attr,
            )
            for i in range(x.shape[0])
        ]
        batch = Batch.from_data_list(datas)

        # # (C, NC, merged_node_dim) -> (C, NC, config_dim)
        # if self.use_attr:
        #     x = self.config_gnn(batch.x, batch.edge_index, batch.edge_attr)
        # elif self.use_weight:
        # x = self.config_gnn(batch.x, batch.edge_index, batch.edge_weight)
        # else:
        x = self.config_gnn(batch.x, batch.edge_index)

        # (C, NC, config_dim) -> (C, config_dim)
        x = geonn.pool.global_mean_pool(x, batch.batch)

        # (C, config_dim) -> (C,)
        x = self.dense(x).flatten()

        return x
