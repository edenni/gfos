import logging
from typing import Literal

import torch
import torch.nn as nn
import torch_geometric.nn as geonn
from torch_geometric.data import Batch, Data

from .utils import aggregate_neighbors

logger = logging.getLogger(__name__)


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        v2: bool = False,
        dropout: float = 0,
        **conv_kwargs,
    ):
        super(GAT, self).__init__()

        if conv_kwargs is not None:
            logger.info(f"Set kwargs: {conv_kwargs} for GATConv")
        conv = geonn.GATv2Conv if v2 else geonn.GATConv
        self.dropout = dropout

        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        self.conv_layers = nn.ModuleList()

        for in_plane, out_plane in zip(channels[:-1], channels[1:]):
            self.conv_layers.append(conv(in_plane, out_plane, **conv_kwargs))

    def forward(self, x, edge_index):
        if self.dropout > 0:
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        for conv_layer in self.conv_layers:
            x, (edge_index, edge_weight) = conv_layer(
                x, edge_index, return_attention_weights=True
            )
            x = nn.functional.leaky_relu(x)

        return x, (edge_index, edge_weight)


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
        config_neighbor_layer: Literal["GATConv", "GCNConv", "SAGEConv"] = "SAGEConv",
        num_config_neighbor_layers: int = 2,
        config_neighbor_dim: int = 64,
        config_neighbor_dropout_between_layers: float = 0.0,
        return_attention_weights: bool = False,
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
        self.return_attention_weights = return_attention_weights
        # self.opcode_weight = nn.Parameter(torch.ones(1, requires_grad=True) * 100)
        # self.config_weights = nn.Parameter(
        #     torch.ones(config_dim, requires_grad=True) * 100
        # )

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
            return_attention_weights=self.return_attention_weights,
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
            use_edge_weight=self.use_weight
            or self.use_attr
            or return_attention_weights,
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
        return_attention_weights: bool = False,
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
        assert num_layers > 1, f"num_layers must be greater than 1 but got {num_layers}"

        conv_layer = getattr(geonn, conv_layer)
        activation = getattr(nn, activation)

        if use_edge_weight:
            logger.info(f"Use edge weight for {conv_layer}")

        if return_attention_weights and (
            conv_layer in (geonn.conv.gat_conv.GATConv, geonn.conv.gatv2_conv.GATv2Conv)
        ):
            logger.info(f"return attention weights from {conv_layer}")

            return GAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                v2=conv_layer == geonn.GATv2Conv,
                dropout=dropout,
                **conv_kwargs,
            )

        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        if conv_kwargs:
            logger.info(f"Set kwargs: {conv_kwargs} for {conv_layer}")

        conv_layers = []
        if dropout > 0:
            conv_layers.append((nn.Dropout(p=dropout), "x0 -> x0"))

        for i, (in_plane, out_plane) in enumerate(zip(channels[:-1], channels[1:])):
            # Not added at the start of the module
            if inner_dropout > 0 and len(conv_layers) > 0:
                if not (
                    isinstance(conv_layers[-1], tuple)
                    and len(conv_layers[-1]) > 0
                    and isinstance(conv_layers[-1][0], nn.Dropout)
                ):
                    conv_layers.append((nn.Dropout(p=inner_dropout), f"x{i} -> x{i}"))

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
            "x0, edge_index, edge_weight" if use_edge_weight else "x0, edge_index",
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
        node_config_feat_batch: torch.Tensor = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        c = node_config_feat.size(0)
        code_feat = self.embedding(node_opcode)

        # x = torch.cat([node_feat, self.opcode_weight * code_feat], dim=1)
        x = torch.cat([node_feat, code_feat], dim=1)

        x = nn.functional.normalize(x, dim=-1)

        # (N, in_channels) -> (N, node_dim)
        x = self.node_gnn(x, edge_index)

        # (N, node_dim) -> (NC, node_dim)
        config_neighbors = aggregate_neighbors(x, edge_index)[node_config_ids]

        if not self.return_attention_weights:
            config_neighbors = self.config_neighbor_gnn(
                config_neighbors, config_edge_index
            )
        else:
            (
                config_neighbors,
                (
                    config_edge_index,
                    config_edge_weight,
                ),
            ) = self.config_neighbor_gnn(
                config_neighbors,
                config_edge_index,
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
                node_config_feat,  # self.config_weights * node_config_feat,
            ],
            dim=-1,
        )
        x = nn.functional.normalize(x, dim=-1)

        # (C, NC, merged_node_dim) -> (C, NC, config_dim)
        if self.return_attention_weights:
            datas = [
                Data(
                    x=x[i],
                    edge_index=config_edge_index,
                    edge_weight=config_edge_weight,
                )
                for i in range(x.shape[0])
            ]
            batch = Batch.from_data_list(datas)
            x = self.config_gnn(batch.x, batch.edge_index, batch.edge_weight)
        else:
            datas = [
                Data(
                    x=x[i],
                    edge_index=config_edge_index,
                )
                for i in range(x.shape[0])
            ]
            batch = Batch.from_data_list(datas)
            x = self.config_gnn(batch.x, batch.edge_index)

            b = (
                batch.batch
                if batch_size < 2
                else (
                    node_config_feat_batch.repeat((c, 1))
                    + torch.arange(c).unsqueeze(0).T.to(node_config_feat_batch.device)
                    * batch_size
                ).view(-1)
            )

        # (C*NC, config_dim) -> (C, config_dim)
        x = geonn.pool.global_mean_pool(x, b)
        # (C, config_dim) -> (C,)
        x = self.dense(x).flatten()

        return x
