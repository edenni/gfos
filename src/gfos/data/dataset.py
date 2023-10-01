from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gfos.data.graph import get_config_graph


class LazyLayoutDataset(Dataset):
    """Load data when getitem."""

    def __init__(
        self,
        files: list[str],
        max_configs: int = -1,
        permute: bool = True,
        config_edges: bool = False,
    ):
        self.max_configs = max_configs
        self.files = files
        self.permute = permute
        self.config_edges = config_edges

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> dict[str, Any]:
        file = self.files[idx]
        model_id = file.split("\\")[-1].split(".")[0]
        record = np.load(file)

        # Random sample `max_configs` configs from all `c` configs
        max_configs = (
            self.max_configs
            if self.max_configs > 0
            else len(record["config_runtime"])
        )

        config_runtime = torch.tensor(
            record["config_runtime"], dtype=torch.float
        )
        config_runtime = (config_runtime - config_runtime.min()) / (
            config_runtime.max() - config_runtime.min()
        )

        if self.permute:
            config_indices = torch.randperm(config_runtime.size(0))[
                :max_configs
            ]
        else:
            config_indices = torch.arange(max_configs)
        config_runtime = config_runtime[config_indices]

        node_feat = torch.tensor(record["node_feat"], dtype=torch.float)
        node_opcode = torch.tensor(record["node_opcode"], dtype=torch.long)
        edge_index = torch.tensor(
            np.swapaxes(record["edge_index"], 0, 1), dtype=torch.long
        )

        node_config_feat = torch.tensor(
            record["node_config_feat"], dtype=torch.float
        )[config_indices]

        node_config_ids = torch.tensor(
            record["node_config_ids"], dtype=torch.long
        )

        sample = dict(
            model_id=model_id,
            node_feat=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            node_config_feat=node_config_feat,
            node_config_ids=node_config_ids,
            config_runtime=config_runtime,
        )

        if self.config_edges:
            config_edge_index = get_config_graph(
                record["edge_index"], record["node_config_ids"]
            )
            config_edge_index = torch.tensor(
                np.swapaxes(config_edge_index, 0, 1), dtype=torch.long
            )
            sample["config_edge_index"] = config_edge_index

        return sample


class LayoutDataset(Dataset):
    """Load all data in advance."""

    def __init__(
        self,
        files: list[str],
        max_configs: int = -1,
        num_configs: int = -1,
        config_edges: bool = False,
    ):
        self.max_configs = max_configs
        self.num_configs = num_configs
        self.files = files
        self.config_edges = config_edges

        self.data = []

        for file in tqdm(self.files, desc="Loading data"):
            record = dict(np.load(file))
            model_id = file.split("\\")[-1].split(".")[0]
            record["model_id"] = model_id
            runtime = record["config_runtime"]

            runtime_sampled, config_indices = sample_configs(
                runtime, max_configs
            )
            runtime_norm = (runtime_sampled - runtime_sampled.min()) / (
                runtime_sampled.max() - runtime_sampled.min()
            )

            record["config_runtime"] = runtime_norm
            record["node_config_feat"] = record["node_config_feat"][
                config_indices
            ]

            if self.config_edges:
                record["config_edge_index"] = get_config_graph(
                    record["edge_index"], record["node_config_ids"]
                )

            self.data.append(record)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> dict[str, Any]:
        record = self.data[idx]
        config_runtime = torch.tensor(
            record["config_runtime"], dtype=torch.float
        )

        if self.num_configs > 0:
            num_configs = self.num_configs
        elif self.max_configs > 0:
            num_configs = self.max_configs
        else:
            num_configs = config_runtime.size(0)

        # Shuffle
        if self.max_configs > 0 or self.num_configs > 0:
            config_indices = torch.randperm(config_runtime.size(0))[
                :num_configs
            ]
        else:
            config_indices = torch.arange(num_configs)
        config_runtime = config_runtime[config_indices]

        model_id = record["model_id"]
        node_feat = torch.tensor(record["node_feat"], dtype=torch.float)
        node_opcode = torch.tensor(record["node_opcode"], dtype=torch.long)
        edge_index = torch.tensor(
            np.swapaxes(record["edge_index"], 0, 1), dtype=torch.long
        )

        node_config_feat = torch.tensor(
            record["node_config_feat"], dtype=torch.float
        )
        node_config_feat = node_config_feat[config_indices]

        node_config_ids = torch.tensor(
            record["node_config_ids"], dtype=torch.long
        )

        sample = dict(
            model_id=model_id,
            node_feat=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            node_config_feat=node_config_feat,
            node_config_ids=node_config_ids,
            config_runtime=config_runtime,
        )

        if self.config_edges:
            config_edge_index = torch.tensor(
                np.swapaxes(record["config_edge_index"], 0, 1),
                dtype=torch.long,
            )
            sample["config_edge_index"] = config_edge_index

        return sample


def sample_configs(
    config_runtime: np.array, max_configs: int
) -> (np.array, np.array):
    """Sample 1/3 max_configs of best configs and 1/3 of worst configs,
    and the rest randomly. Return the sampled configs and indices.
    """
    c = len(config_runtime)
    max_configs = min(max_configs, c) if max_configs > 0 else c
    third = max_configs // 3

    sorted_indices = np.argsort(config_runtime)

    keep_indices = np.concatenate(
        [
            sorted_indices[:third],  # Good configs.
            sorted_indices[-third:],  # Bad configs.
            np.random.choice(
                sorted_indices[third:-third],
                max_configs - 2 * third,
            ),
        ]
    )

    return config_runtime[keep_indices], keep_indices