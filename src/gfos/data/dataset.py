from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LazyLayoutDataset(Dataset):
    """Load data when getitem."""

    def __init__(
        self, files: list[str], max_configs: int = -1, permute: bool = True
    ):
        self.max_configs = max_configs
        self.files = files
        self.permute = permute

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

        return dict(
            model_id=model_id,
            node_feat=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            node_config_feat=node_config_feat,
            node_config_ids=node_config_ids,
            config_runtime=config_runtime,
        )


class LayoutDataset(Dataset):
    """Load all data in advance."""

    def __init__(
        self,
        files: list[str],
        max_configs: int = -1,
        num_configs: int = -1,
        permute: bool = True,
    ):
        self.max_configs = max_configs
        self.num_configs = num_configs
        self.files = files
        self.permute = permute

        self.data = []
        for file in tqdm(self.files, desc="Loading data"):
            record = dict(np.load(file))
            model_id = file.split("\\")[-1].split(".")[0]
            record["model_id"] = model_id
            runtime = record["config_runtime"]
            record["config_runtime"] = (runtime - runtime.min()) / (
                runtime.max() - runtime.min()
            )

            c = len(record["config_runtime"])
            max_configs = self.max_configs if self.max_configs > 0 else c
            if self.permute:
                config_indices = torch.randperm(c)[:max_configs]
            else:
                config_indices = torch.arange(max_configs)

            record["config_runtime"] = record["config_runtime"][config_indices]
            record["node_config_feat"] = record["node_config_feat"][
                config_indices
            ]

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
        config_indices = torch.randperm(config_runtime.size(0))[:num_configs]
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

        return dict(
            model_id=model_id,
            node_feat=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            node_config_feat=node_config_feat,
            node_config_ids=node_config_ids,
            config_runtime=config_runtime,
        )
