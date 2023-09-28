from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class LayoutDataset(Dataset):
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

        target = torch.tensor(record["config_runtime"], dtype=torch.float)
        target = (target - target.mean()) / (target.std() + 1e-7)

        if self.permute:
            config_indices = torch.randperm(target.size(0))[:max_configs]
        else:
            config_indices = torch.arange(max_configs)
        target = target[config_indices]

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
            target=target,
        )
