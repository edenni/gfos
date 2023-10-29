from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from gfos.data.graph import get_config_graph


@dataclass
class Normalizer:
    node_feat_mask: torch.Tensor
    node_feat_min: torch.Tensor
    node_feat_max: torch.Tensor
    node_config_feat_mask: torch.Tensor
    node_config_feat_min: torch.Tensor
    node_config_feat_max: torch.Tensor

    def normalize_node_feat(self, node_feat: torch.Tensor) -> torch.Tensor:
        assert node_feat.ndim == 2, "node_feat must be 2D"
        node_feat = node_feat[:, self.node_feat_mask]

        return (node_feat - self.node_feat_min) / (
            self.node_feat_max - self.node_feat_min
        )

    def normalize_node_config_feat(
        self, node_config_feat: torch.Tensor
    ) -> torch.Tensor:
        assert node_config_feat.ndim == 3, "node_config_feat must be 3D"
        node_config_feat = node_config_feat[:, :, self.node_config_feat_mask]
        return (node_config_feat - self.node_config_feat_min) / (
            self.node_config_feat_max - self.node_config_feat_min
        )

    @classmethod
    def from_dict(
        cls,
        configs: dict,
        source: Literal["xla", "nlp"],
        search: Literal["default", "random"],
    ) -> "Normalizer":
        try:
            data = configs[source][search]
        except KeyError:
            raise KeyError(
                f"Invalid source or search: source={source}, search={search}"
            )
        else:
            node_feat_mask = torch.tensor(data["node_feat_mask"], dtype=torch.bool)
            node_feat_min = torch.tensor(data["node_feat_min"], dtype=torch.float)[
                node_feat_mask
            ]
            node_feat_max = torch.tensor(data["node_feat_max"], dtype=torch.float)[
                node_feat_mask
            ]
            node_config_feat_mask = torch.tensor(
                data["node_config_feat_mask"], dtype=torch.bool
            )
            node_config_feat_min = torch.tensor(
                data["node_config_feat_min"], dtype=torch.float
            )[node_config_feat_mask]
            node_config_feat_max = torch.tensor(
                data["node_config_feat_max"], dtype=torch.float
            )[node_config_feat_mask]

            return Normalizer(
                node_feat_mask=node_feat_mask,
                node_feat_min=node_feat_min,
                node_feat_max=node_feat_max,
                node_config_feat_mask=node_config_feat_mask,
                node_config_feat_min=node_config_feat_min,
                node_config_feat_max=node_config_feat_max,
            )

    @classmethod
    def from_json(cls, path, source, search):
        import json

        json_data = json.load(open(path))
        return Normalizer.from_dict(json_data, source, search)


class LayoutDataset(Dataset):
    """Load all data in advance."""

    def __init__(
        self,
        files: list[str],
        max_configs: int = -1,
        num_configs: int = -1,
        normalizer: Normalizer = None,
        bins: np.array = None,
        indices_dir: str = None,
        runtime_mean: float = None,
        runtime_std: float = None,
    ):
        self.max_configs = max_configs
        self.num_configs = num_configs
        self.files = files
        self.normalizer = normalizer

        if indices_dir is not None:
            if not Path(indices_dir).exists():
                raise FileNotFoundError(
                    f"Fold index dir <{indices_dir}> " "specified but does not exist"
                )
            indices_dir = Path(indices_dir)
            target_models = set([f.stem for f in indices_dir.glob("*.npy")])
            self.files = [f for f in files if Path(f).stem in target_models]
        else:
            self.files = files

        self.data = []
        pbar = tqdm(self.files, desc="Loading data")

        for file in pbar:
            record = dict(np.load(file))
            model_id = Path(file).stem
            pbar.set_postfix_str(model_id)

            record["model_id"] = model_id
            runtime = record["config_runtime"]

            if bins is not None:
                cls_lables = np.digitize(runtime, bins)

            if runtime_mean is None or runtime_std is None:
                runtime = (runtime - runtime.mean()) / runtime.std()
            else:
                runtime = (runtime - runtime_mean) / runtime_std

            if indices_dir is not None:
                indices_file = Path(indices_dir) / f"{model_id}.npy"
                if indices_file.exists():
                    config_indices = np.load(indices_file)
                    runtime_sampled = runtime[config_indices]
                else:
                    raise FileNotFoundError(f"{indices_file} does not exist")
            else:
                if self.max_configs > 0:
                    # sample `max_configs` with order
                    # [good_configs, bad_configs, random_configs]
                    runtime_sampled, config_indices = sample_configs(
                        runtime, max_configs
                    )
                else:
                    # use all configs
                    runtime_sampled = runtime
                    config_indices = torch.arange(len(runtime))

            record["config_runtime"] = runtime_sampled
            record["node_config_feat"] = record["node_config_feat"][config_indices]
            record["argsort_runtime"] = np.argsort(runtime_sampled)

            if bins is not None:
                record["cls_label"] = cls_lables[config_indices]

            # create graph for configurable nodes
            config_edge_index, edge_weight, paths = get_config_graph(
                record["edge_index"],
                record["node_config_ids"],
            )
            record["config_edge_weight"] = torch.tensor(edge_weight, dtype=torch.float)
            record["config_edge_path"] = paths

            config_edge_index = torch.tensor(
                np.swapaxes(config_edge_index, 0, 1),
                dtype=torch.long,
            )
            record["config_edge_index"] = config_edge_index

            record["config_runtime"] = torch.tensor(
                record["config_runtime"], dtype=torch.float
            )
            record["argsort_runtime"] = torch.tensor(
                record["argsort_runtime"], dtype=torch.long
            )
            record["node_feat"] = torch.tensor(record["node_feat"], dtype=torch.float)
            record["node_opcode"] = torch.tensor(
                record["node_opcode"], dtype=torch.long
            )
            record["edge_index"] = torch.tensor(
                np.swapaxes(record["edge_index"], 0, 1), dtype=torch.long
            )
            record["node_config_feat"] = torch.tensor(
                record["node_config_feat"], dtype=torch.float
            )
            record["node_config_ids"] = torch.tensor(
                record["node_config_ids"], dtype=torch.long
            )

            if self.normalizer is not None:
                record["node_feat"] = self.normalizer.normalize_node_feat(
                    record["node_feat"]
                )
                record["node_config_feat"] = self.normalizer.normalize_node_config_feat(
                    record["node_config_feat"]
                )

            if bins is not None:
                record["cls_label"] = torch.tensor(
                    record["cls_label"], dtype=torch.long
                )

            self.data.append(record)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> dict[str, Any]:
        record = self.data[idx]

        config_runtime = record["config_runtime"]
        node_feat = record["node_feat"]
        node_opcode = record["node_opcode"]
        edge_index = record["edge_index"]
        node_config_feat = record["node_config_feat"]
        node_config_ids = record["node_config_ids"]
        argsort_runtime = record["argsort_runtime"]
        config_edge_index = record["config_edge_index"]
        config_edge_weight = record["config_edge_weight"]
        config_edge_path = record["config_edge_path"]

        c = len(config_runtime)

        if self.num_configs > 0:
            num_configs = min(self.num_configs, c)
        elif self.max_configs > 0:
            num_configs = min(self.max_configs, c)
        else:
            num_configs = c

        # Sample
        if self.max_configs > 0 or self.num_configs > 0:
            # config_indices = torch.randperm(config_runtime.size(0))[
            #     :num_configs
            # ]
            idx = torch.topk(
                # Sample wrt GumbulSoftmax([NumConfs, NumConfs-1, ..., 1])
                (c - torch.arange(c)) / c - torch.log(-torch.log(torch.rand(c))),
                num_configs,
            )[1]
            config_indices = argsort_runtime[idx]
        else:
            config_indices = torch.arange(num_configs)
        config_runtime = config_runtime[config_indices]

        model_id = record["model_id"]

        node_config_feat = node_config_feat[config_indices]

        sample = dict(
            model_id=model_id,
            x=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            node_config_feat=node_config_feat,
            node_config_ids=node_config_ids,
            config_runtime=config_runtime,
            config_edge_index=config_edge_index,
            config_edge_weight=config_edge_weight,
            config_edge_path=config_edge_path,
        )

        if "cls_label" in record:
            sample["cls_label"] = record["cls_label"][config_indices]

        return Data(**sample)


def sample_configs(config_runtime: np.array, max_configs: int) -> (np.array, np.array):
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
