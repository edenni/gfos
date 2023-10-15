import logging
import os
import datetime

import wandb
import torch
from torch import optim
from hydra.utils import instantiate
import numpy as np
from ..data.dataset import LayoutDataset, Normalizer
from ..data.utils import load_layout
from ..loss import MultiElementRankLoss
from ..model.gnn import LayoutModel
from .base import Pipeline
from ..utils.logging import flatten_dict
from tqdm import tqdm
from ..metrics import kendall, topk_error

logger = logging.getLogger(__name__)


class LayoutPipeline(Pipeline):
    pipeline_name = "layout"

    def create_dataset(self):
        # Read configs
        layout_dir = self.cfg.paths.layout_dir
        source = self.cfg.dataset.source
        search = self.cfg.dataset.search
        max_configs = self.cfg.dataset.max_configs
        num_configs = self.cfg.dataset.num_configs
        config_edges = self.cfg.dataset.config_edges
        normalizer_path = self.cfg.dataset.normalizer_path

        # Validate configs
        assert source in ("xla", "nlp"), f"Unknown source {source}"
        assert search in ("default", "random"), f"Unknown search {search}"
        assert config_edges in (
            "simple",
            "full_connect",
        ), f"Unknown config_edges {config_edges}"

        # Check if layout directory exists
        if not os.path.exists(layout_dir):
            raise FileNotFoundError(f"Layout directory {layout_dir} does not exist")

        # Check if normalizer exists
        if normalizer_path and not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Normalizer {normalizer_path} does not exist")

        # Load layout files
        logger.info(f"Loading layout from {layout_dir}")
        layout_files = load_layout(
            base_dir=layout_dir, model_type=source, compile_type=search
        )

        # Load normalizer
        normalizer = Normalizer.from_json(normalizer_path, source=source, search=search)

        # Create training and validation dataset
        self.train_dataset = LayoutDataset(
            files=layout_files["train"],
            max_configs=max_configs,
            num_configs=num_configs,
            config_edges=config_edges,
            normalizer=normalizer,
        )

        self.valid_dataset = LayoutDataset(
            files=layout_files["valid"],
            config_edges=config_edges,
            normalizer=normalizer,
        )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_model(self):
        """Setup model, loss function, optimizer, and scheduler"""
        # Create model
        node_feat_dim = self.train_dataset[0]["node_feat"].shape[-1]
        node_config_dim = self.train_dataset[0]["node_config_feat"].shape[-1]
        self.model = instantiate(
            self.cfg.model,
            node_feat_dim=node_feat_dim,
            node_config_dim=node_config_dim,
        ).to(self.device)

        # Loss function
        self.criterion = instantiate(self.cfg.loss)

        # Optimizer
        lr = self.cfg.optimizer.lr
        self.optimizer = instantiate(
            self.cfg.optimizer,
            [
                {
                    "name": "lr_embed",
                    "params": self.model.embedding.parameters(),
                    "lr": lr / 10,
                },
                {
                    "name": "lr_model_gnn",
                    "params": self.model.model_gnn.parameters(),
                    "lr": lr / 10,
                },
                {
                    "name": "lr_config_prj",
                    "params": self.model.config_prj.parameters(),
                    "lr": lr / 10,
                },
                {
                    "name": "lr_config_mp",
                    "params": self.model.config_mp.parameters(),
                    "lr": lr / 10,
                },
                {
                    "name": "lr_config_gnn",
                    "params": self.model.config_gnn.parameters(),
                    "lr": lr,
                },
                {
                    "name": "lr_dense",
                    "params": self.model.dense.parameters(),
                    "lr": lr,
                },
            ],
        )

        self.scheduler = instantiate(
            self.cfg.scheduler,
            optimizer=self.optimizer,
        )

    def train(self):
        self._setup_model()

        use_logger: bool = self.cfg.get("logger") is not None
        if use_logger:
            run = wandb.init(
                project=self.cfg.logger.project,
                dir=self.cfg.logger.dir,
                name=self.cfg.logger.name,
                config=flatten_dict(self.cfg),
                tags=self.cfg.logger.tags,
            )
            run.watch(self.model, log="all")

            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"../../logs/{self.cfg.logger.name}/{time_str}"
            os.makedirs(log_dir, exist_ok=True)

        device = self.device
        num_epochs = self.cfg.trainer.num_epochs
        num_val_epochs = self.cfg.trainer.num_val_epochs
        infer_bs = self.cfg.trainer.infer_bs
        accum_iter = self.cfg.trainer.accum_iter
        grad_clip = self.cfg.trainer.grad_clip

        best_score = -1
        loss_mean = 0

        # scaler = GradScaler()
        for epoch in range(num_epochs):
            # Shuffle the training dataset
            permutation = np.random.permutation(len(self.train_dataset))

            # Training phase
            self.model.train()
            pbar = tqdm(permutation, leave=False, desc=f"Epoch: {epoch}")

            for i in pbar:
                record = self.train_dataset[i]
                node_feat = record["node_feat"]
                node_opcode = record["node_opcode"]
                edge_index = record["edge_index"]
                node_config_feat = record["node_config_feat"]
                node_config_ids = record["node_config_ids"]
                config_runtime = record["config_runtime"]
                config_edge_index = record["config_edge_index"]

                (
                    node_feat,
                    node_opcode,
                    edge_index,
                    node_config_feat,
                    node_config_ids,
                    config_runtime,
                    config_edge_index,
                ) = (
                    node_feat.to(device),
                    node_opcode.to(device),
                    edge_index.to(device),
                    node_config_feat.to(device),
                    node_config_ids.to(device),
                    config_runtime.to(device),
                    config_edge_index.to(device),
                )

                # with autocast():
                out = self.model(
                    node_feat,
                    node_opcode,
                    edge_index,
                    node_config_feat,
                    node_config_ids,
                    config_edge_index,
                )

                loss = self.criterion(out, config_runtime)
                loss = loss / accum_iter
                loss_mean += loss.item()
                loss.backward()
                # scaler.scale(loss).backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                pbar.set_postfix_str(f"loss: {loss_mean:.4f}")

                if ((i + 1) % self.cfg.trainer.accum_iter == 0) or (
                    i + 1 == len(self.train_dataset)
                ):
                    # scaler.step(optimizer)
                    # scaler.update()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if use_logger:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "train/lr_slow": self.optimizer.param_groups[0]["lr"],
                                "train/lr_fast": self.optimizer.param_groups[-1]["lr"],
                                "train/loss": loss_mean,
                            }
                        )
                    loss_mean = 0

            pbar.close()

            if (epoch + 1) % num_val_epochs != 0 and epoch != num_epochs - 1:
                continue

            self.model.eval()

            # Validation phase
            # Scores placeholder
            kendalltau_scores = []
            raw_kendalltau_scores = []
            top500_scores = []
            top100_scores = []

            pbar = tqdm(self.valid_dataset, desc=f"Epoch: {epoch}", leave=True)
            with torch.no_grad():
                for record in pbar:
                    node_feat = record["node_feat"]
                    node_opcode = record["node_opcode"]
                    edge_index = record["edge_index"]
                    node_config_feat = record["node_config_feat"]
                    node_config_ids = record["node_config_ids"]
                    config_runtime = record["config_runtime"]
                    config_edge_index = record["config_edge_index"]

                    (
                        node_feat,
                        node_opcode,
                        edge_index,
                        node_config_feat,
                        node_config_ids,
                        config_edge_index,
                    ) = (
                        node_feat.to(device),
                        node_opcode.to(device),
                        edge_index.to(device),
                        node_config_feat.to(device),
                        node_config_ids.to(device),
                        config_edge_index.to(device),
                    )
                    config_runtime = config_runtime.numpy()
                    num_configs = config_runtime.shape[-1]
                    outs = []

                    for i in range(0, num_configs, infer_bs):
                        end_i = min(i + infer_bs, num_configs)
                        out: torch.Tensor = self.model(
                            node_feat,
                            node_opcode,
                            edge_index,
                            node_config_feat[i:end_i],
                            node_config_ids,
                            config_edge_index,
                        )
                        outs.append(out.detach().cpu())

                    outs = torch.concat(outs).numpy()

                    kendalltau_scores.append(
                        kendall(np.argsort(outs), np.argsort(config_runtime))
                    )
                    raw_kendalltau_scores.append(kendall(outs, config_runtime))
                    top100_scores.append(topk_error(outs, config_runtime, top_k=100))
                    top500_scores.append(topk_error(outs, config_runtime, top_k=500))

            kendalltau_mean = np.mean(kendalltau_scores)
            raw_kendalltau_mean = np.mean(raw_kendalltau_scores)
            top100_mean = np.mean(top100_scores)
            top500_mean = np.mean(top500_scores)

            self.scheduler.step(kendalltau_mean)

            if use_logger:
                wandb.log(
                    {
                        "val/kendalltau": kendalltau_mean,
                        "val/raw_kendalltau": raw_kendalltau_mean,
                        "val/top100_error": top100_mean,
                        "val/top500_error": top500_mean,
                    }
                )

            print(
                f"epoch {epoch}, kendall = {raw_kendalltau_mean:.4f}, "
                f"top500 = {top500_mean:.4f}"
            )

            # Update best scores and save the model if the mean score improves
            if raw_kendalltau_mean > best_score:
                best_score = raw_kendalltau_mean
                print(f"Best score updated: {best_score:.4f}")
                if use_logger:
                    filename = f"{epoch}_{best_score:.4f}.pth"
                    path = os.path.join(wandb.run.dir, filename)
                    torch.save(
                        self.model.state_dict(),
                        path,
                    )

        if use_logger:
            run.finish()
