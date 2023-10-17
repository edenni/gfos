import logging
import os

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from ..data.dataset import LayoutDataset, Normalizer
from ..data.utils import load_layout
from ..metrics import LayoutMetrics
from .base import Pipeline

logger = logging.getLogger(__name__)


class LayoutPipeline(Pipeline):
    pipeline_name = "layout"

    def create_dataset(self, train: bool = True, test: bool = False):
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
            raise FileNotFoundError(
                f"Layout directory {layout_dir} does not exist"
            )

        # Check if normalizer exists
        if normalizer_path and not os.path.exists(normalizer_path):
            raise FileNotFoundError(
                f"Normalizer {normalizer_path} does not exist"
            )

        # Load layout files
        logger.info(f"Loading {source} {search} layouts from {layout_dir}")
        layout_files = load_layout(
            base_dir=layout_dir, model_type=source, compile_type=search
        )

        # Load normalizer
        normalizer = Normalizer.from_json(
            normalizer_path, source=source, search=search
        )

        # Create training and validation dataset
        if train:
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
        if test:
            self.test_dataset = LayoutDataset(
                files=layout_files["test"],
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

        self.cfg.model.node_feat_dim = node_feat_dim
        self.cfg.model.node_config_dim = node_config_dim

        logger.info(f"Node feature dim: {node_feat_dim}")
        logger.info(f"Node config feature dim: {node_config_dim}")

        self.model = instantiate(
            self.cfg.model,
        ).to(self.device)

        # Loss function
        self.criterion = instantiate(self.cfg.loss)

        # Optimizer
        # TODO: add support for multiple learning rate
        self.optimizer = instantiate(
            self.cfg.optimizer,
            self.model.parameters(),
        )

        self.scheduler = instantiate(
            self.cfg.scheduler,
            self.optimizer,
        )

    def train(self):
        self.create_dataset(test="test" in self.cfg.tasks)
        self._setup_model()

        use_logger: bool = self.cfg.get("logger") is not None
        if use_logger:
            run = wandb.init(
                project=self.cfg.logger.project,
                dir=self.cfg.paths.output_dir,
                name=self.cfg.logger.name,
                config=OmegaConf.to_container(
                    self.cfg, resolve=True, throw_on_missing=True
                ),
                tags=self.cfg.logger.tags,
            )
            run.watch(self.model, log="all")
            run.log_code("./src/gfos")

        # Training configs
        device = self.device
        num_epochs = self.cfg.trainer.num_epochs
        num_val_epochs = self.cfg.trainer.num_val_epochs
        infer_bs = self.cfg.trainer.infer_bs
        accum_iter = self.cfg.trainer.accum_iter
        grad_clip = self.cfg.trainer.grad_clip
        early_stopping = self.cfg.trainer.early_stopping

        best_score = -1
        loss_mean = 0
        not_improved = 0

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

                pbar.set_postfix_str(f"loss: {loss:.4f}")

                if ((i + 1) % self.cfg.trainer.accum_iter == 0) or (
                    i + 1 == len(self.train_dataset)
                ):
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if use_logger:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "train/lr_slow": self.optimizer.param_groups[
                                    0
                                ]["lr"],
                                "train/lr_fast": self.optimizer.param_groups[
                                    -1
                                ]["lr"],
                                "train/loss": loss_mean,
                            }
                        )
                    loss_mean = 0
            pbar.close()

            # Validation phase
            if (epoch + 1) % num_val_epochs != 0 and epoch != num_epochs - 1:
                continue

            self.model.eval()
            metrics = LayoutMetrics()

            for record in tqdm(
                self.valid_dataset, desc=f"Valid epoch: {epoch}", leave=False
            ):
                config_runtime: torch.Tensor = record["config_runtime"]
                outs: torch.Tensor = self._predict_one(
                    record, infer_bs, device
                )
                metrics.add(
                    record["model_id"], outs.numpy(), config_runtime.numpy()
                )

            prefix = "val/"
            scores = metrics.compute_scores(prefix=prefix)

            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                kendall = scores[f"{prefix}index_kendall"]
                self.scheduler.step(kendall)

            if use_logger:
                wandb.log(scores)

            # Update best scores and save the model
            if kendall > best_score:
                best_score = kendall
                print(f"Best score updated: {best_score:.4f}")
                if use_logger:
                    self.best_model_path = self._save_model(epoch, kendall)
                not_improved = 0
            else:
                not_improved += 1
                if early_stopping > 0 and not_improved == early_stopping:
                    break

        if use_logger:
            self._save_model(epoch, kendall, suffix="_last")

        if "test" not in self.cfg.tasks:
            wandb.finish()

    def _save_model(self, epoch: int, score: float, suffix: str = "") -> str:
        filename = f"{epoch}_{score:.4f}{suffix}.pth"
        path = os.path.join(wandb.run.dir, filename)
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(
            state,
            path,
        )
        return path

    def _predict_one(
        self, record: dict, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
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
            c = len(config_runtime)
            outs = []

            for i in range(0, c, batch_size):
                end_i = min(i + batch_size, c)
                out: torch.Tensor = self.model(
                    node_feat,
                    node_opcode,
                    edge_index,
                    node_config_feat[i:end_i],
                    node_config_ids,
                    config_edge_index,
                )
                outs.append(out.detach().cpu())
            return torch.concat(outs)

    def test(self):
        if self.best_model_path is not None:
            state_dict = torch.load(self.best_model_path)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
        elif self.cfg.trainer.load_path is not None:
            raise NotImplementedError("Cannot load model from path yet")
        else:
            raise ValueError("No model path found")

        if getattr(self, "test_dataset", None) is None:
            self.create_dataset(train=False, test=True)

        device = self.device
        infer_bs = self.cfg.trainer.infer_bs
        self.model.to(device).eval()

        results = {}
        for record in tqdm(self.test_dataset, desc="Testing"):
            model_id = record["model_id"]
            outs = self._predict_one(record, infer_bs, device)
            pred_idx = np.argsort(outs.numpy())
            results[model_id] = pred_idx.tolist()

        # Write test results to file
        source = self.cfg.dataset.source
        search = self.cfg.dataset.search
        output_path = os.path.join(
            wandb.run.dir, f"{source}_{search}_" + wandb.run.id + ".csv"
        )

        logger.info(f"Writing results to {output_path}")
        with open(output_path, "w") as f:
            f.write("ID,TopConfigs\n")
            for k, v in results.items():
                model_id = f"layout:{source}:{search}:" + k
                values = ";".join([str(i) for i in v])
                f.write(f"{model_id},{values}\n")

        wandb.finish()
