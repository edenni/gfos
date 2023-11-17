import copy
import logging
import os
import pickle

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data.constants import CONFIG_RUNTIME_MEAN_STD
from ..data.dataset import LayoutData, LayoutDataset, Normalizer
from ..data.utils import load_layout
from ..metrics import LayoutMetrics
from ..model.history import History
from ..model.utils import get_adj
from .base import Pipeline

logger = logging.getLogger(__name__)


class LayoutPipeline(Pipeline):
    pipeline_name = "layout"

    def create_dataset(
        self, train: bool = True, valid: bool = True, test: bool = False
    ):
        if (
            getattr(self, "train_dataset", None) is not None
            and getattr(self, "valid_dataset", None) is not None
        ):
            return
        # Read configs
        layout_dir = self.cfg.paths.layout_dir
        indices_dir = self.cfg.paths.indices_dir
        source = self.cfg.dataset.source
        search = self.cfg.dataset.search
        max_configs = self.cfg.dataset.max_configs
        num_configs = self.cfg.dataset.num_configs
        normalizer_path = self.cfg.dataset.normalizer_path
        fold = self.cfg.dataset.fold

        # Validate configs
        assert source in ("xla", "nlp"), f"Unknown source {source}"
        assert search in ("default", "random"), f"Unknown search {search}"

        # Check if layout directory exists
        if not os.path.exists(layout_dir):
            raise FileNotFoundError(f"Layout directory {layout_dir} does not exist")

        # Check if normalizer exists
        if normalizer_path and not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Normalizer {normalizer_path} does not exist")

        # Load layout files
        logger.info(f"Loading {source} {search} layouts from {layout_dir}")
        layout_files = load_layout(
            base_dir=layout_dir, model_type=source, compile_type=search
        )

        # Load normalizer
        normalizer = Normalizer.from_json(normalizer_path, source=source, search=search)
        runtime_mean = CONFIG_RUNTIME_MEAN_STD[source][search]["mean"]
        runtime_std = CONFIG_RUNTIME_MEAN_STD[source][search]["std"]

        if fold is None or fold < 0:
            train_files = layout_files["train"]
            valid_files = layout_files["valid"]
            train_indices_dir = valid_indices_dir = None
        elif fold >= 0:
            fold_dir = f"{indices_dir}/{source}_{search}/{fold}"
            logger.info(f"Using fold {fold}")
            logger.info(f"Loading indices from {fold_dir}")

            if indices_dir is None or not os.path.exists(fold_dir):
                raise FileNotFoundError(f"Indices directory {fold_dir} not found")

            # Pass all files to Dataset and filter files inside __init__
            train_files = layout_files["train"] + layout_files["valid"]
            valid_files = layout_files["train"] + layout_files["valid"]
            train_indices_dir = f"{fold_dir}/train"
            valid_indices_dir = f"{fold_dir}/valid"

        if train:
            if os.path.exists(".train_dataset.cache"):
                self.train_dataset = pickle.load(open(".train_dataset.cache", "rb"))
                self.valid_dataset = pickle.load(open(".valid_dataset.cache", "rb"))
            else:
                train_files = layout_files["train"] + layout_files["valid"]
                self.train_dataset = LayoutDataset(
                    files=train_files,
                    max_configs=max_configs,
                    num_configs=num_configs,
                    normalizer=normalizer,
                    indices_dir=train_indices_dir,
                    runtime_mean=runtime_mean,
                    runtime_std=runtime_std,
                )
                another_search = "random" if search == "default" else "default"
                valid_files = load_layout(
                    base_dir=layout_dir, model_type=source, compile_type=another_search
                )
                normalizer = Normalizer.from_json(
                    normalizer_path, source=source, search=another_search
                )
                runtime_mean = CONFIG_RUNTIME_MEAN_STD[source][another_search]["mean"]
                runtime_std = CONFIG_RUNTIME_MEAN_STD[source][another_search]["std"]
                self.valid_dataset = LayoutDataset(
                    files=valid_files["valid"],
                    normalizer=normalizer,
                    indices_dir=valid_indices_dir,
                    runtime_mean=runtime_mean,
                    runtime_std=runtime_std,
                )
                # pickle.dump(self.train_dataset, open(".train_dataset.cache", "wb"))
                # pickle.dump(self.valid_dataset, open(".valid_dataset.cache", "wb"))
        if test:
            self.test_dataset = LayoutDataset(
                files=layout_files["test"],
                normalizer=normalizer,
            )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_model(self, load_path: str = None):
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

        if load_path is not None:
            logger.info(f"Loading model from {self.cfg.trainer.load_path}")
            state_dict = torch.load(self.cfg.trainer.load_path)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
            # for layer in self.model.dense.children():
            #     if hasattr(layer, "reset_parameters"):
            #         logger.warning(f"Resetting parameters for {layer}")
            #         layer.reset_parameters()
            self.model.dense = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 1),
            ).to(self.device)

        # Loss function
        self.criterion = instantiate(self.cfg.loss)

        # Optimizer
        self.optimizer = instantiate(
            self.cfg.optimizer,
            [
                {
                    "name": "lr_embed",
                    "params": self.model.embedding.parameters(),
                    "lr": self.cfg.optimizer.lr / 2,
                },
                {
                    "name": "lr_node_gnn",
                    "params": self.model.node_gnn.parameters(),
                    "lr": self.cfg.optimizer.lr / 2,
                },
                {
                    "name": "lr_config_prj",
                    "params": self.model.config_prj.parameters(),
                    "lr": self.cfg.optimizer.lr / 2,
                },
                {
                    "name": "lr_config_neighbor_gnn",
                    "params": self.model.config_neighbor_gnn.parameters(),
                    "lr": self.cfg.optimizer.lr / 2,
                },
                {
                    "name": "lr_config_gnn",
                    "params": self.model.config_gnn.parameters(),
                    "lr": self.cfg.optimizer.lr / 2,
                },
                {
                    "name": "lr_dense",
                    "params": self.model.dense.parameters(),
                    "lr": self.cfg.optimizer.lr,
                },
            ],
        )

        self.scheduler = instantiate(
            self.cfg.scheduler,
            self.optimizer,
        )

    def _train_one(
        self,
        record: dict,
        device: torch.device,
        accum_iter: int,
    ):
        node_feat = record["node_feat"]
        node_opcode = record["node_opcode"]
        edge_index = record["edge_index"]
        node_config_feat = record["node_config_feat"]
        node_config_ids = record["node_config_ids"]
        config_runtime = record["config_runtime"]
        config_edge_index = record["config_edge_index"]
        node_config_feat_batch = record.node_config_feat_batch
        batch_size = len(record.model_id)

        (
            node_feat,
            node_opcode,
            edge_index,
            node_config_feat,
            node_config_ids,
            config_edge_index,
            config_runtime,
            node_config_feat_batch,
        ) = (
            node_feat.to(device),
            node_opcode.to(device),
            edge_index.to(device),
            node_config_feat.to(device),
            node_config_ids.to(device),
            config_edge_index.to(device),
            config_runtime.to(device),
            node_config_feat_batch.to(device),
        )

        out = self.model(
            node_feat,
            node_opcode,
            edge_index,
            node_config_feat,
            node_config_ids,
            config_edge_index,
            node_config_feat_batch,
            batch_size,
        )

        out = out.reshape(self.cfg.dataset.num_configs, -1).T.contiguous()
        config_runtime = config_runtime.reshape(-1, self.cfg.dataset.num_configs)

        loss = self.criterion(out, config_runtime)
        loss = loss / accum_iter
        loss.backward()

        return loss

    def train(self):
        self.create_dataset(test="test" in self.cfg.tasks)
        self._setup_model(self.cfg.trainer.load_path)

        use_logger: bool = self.cfg.get("logger") is not None
        if use_logger:
            run = wandb.init(
                project=self.cfg.logger.project,
                dir=self.cfg.paths.output_dir,
                group=self.cfg.logger.group,
                name=self.cfg.logger.name,
                tags=self.cfg.logger.tags,
            )

            if self.cfg.get("sweep") is not None:
                self.cfg.model.num_node_layers = wandb.config.num_node_layers
                self.cfg.model.num_config_layers = wandb.config.num_config_layers
                self.cfg.model.num_config_neighbor_layers = (
                    wandb.config.num_config_neighbor_layers
                )
                self.cfg.model.node_dim = wandb.config.node_dim
                self.cfg.model.config_dim = wandb.config.config_dim
                self.cfg.model.config_neighbor_dim = wandb.config.config_neighbor_dim

                self.cfg.model.node_dropout_between_layers = (
                    wandb.config.node_dropout_between_layers
                )
                self.cfg.model.config_dropout_between_layers = (
                    wandb.config.config_dropout_between_layers
                )
                self.cfg.model.config_neighbor_dropout_between_layers = (
                    wandb.config.config_neighbor_dropout_between_layers
                )
                self.cfg.model.head_dim = wandb.config.head_dim
                self.cfg.model.dropout = wandb.config.dropout
                self.cfg.model.jk_mode = wandb.config.jk_mode

                self.cfg.optimizer.weight_decay = wandb.config.weight_decay

            wandb.config.update(
                OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
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

        # Catch keyboard interrupt, infer on test set, and exit
        try:
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.cfg.trainer.batch_size,
                    shuffle=True,
                    follow_batch=["node_config_feat", "node_feat"],
                )
                # Shuffle the training dataset
                # permutation = np.random.permutation(len(self.train_dataset))
                pbar = tqdm(loader, leave=False, desc=f"Epoch: {epoch}")

                for i, batch in enumerate(pbar):
                    # record = self.train_dataset[i]
                    loss = self._train_one(
                        batch,
                        device,
                        accum_iter,
                    )
                    loss_mean += loss.item()

                    pbar.set_postfix_str(f"loss: {loss:.4f}")

                    # Backward
                    if ((i + 1) % self.cfg.trainer.accum_iter == 0) or (
                        i + 1 == len(loader)
                    ):
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), grad_clip
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        log_params = {
                            "epoch": epoch,
                            "train/lr": self.optimizer.param_groups[-1]["lr"],
                            "train/loss": loss.item() * accum_iter,
                        }
                        if use_logger:
                            wandb.log(log_params)
                        loss_mean = 0

                if not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
                pbar.close()

                # Validation phase
                if (epoch + 1) % num_val_epochs != 0 and epoch != num_epochs - 1:
                    continue

                self.model.eval()
                metrics = LayoutMetrics()
                val_outs = {}  # save the output for each model

                for record in tqdm(
                    self.valid_dataset,
                    desc=f"Valid epoch: {epoch}",
                    leave=False,
                ):
                    config_runtime: torch.Tensor = record["config_runtime"]
                    outs: torch.Tensor = self._predict_one(record, infer_bs, device)
                    metrics.add(
                        record["model_id"],
                        outs.numpy(),
                        config_runtime.numpy(),
                    )

                    val_outs[record["model_id"]] = outs.numpy()

                prefix = "val/"
                scores = metrics.compute_scores(prefix=prefix)

                kendall = scores[f"{prefix}index_kendall"]
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(kendall)

                if use_logger:
                    wandb.log(scores)

                # Update best scores and save the model
                if kendall > best_score:
                    best_score = kendall
                    print("Best score updated " f"at epoch {epoch}: {best_score:.4f}")
                    if use_logger:
                        self.best_model_path = self._save_model(epoch, kendall)
                    not_improved = 0

                    output_path = os.path.join(wandb.run.dir, f"val_outs_{epoch}.plk")
                    with open(output_path, "wb") as f:
                        pickle.dump(val_outs, f)
                else:
                    not_improved += 1
                    if early_stopping > 0 and not_improved == early_stopping:
                        break

        except KeyboardInterrupt:
            pass

        if use_logger:
            self._save_model(epoch, kendall, suffix="_last")

        if "test" not in self.cfg.tasks:
            wandb.finish()

        return best_score

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
            logger.info(f"Loading best model from {self.best_model_path}")
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
        logits = {}
        for record in tqdm(self.test_dataset, desc="Testing"):
            model_id = record["model_id"]
            outs = self._predict_one(record, infer_bs, device)
            pred_idx = np.argsort(outs.numpy())
            results[model_id] = pred_idx.tolist()
            logits[model_id] = outs.numpy()

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

        with open(output_path.replace(".csv", "_logits.plk"), "wb") as f:
            pickle.dump(logits, f)

        wandb.finish()

    def tune(self):
        sweep_id = wandb.sweep(
            sweep=OmegaConf.to_container(self.cfg.sweep),
            project=self.cfg.logger.project,
            entity="edenn0",
        )
        wandb.agent(
            sweep_id=sweep_id,
            function=self.train,
            entity="edenn0",
            project=self.cfg.logger.project,
            count=100,
        )

    def gst(self):
        self.create_dataset(test="test" in self.cfg.tasks)
        self._setup_model()

        if self.cfg.trainer.load_path is not None:
            logger.info(f"Loading model from {self.cfg.trainer.load_path}")
            state_dict = torch.load(self.cfg.trainer.load_path)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
            # for layer in self.model.dense.children():
            #     if hasattr(layer, "reset_parameters"):
            #         logger.warning(f"Resetting parameters for {layer}")
            #         layer.reset_parameters()
            self.model.to(self.device).train()

        use_logger: bool = self.cfg.get("logger") is not None
        if use_logger:
            run = wandb.init(
                project=self.cfg.logger.project,
                dir=self.cfg.paths.output_dir,
                config=OmegaConf.to_container(
                    self.cfg, resolve=True, throw_on_missing=True
                ),
                group=self.cfg.logger.group,
                name=self.cfg.logger.name,
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

        emb_table = History(500000000, 1)

        # Catch keyboard interrupt, infer on test set, and exit
        try:
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.cfg.trainer.batch_size,
                    shuffle=True,
                    follow_batch=["node_config_feat", "node_feat"],
                )
                # Shuffle the training dataset
                # permutation = np.random.permutation(len(self.train_dataset))
                pbar = tqdm(loader, leave=False, desc=f"Epoch: {epoch}")

                for iter, batch in enumerate(loader):
                    batch = get_adj(batch)

                    batch_list = batch.to_data_list()
                    batch_train_list = []
                    batch_other = []
                    batch_num_parts = []
                    segments_to_train = []
                    skipped_batch = []
                    for i in range(len(batch_list)):
                        num_parts = len(batch_list[i].partptr) - 1

                        segment_to_train = np.random.randint(num_parts)

                        batch_other_ = []
                        add_target = False
                        for j in range(num_parts):
                            start = int(batch_list[i].partptr.cpu().numpy()[j])
                            length = (
                                int(batch_list[i].partptr.cpu().numpy()[j + 1]) - start
                            )

                            cidx = torch.where(
                                (batch_list[i].node_config_ids >= start)
                                & (batch_list[i].node_config_ids < start + length)
                            )[0]
                            if len(cidx) == 0:
                                if j == segment_to_train:
                                    break
                                continue
                            cstart = cidx[0]
                            clength = len(cidx)

                            N, NC = (
                                batch_list[i].num_nodes,
                                batch_list[i].num_config_nodes,
                            )

                            data = copy.copy(batch_list[i])
                            del data.num_nodes
                            adj, data.adj = data.adj, None
                            cadj, data.cadj = data.cadj, None

                            adj = adj.narrow(0, start, length).narrow(1, start, length)
                            cadj = cadj.narrow(0, cstart, clength).narrow(
                                1, cstart, clength
                            )

                            for key, item in data:
                                if (
                                    isinstance(item, torch.Tensor) and item.size(0) == N
                                ):  # node_feat, node_opcode
                                    data[key] = item.narrow(0, start, length)
                                elif (
                                    isinstance(item, torch.Tensor)
                                    and item.ndim > 1
                                    and item.size(1) == NC
                                ):
                                    data[key] = item.narrow(
                                        1, cstart, clength
                                    )  # node_config_feat
                                elif (
                                    isinstance(item, torch.Tensor)
                                    and item.size(0) == NC
                                ):
                                    data[key] = (
                                        item.narrow(0, cstart, clength) - start
                                    )  # node_config_ids
                                else:
                                    data[key] = item

                            row, col, _ = adj.coo()
                            data.edge_index = torch.stack([row, col], dim=0)
                            row, col, _ = cadj.coo()
                            data.config_edge_index = torch.stack([row, col], dim=0)
                            if j == segment_to_train:
                                batch_train_list.append(
                                    LayoutData(
                                        node_feat=data.node_feat,
                                        node_opcode=data.node_opcode,
                                        edge_index=data.edge_index,
                                        node_config_feat=data.node_config_feat,
                                        node_config_ids=data.node_config_ids,
                                        config_runtime=data.config_runtime,
                                        config_edge_index=data.config_edge_index,
                                        num_config_nodes=len(data.node_config_ids),
                                        model_id=data.model_id,
                                    )
                                )
                                add_target = True
                            else:
                                batch_other_.append(
                                    emb_table.pull(
                                        batch_list[i].partition_idx.cpu()
                                        + data.config_indices * num_parts
                                        + j
                                    )
                                )
                        if len(batch_other_) > 0 and add_target:
                            batch_other_ = torch.mean(
                                torch.stack(batch_other_, dim=0), dim=0
                            )
                            batch_other.append(batch_other_)
                            batch_num_parts.extend(
                                [num_parts] * self.cfg.dataset.num_configs
                            )
                            segments_to_train.append(segment_to_train)
                        elif add_target:
                            batch_other.append(
                                torch.zeros_like(
                                    batch_train_list[-1].config_runtime
                                ).unsqueeze(1)
                            )
                            batch_num_parts.extend(
                                [num_parts] * self.cfg.dataset.num_configs
                            )
                            segments_to_train.append(segment_to_train)
                        else:
                            skipped_batch.append(i)

                    batch_seg = Batch.from_data_list(
                        batch_train_list, follow_batch=["node_config_feat", "node_feat"]
                    )

                    node_feat = batch_seg["node_feat"]
                    node_opcode = batch_seg["node_opcode"]
                    edge_index = batch_seg["edge_index"]
                    node_config_feat = batch_seg["node_config_feat"]
                    node_config_ids = batch_seg["node_config_ids"]
                    config_runtime = batch_seg["config_runtime"]
                    config_edge_index = batch_seg["config_edge_index"]
                    node_config_feat_batch = batch_seg["node_config_feat_batch"]
                    batch_size = len(batch_seg.model_id)

                    (
                        node_feat,
                        node_opcode,
                        edge_index,
                        node_config_feat,
                        node_config_ids,
                        config_edge_index,
                        config_runtime,
                        node_config_feat_batch,
                    ) = (
                        node_feat.to(device),
                        node_opcode.to(device),
                        edge_index.to(device),
                        node_config_feat.to(device),
                        node_config_ids.to(device),
                        config_edge_index.to(device),
                        config_runtime.to(device),
                        node_config_feat_batch.to(device),
                    )

                    out = self.model(
                        node_feat,
                        node_opcode,
                        edge_index,
                        node_config_feat,
                        node_config_ids,
                        config_edge_index,
                        node_config_feat_batch,
                        batch_size,
                    )

                    out = out.reshape(64, -1).T.contiguous().reshape(-1, 1)

                    binomial = torch.distributions.binomial.Binomial(probs=0.5)
                    if len(batch_other) > 0:
                        batch_other = torch.cat(batch_other, dim=0)
                        # mask = binomial.sample(
                        #     (batch_other.shape[0], batch_other.shape[1])
                        # )
                        mask = binomial.sample((batch_other.shape[0], 1)).to(
                            self.device
                        )
                        batch_other = batch_other.to(self.device)
                        batch_other_embed = batch_other * mask
                        # batch_other_embed = torch.zeros_like(out)
                        # part_cnt = 0
                        # for i, num_parts in enumerate(batch_num_parts):
                        #     for j in range(num_parts - 1):
                        #         batch_other_embed[i, :] += batch_other[part_cnt, :]
                        #         part_cnt += 1
                        batch_num_parts = torch.Tensor(batch_num_parts).to(self.device)
                        batch_num_parts = batch_num_parts.view(-1, 1)
                        multiplier_num = (batch_num_parts - 1) / 2 + 1
                        pred = out * multiplier_num + batch_other_embed
                    else:
                        pred = out

                    out = out.reshape(-1, self.cfg.dataset.num_configs)
                    config_runtime = config_runtime.reshape(
                        -1, self.cfg.dataset.num_configs
                    )

                    loss = self.criterion(pred, config_runtime)
                    loss = loss / accum_iter
                    loss.backward()
                    loss_mean += loss.item()

                    self.optimizer.step()
                    # Backward
                    if ((i + 1) % self.cfg.trainer.accum_iter == 0) or (
                        i + 1 == len(self.train_dataset)
                    ):
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), grad_clip
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        log_params = {
                            "epoch": epoch,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/loss": loss_mean,
                        }
                        if use_logger:
                            wandb.log(log_params)
                        loss_mean = 0

                    # Update embedding table
                    used_batch = [
                        batch_list[i]
                        for i in range(len(batch_list))
                        if i not in skipped_batch
                    ]
                    for i in range(out.shape[0]):
                        config_idx = used_batch[i].config_indices
                        push_idx = (
                            batch_list[i].partition_idx.cpu()
                            + config_idx * (len(batch_list[i].partptr) - 1)
                            + segments_to_train[i]
                        )
                        emb_table.push(out[i].unsqueeze(1).cpu(), push_idx)

                    pbar.set_postfix_str(f"loss: {loss:.4f}")

                if not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
                pbar.close()

                # Validation phase
                if (epoch + 1) % num_val_epochs != 0 and epoch != num_epochs - 1:
                    continue

                self.model.eval()
                metrics = LayoutMetrics()
                val_outs = {}  # save the output for each model

                for record in tqdm(
                    self.valid_dataset,
                    desc=f"Valid epoch: {epoch}",
                    leave=False,
                ):
                    config_runtime: torch.Tensor = record["config_runtime"]
                    outs: torch.Tensor = self._predict_one(record, infer_bs, device)
                    metrics.add(
                        record["model_id"],
                        outs.numpy(),
                        config_runtime.numpy(),
                    )

                    val_outs[record["model_id"]] = outs.numpy()

                prefix = "val/"
                scores = metrics.compute_scores(prefix=prefix)

                kendall = scores[f"{prefix}index_kendall"]
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(kendall)

                if use_logger:
                    wandb.log(scores)

                # Update best scores and save the model
                if kendall > best_score:
                    best_score = kendall
                    print("Best score updated " f"at epoch {epoch}: {best_score:.4f}")
                    if use_logger:
                        self.best_model_path = self._save_model(epoch, kendall)
                    not_improved = 0

                    output_path = os.path.join(wandb.run.dir, f"val_outs_{epoch}.plk")
                    with open(output_path, "wb") as f:
                        pickle.dump(val_outs, f)
                else:
                    not_improved += 1
                    if early_stopping > 0 and not_improved == early_stopping:
                        break

        except KeyboardInterrupt:
            pass

        if use_logger:
            self._save_model(epoch, kendall, suffix="_last")

        if "test" not in self.cfg.tasks:
            wandb.finish()

        return best_score
