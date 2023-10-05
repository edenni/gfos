import logging
import os

import torch
from hydra.utils import instantiate

from ..data.dataset import LayoutDataset, Normalizer
from ..data.utils import load_layout
from ..loss import MultiElementRankLoss
from ..model.gnn import LayoutModel
from .base import Pipeline

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
            raise FileNotFoundError(
                f"Layout directory {layout_dir} does not exist"
            )

        # Check if normalizer exists
        if normalizer_path and not os.path.exists(normalizer_path):
            raise FileNotFoundError(
                f"Normalizer {normalizer_path} does not exist"
            )

        # Load layout files
        logger.info(f"Loading layout from {layout_dir}")
        layout_files = load_layout(
            base_dir=layout_dir, model_type=source, compile_type=search
        )

        # Load normalizer
        normalizer = Normalizer.from_json(
            normalizer_path, source=source, search=search
        )

        # Create training and validation dataset
        self.train_dataset = LayoutDataset(
            files=layout_files,
            max_configs=max_configs,
            num_configs=num_configs,
            config_edges=config_edges,
            normalizer=normalizer,
        )

        self.valid_dataset = LayoutDataset(
            files=layout_files,
            config_edges=config_edges,
            normalizer=normalizer,
        )

    def _setup_model(self):
        conv_layer = self.cfg.model.conv_layer
        op_embedding_dim = self.cfg.model.op_embedding_dim
        config_dim = self.cfg.model.config_dim
        graph_dim = self.cfg.model.graph_dim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_feat_dim = self.train_dataset[0]["node_feat"].shape[-1]
        node_config_dim = self.train_dataset[0]["node_config_feat"].shape[-1]
        model = instantiate(
            self.cfg.model,
            node_feat_dim=node_feat_dim,
            node_config_dim=node_config_dim,
        ).to(device)

        criterion = MultiElementRankLoss(
            margin=margin, number_permutations=number_permutations
        )

        optimizer = optim.AdamW(
            [
                {
                    "name": "lr_embed",
                    "params": model.embedding.parameters(),
                    "lr": learning_rate / 10,
                },
                {
                    "name": "lr_model_gnn",
                    "params": model.model_gnn.parameters(),
                    "lr": learning_rate / 10,
                },
                {
                    "name": "lr_config_prj",
                    "params": model.config_prj.parameters(),
                    "lr": learning_rate / 10,
                },
                {
                    "name": "lr_config_mp",
                    "params": model.config_mp.parameters(),
                    "lr": learning_rate / 10,
                },
                {
                    "name": "lr_config_gnn",
                    "params": model.config_gnn.parameters(),
                    "lr": learning_rate,
                },
                {
                    "name": "lr_dense",
                    "params": model.dense.parameters(),
                    "lr": learning_rate,
                },
            ],
            betas=[0.85, 0.9],
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.1,
            patience=2,  # 5 times evaluation = 5 * NUM_VAL_EPOCHS epochs
            threshold=0.01,
            min_lr=min_lr,
        )

    def train(self):
        if not DEBUG:
            run = wandb.init(
                project=WANDB_PROJECT,
                dir=WANDB_DIR,
                name=WANDB_RUN_NAME,
                config=configs,
                tags=TAGS,
            )
            run.watch(model, log="all")
            run.log_code("../")

            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"../../logs/{WANDB_RUN_NAME}/{time_str}"
            os.makedirs(log_dir, exist_ok=True)

        best_score = -1

        # scaler = GradScaler()
        loss_mean = 0
        for epoch in range(num_epochs):
            # Shuffle the training dataset
            permutation = np.random.permutation(len(train_dataset))

            # Training phase
            model.train()
            pbar = tqdm(permutation, leave=False)

            for i in pbar:
                record = train_dataset[i]
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
                out = model(
                    node_feat,
                    node_opcode,
                    edge_index,
                    node_config_feat,
                    node_config_ids,
                    config_edge_index,
                )

                loss = criterion(out, config_runtime)
                loss = loss / accum_iter
                loss_mean += loss.item()
                loss.backward()
                # scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                pbar.set_description(f"epoch: {epoch} loss: {loss_mean:.4f}")

                if ((i + 1) % accum_iter == 0) or (
                    i + 1 == len(train_dataset)
                ):
                    # scaler.step(optimizer)
                    # scaler.update()
                    optimizer.step()
                    optimizer.zero_grad()

                    if not DEBUG:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "train/lr_slow": optimizer.param_groups[0][
                                    "lr"
                                ],
                                "train/lr_fast": optimizer.param_groups[-1][
                                    "lr"
                                ],
                                "train/loss": loss_mean,
                            }
                        )
                    loss_mean = 0

            pbar.close()

            if (epoch + 1) % NUM_VAL_EPOCHS != 0 and epoch != num_epochs - 1:
                continue

            model.eval()

            # Validation phase
            # Scores placeholder
            val_loss = []
            kendalltau_scores = []
            opa_scores = []
            top500_scores = []
            top100_scores = []

            with torch.no_grad():
                for record in tqdm(val_dataset, desc="valid", leave=False):
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

                    for i in range(
                        0, num_configs, INFERENCE_CONFIGS_BATCH_SIZE
                    ):
                        end_i = min(
                            i + INFERENCE_CONFIGS_BATCH_SIZE, num_configs
                        )
                        out: torch.Tensor = model(
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
                    # opa_scores.append(opa(config_runtime[None], outs[None]))
                    top100_scores.append(
                        topk_error(outs, config_runtime, top_k=100)
                    )
                    top500_scores.append(
                        topk_error(outs, config_runtime, top_k=500)
                    )

            kendalltau_mean = np.mean(kendalltau_scores)
            # opa_mean = np.mean(opa_scores)
            top100_mean = np.mean(top100_scores)
            top500_mean = np.mean(top500_scores)
            scheduler.step(kendalltau_mean)

            if not DEBUG:
                wandb.log(
                    {
                        "val/kendalltau": kendalltau_mean,
                        # "val/opa": opa_mean,
                        "val/top100_error": top100_mean,
                        "val/top500_error": top500_mean,
                    }
                )

            print(
                f"epoch {epoch}, kendall = {kendalltau_mean:.4f}, "
                # f"opa = {opa_mean:.4f}, "
                f"top500 = {top500_mean:.4f}"
            )

            # Update best scores and save the model if the mean score improves
            if kendalltau_mean > best_score:
                best_score = kendalltau_mean
                print(f"Best score updated: {best_score:.4f}")
                if not DEBUG:
                    filename = f"{epoch}_{kendalltau_mean:.4f}.pth"
                    path = os.path.join(wandb.run.dir, filename)
                    torch.save(
                        model.state_dict(),
                        path,
                    )

        if not DEBUG:
            run.finish()
