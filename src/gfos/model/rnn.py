from typing import Callable, List

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score
from torchmetrics import ConfusionMatrix


class RNNEncoder(LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = False,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        output_size: int = 1,
        threshold: float = 0.5,
        linear_hidden_size: int = 256,
        pos_weight: float = 0.5,
        optimizer: Callable = None,
        scheduler: Callable = None,
        scheduler_conf: DictConfig = None,
        net_name: str = "lstm",
    ):
        super().__init__()
        self.save_hyperparameters()

        net_name = net_name.upper()
        assert net_name in (
            "GRU",
            "LSTM",
        ), "Model should be LSTM or GRU"

        net_class = getattr(nn, net_name)

        self.lstm = net_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * 2 if bidirectional else hidden_size

        # self.prj = nn.Linear(fc_input, linear_hidden_size)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(fc_input, output_size)

        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        self.criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
        self.cm = ConfusionMatrix(task="binary")

    def forward(self, x):
        if isinstance(x, tuple) and len(x) > 1:
            x = x[0]
        lstm_out, _ = self.lstm(x)

        if isinstance(lstm_out, nn.utils.rnn.PackedSequence):
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=self.hparams.batch_first
            )

        last = lstm_out[:, -1] if self.hparams.batch_first else lstm_out[-1]
        # y_pred = self.fc(self.relu(self.prj(last)))
        y_pred = self.fc(last)

        return y_pred

    def _shared_step(self, x, target):
        preds = self(x)
        loss = self.criterion(preds, target)
        return loss, preds

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss, _ = self._shared_step(x, target)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        loss, preds = self._shared_step(x, target)
        self.log("val/loss", loss)

        return preds, target

    def validation_epoch_end(self, outputs):
        preds = torch.concat([x[0] for x in outputs]).detach().cpu()
        targets = torch.concat([x[1] for x in outputs]).detach().cpu()

        f1 = f1_score(targets, preds > self.hparams.threshold, average="macro")
        self.log("val/f1", f1, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            # Karpathy's lr :)
            optimizer = optim.Adam(self.parameters(), lr=3e-4)
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        sche_conf = self.hparams.scheduler_conf
        sche_conf["scheduler"] = scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": dict(sche_conf),
        }
