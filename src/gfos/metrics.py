from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kendalltau


def kendall(predicted_rankings: np.array, actual_rankings: np.array) -> float:
    """
    Calcuates the kendal tau correaltion between the
    predicted and actual performance rankings.

    Args:
        predicted_rankings (np.array): Performance rankings predicted by model
        actual_rankings (np.array): Actual performance rankings

    Returns:
        Kendall's Tau correlation coefficent value of the two lists

    """
    if isinstance(predicted_rankings, list):
        predicted_rankings = np.array(predicted_rankings)
    if isinstance(actual_rankings, list):
        actual_rankings = np.array(actual_rankings)

    if len(predicted_rankings) != len(actual_rankings):
        raise ValueError(
            f"Length of predicted rankings (len = {len(predicted_rankings)})"
            f" and actual rankings (len = {len(actual_rankings)}) must be equal."
        )

    corr, _ = kendalltau(predicted_rankings, actual_rankings)
    return corr


def topk_error(preds: np.array, target: np.array, top_k: int = 5, index=False):
    """
    Calculates the top-k error between the predicted and actual
    performance rankings.

    Args:
        preds (np.array): Performance rankings predicted by model
        target (np.array): Actual performance rankings
        top_k (int): Number of top elements to compare

    Returns:
        Top-k error between the two lists

    """
    # Get the indices of the top k elements in target
    if not index:
        preds = preds.argsort()
        target = target.argsort()

    target_top_k = target[:top_k]
    preds_top_k = preds[:top_k]

    # Calculate error
    error = len(set(target_top_k.tolist()) - set(preds_top_k.tolist()))

    return error / top_k


# class TopKError(Metric):
#     def __init__(self, top_k=5, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.top_k = top_k
#         self.add_state("error", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         # Get the indices of the top k elements in target
#         target_top_k = target.topk(self.top_k, largest=True)[1]
#         preds_top_k = preds.topk(self.top_k, largest=True)[1]

#         # Calculate error
#         error = len(set(target_top_k.tolist()) - set(preds_top_k.tolist()))

#         self.error += error
#         self.total += self.top_k  # since we are comparing top_k elements

#     def compute(self):
#         top_k_error = self.error.float() / self.total
#         return top_k_error


class LayoutMetrics:
    def __init__(self) -> None:
        self.model_ids: list[str] = []
        self.preds: list[np.array] = []
        self.targets: list[np.array] = []

    def add(self, model_id: str, preds: np.array, targets: np.array) -> None:
        assert isinstance(model_id, str)
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self.model_ids.append(model_id)
        self.preds.append(preds)
        self.targets.append(targets)

    @property
    def raw_kendall(self) -> list[float]:
        return [
            kendall(pred, target)
            for pred, target in zip(self.preds, self.targets)
        ]

    @property
    def index_kendall(self) -> list[float]:
        kendalls = []
        for pred, target in zip(self.preds, self.targets):
            sorted_index = np.argsort(target)
            sorted_preds = pred[sorted_index]
            kendalls.append(
                kendall(sorted_preds.argsort(), np.arange(len(target)))
            )
        return kendalls

    @property
    def top100_error(self) -> list[float]:
        return [
            topk_error(pred, target, top_k=100)
            for pred, target in zip(self.preds, self.targets)
        ]

    @property
    def top500_error(self) -> list[float]:
        return [
            topk_error(pred, target, top_k=500)
            for pred, target in zip(self.preds, self.targets)
        ]

    def compute_scores(self, prefix: str = "") -> dict[str, float]:
        scores = {
            f"{prefix}raw_kendall": np.mean(self.raw_kendall),
            f"{prefix}index_kendall": np.mean(self.index_kendall),
            f"{prefix}top100_error": np.mean(self.top100_error),
            f"{prefix}top500_error": np.mean(self.top500_error),
        }
        scores.update(
            {
                f"{prefix}kendall_{model}": score
                for model, score in zip(self.model_ids, self.index_kendall)
            }
        )
        return scores
