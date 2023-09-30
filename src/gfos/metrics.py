import numpy as np
import torch
from scipy.stats import kendalltau
from torchmetrics import Metric


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


class TopKError(Metric):
    def __init__(self, top_k=5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.top_k = top_k
        self.add_state("error", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Get the indices of the top k elements in target
        target_top_k = target.topk(self.top_k, largest=True)[1]
        preds_top_k = preds.topk(self.top_k, largest=True)[1]

        # Calculate error
        error = len(set(target_top_k.tolist()) - set(preds_top_k.tolist()))

        self.error += error
        self.total += self.top_k  # since we are comparing top_k elements

    def compute(self):
        top_k_error = self.error.float() / self.total
        return top_k_error
