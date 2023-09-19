import numpy as np
from scipy.stats import kendalltau


def metric_for_layout_collections(
    predicted_rankings: np.array, actual_rankings: np.array
) -> float:
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
