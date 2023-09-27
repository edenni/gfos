import torch
import torch.nn as nn


class MultiElementRankLoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """

    def __init__(
        self, margin: float = 0.0, number_permutations: int = 1
    ) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MarginRankingLoss(
            margin=margin, reduction="none"
        )
        self.number_permutations = number_permutations

    def calculate_rank_loss(
        self,
        outputs: torch.Tensor,
        config_runtime: torch.Tensor,
    ):
        """
        Generates a permutation of the predictions and targets and calculates the loss MarginRankingLoss against the permutation
        Args:
            outputs: Tensor of shape (bs, seq_len) with the outputs of the model
            config_runtime: Tensor of shape (bs, seq_len) with the runtime of the model
        Returns:
            loss: Tensor of shape (bs, seq_len) with the loss for each element in the batch
        """
        num_configs = outputs.shape[0]
        permutation = torch.randperm(num_configs)
        permuted_runtime = config_runtime[permutation]
        labels = 2 * ((config_runtime - permuted_runtime) > 0) - 1
        permuted_output = outputs[permutation]
        loss = self.loss_fn(
            outputs.view(-1, 1),
            permuted_output.view(-1, 1),
            labels.view(-1, 1),
        )
        return loss.mean()

    def forward(
        self,
        outputs: torch.Tensor,
        config_runtime: torch.Tensor,
    ):
        loss = 0
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(outputs, config_runtime)
        return loss / self.number_permutations


def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    observation_loss = (
        torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    )

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))
