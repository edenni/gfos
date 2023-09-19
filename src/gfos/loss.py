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
