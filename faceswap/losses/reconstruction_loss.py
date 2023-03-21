import torch
from torch import nn as nn, Tensor


class ReconstructionLoss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = torch.tensor([alpha])

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        side: Tensor,
        final: Tensor,
        target_features: Tensor,
        side_features: Tensor,
        final_features: Tensor,
        **kwargs,
    ):
        if torch.allclose(source, target):
            batch_size = source.size(0)
            return (
                    torch.norm((final - target).view(batch_size, -1), dim=1) +
                    self.alpha * torch.norm((final_features - target_features).view(batch_size, -1), dim=1) +
                    torch.norm((side - target).view(batch_size, -1), dim=1) +
                    self.alpha * torch.norm((side_features - target_features).view(batch_size, -1), dim=1)
            ).mean()
        else:
            return 0.
