import torch
from torch import nn as nn, Tensor


class LandmarkAlignmentLoss(nn.Module):
    def forward(self, side_lmk: Tensor, final_lmk: Tensor, target_lmk: Tensor, **kwargs):
        batch_size = side_lmk.size(0)
        return (torch.norm((side_lmk - target_lmk).view(batch_size, -1), dim=1) +
                torch.norm((final_lmk - target_lmk).view(batch_size, -1), dim=1)
                ).mean()
