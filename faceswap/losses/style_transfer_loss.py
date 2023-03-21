import torch
from torch import nn as nn, Tensor


class StyleTransferLoss(nn.Module):
    def forward(self, final: Tensor, histogram_mapping: Tensor, **kwargs):
        batch_size = final.size(0)
        return torch.norm((final - histogram_mapping).view(batch_size, -1), dim=1).mean()
