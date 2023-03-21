from typing import List

import lpips
import torch
from torch import nn as nn, Tensor


class ReconstructionLoss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = torch.tensor([alpha])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        side: Tensor,
        final: Tensor,
        **kwargs,
    ):
        if torch.allclose(source, target):
            batch_size = source.size(0)
            loss = torch.norm((final - target).view(batch_size, -1), dim=1) + \
                torch.norm((side - target).view(batch_size, -1), dim=1)
            lpips_loss = (self.loss_fn_vgg(final, target) + self.loss_fn_vgg(side, target)).squeeze()
            return (loss + self.alpha * lpips_loss).mean()
        else:
            return 0.
