from typing import List

import lpips
import torch
from torch import nn as nn, Tensor


class ReconstructionLoss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = torch.tensor([alpha])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self._mse = torch.nn.MSELoss(reduction='none')

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        side: Tensor,
        final: Tensor,
        **kwargs,
    ):
        if not torch.allclose(source, target):
            batch_size = source.size(0)

            loss = self._mse(final, target).mean() + self._mse(side, target).mean()
            
            lpips_loss = (self.loss_fn_vgg(final, target) + self.loss_fn_vgg(side, target)).squeeze().mean()
            
            return loss + self.alpha.to(loss.device) * lpips_loss
        else:
            return 0.
