from typing import List

import torch
from torch import nn as nn

from faceswap.utils import camel2snake_case


class FinalLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], coefs: List[nn.Module]):
        super().__init__()
        if len(losses) != len(coefs):
            raise ValueError(f'len(losses) != len(coefs): {len(losses)} != {len(coefs)}')
        self.losses = nn.ModuleList(losses)
        self.coefs = torch.tensor(coefs)

    def forward(self, **kwargs):
        result = {}
        final_loss = 0.
        for loss_func, coef in zip(self.losses, self.coefs):
            loss = loss_func(**kwargs)
            name = camel2snake_case(loss_func.__class__.__name__)
            result[name] = loss
            final_loss = final_loss + coef * loss
        result['loss'] = final_loss
        return result
