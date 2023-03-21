from typing import List

import torch
import torch.nn as nn
from torch import Tensor

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


class GenAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, fake_disc_out: Tensor, **kwargs):
        labels = torch.full(fake_disc_out.shape[:1], 1.).to(fake_disc_out)  # to float and device
        return self.bce_loss(fake_disc_out, labels)


class DiscAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_disc_out: Tensor, real_disc_out: Tensor, **kwargs):
        labels = torch.full(fake_disc_out.shape[:1], 1.).to(fake_disc_out)  # to float and device
        disc_real_loss = self.bce_loss(real_disc_out, labels)
        labels.fill_(0.)
        disc_fake_loss = self.bce_loss(fake_disc_out, labels)
        return disc_fake_loss + disc_real_loss


class IdentityPreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, final_emb: Tensor, source_emb: Tensor, **kwargs):
        labels = torch.full((final_emb.size(0),), 1).to(final_emb.device).long()
        return self.loss(final_emb, source_emb, labels)


class LandmarkAlignmentLoss(nn.Module):
    def forward(self, side_lmk: Tensor, final_lmk: Tensor, target_lmk: Tensor, **kwargs):
        batch_size = side_lmk.size(0)
        return (torch.norm((side_lmk - target_lmk).view(batch_size, -1), dim=1) +
                torch.norm((final_lmk - target_lmk).view(batch_size, -1), dim=1)
                ).mean()


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


class StyleTransferLoss(nn.Module):
    def forward(self, final: Tensor, histogram_mapping: Tensor, **kwargs):
        batch_size = final.size(0)
        return torch.norm((final - histogram_mapping).view(batch_size, -1), dim=1).mean()



def compute_generator_loss(result, source):
    return ((result - source) ** 2).mean()


def compute_discriminator_loss(result, source):
    return ((result - source) ** 2).mean()