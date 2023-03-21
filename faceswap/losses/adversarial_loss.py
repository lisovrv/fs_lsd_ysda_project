import torch
from torch import nn as nn, Tensor


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
