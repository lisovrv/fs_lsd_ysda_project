import torch
from torch import nn as nn, Tensor


class IdentityPreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, final_emb: Tensor, source_emb: Tensor, **kwargs):
        labels = torch.full((final_emb.size(0),), 1).to(final_emb.device).long()
        return self.loss(final_emb, source_emb, labels)
