import torch
from torch import nn as nn, Tensor


from AdaptiveWingLoss.core import models
from faceswap.utils import detect_landmarks


class IdentityPreservationLoss(nn.Module):
    def __init__(self, path_to_model: str):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()
        self.model_ft = models.FAN(4, "False", "False", 98)
        self.setup_model(path_to_model)

    def setup_model(self, path_to_model: str):
        checkpoint = torch.load(path_to_model, map_location='cpu')
        if 'state_dict' not in checkpoint:
            self.model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = self.model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            self.model_ft.load_state_dict(model_weights)
        self.model_ft.eval()

    def forward(self, source: Tensor, final: Tensor, **kwargs):
        batch_size = source.size(0)
        source_emb = detect_landmarks(source, self.model_ft).view(batch_size, -1)  # Add denormalize
        final_emb = detect_landmarks(final, self.model_ft).view(batch_size, -1)  # Add denormalize
        labels = torch.full((batch_size,), 1).to(final_emb.device).long()
        return self.loss(final_emb, source_emb, labels)
