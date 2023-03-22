import torch
from torch import nn as nn, Tensor
import torch.nn.functional as F

from faceswap.AdaptiveWingLoss.core import models
from faceswap.utils import detect_landmarks


class LandmarkAlignmentLoss(nn.Module):
    def __init__(self, path_to_model: str):
        super().__init__()
        self.model_ft = models.FAN(4, "False", "False", 98)
        self.setup_model(path_to_model)
        self._mse = torch.nn.MSELoss(reduction='none')

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

    def forward(self, side: Tensor, final: Tensor, target: Tensor, **kwargs):
        # TODO: Add denormalize
        batch_size = side.size(0)
        
        side = F.interpolate(side, [256, 256], mode='bilinear', align_corners=False)
        final = F.interpolate(final, [256, 256], mode='bilinear', align_corners=False)
        target = F.interpolate(target, [256, 256], mode='bilinear', align_corners=False)
        
        side_lmk = detect_landmarks(side, self.model_ft).view(batch_size, -1)
        final_lmk = detect_landmarks(final, self.model_ft).view(batch_size, -1)
        target_lmk = detect_landmarks(target, self.model_ft).view(batch_size, -1)
        return self._mse(side_lmk, target_lmk).mean() + self._mse(final_lmk, target_lmk).mean()
