import torch
from torch import nn as nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms

from arcface_model.iresnet import iresnet100


class IdentityPreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()
        self.model = iresnet100(fp16=False)
        self.model.load_state_dict(torch.load('arcface_model/backbone.pth', map_location='cuda'))
        self.model.to('cuda').eval()
        
        # self.transforms_arcface = transforms.Compose([
        #     transforms.Resize((224, 224)),
        # ])

    def forward(self, final: Tensor, source: Tensor, **kwargs):
        
        with torch.no_grad():
            source_emb = self.model(F.interpolate(source, [112, 112], mode='bilinear', align_corners=False))
            final_emb = self.model(F.interpolate(final, [112, 112], mode='bilinear', align_corners=False))
            
        labels = torch.full((final_emb.size(0),), 1).to(final_emb.device).long()
        return self.loss(final_emb, source_emb, labels)
