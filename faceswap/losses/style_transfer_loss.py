import torch
from torch import nn as nn, Tensor
from torch.autograd import Variable
from faceswap.losses.hm import histogram_matching


class StyleTransferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._criterion = torch.nn.L1Loss()
        
    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    
    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)
    
    def mask_preprocess(self, mask):
        index_tmp = mask.nonzero()
        # print('4_1', index_tmp.shape)
        x_index = index_tmp[:, 2]
        y_index = index_tmp[:, 3]
        
        mask = self.to_var(mask, requires_grad=False)
        
        return mask, [x_index, y_index, x_index, y_index]
    
    def forward(self, final: Tensor, target: Tensor, t_mask: Tensor, **kwargs):
        
        final = (self.de_norm(final) * 255).squeeze()
        target = (self.de_norm(target) * 255).squeeze()
        
        loss = 0.
        for img, tar, mask in zip(final, target, t_mask):
            # img = torch.unsqueeze(img, 0)
            # tar = torch.unsqueeze(tar, 0)
            mask = torch.unsqueeze(mask, 0)
            # print('4_0', mask.shape)
        
            mask, index = self.mask_preprocess(mask)


            # print('4', mask.shape)

            mask = mask.expand(1, 3, mask.size(2), mask.size(2)).squeeze()

            # print('5', mask.shape)

            img_masked = img * mask
            tar_masked = tar * mask

        
            img_match = histogram_matching(img_masked, tar_masked, index)

            img_match = self.to_var(img_match, requires_grad=False)

            loss += self._criterion(img_masked, img_match)
            
        # print(loss)
        return loss

