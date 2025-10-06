import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from skimage.metrics import structural_similarity as calculate_ssim

def compute_psnr(gt, pred):
    mse = torch.mean((gt - pred)**2)
    device = gt.device
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(device))
    psnr = psnr.cpu().item()
    return psnr

def compute_ssim(gt, pred):
    '''image size: (h, w, 3)'''
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    ssim = calculate_ssim(pred, gt, data_range=gt.max() - gt.min(), channel_axis=-1)
    return ssim

class LPIPSVal(nn.Module):
    def __init__(self):
        super().__init__()
        self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
        for p in self.val_lpips.net.parameters():
            p.requires_grad = False

    def forward(self, pred, x):
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            self.val_lpips(torch.clip(pred * 2 -1, -1, 1),
                           torch.clip( x * 2 -1, -1, 1))
            lpips = self.val_lpips.compute()
            self.val_lpips.reset()
        return lpips


def setup_eval_images():
    return compute_psnr, compute_ssim, LPIPSVal().cuda()