import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dpt_depth import DPTDepthModel
import torch

def load_normal_model(ckpt_path):
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model.cuda().eval()

@torch.no_grad()
def infer_normal(model, image):
    # image: numpy array range (h, w, 3) [0, 1] np.float32
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
    output = model(img_tensor).clamp(min=0, max=1)[0].permute(1, 2, 0).cpu().numpy()
    return output * 2 - 1