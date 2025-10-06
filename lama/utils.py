import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from saicinpainting.training.trainers import load_checkpoint

from omegaconf import OmegaConf
import yaml

def load_model(config_path, checkpoint_path):
    predict_config = OmegaConf.load(config_path)
    predict_config.model.path = checkpoint_path
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    out_ext = predict_config.get('out_ext', '.png')
    checkpoint_path = os.path.join(predict_config.model.path,
                                   'models',
                                   predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()

    model = model.to('cuda')

    return model, predict_config

@torch.no_grad()
def inpaint(model, predict_config, image, mask):
    """

    :param model:
    :param image: (H, W, 3) torch.Tensor float32
    :param mask: (H, W) torch.Tensor boolean
    :return:
    """
    batch = {}

    H, W = mask.shape

    batch["mask"] = mask.reshape(1, 1, H, W).float().cuda().detach()
    batch["image"] = image.permute(2, 0, 1).unsqueeze(0).float().cuda().detach()

    lama_inpaint_image = model(batch)[predict_config.out_key][0].permute(1, 2, 0).detach()
    return lama_inpaint_image