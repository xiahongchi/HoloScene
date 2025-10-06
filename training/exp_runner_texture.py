import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNX_DISABLE_CPU_AFFINITY"] = "1"

import sys

sys.path.append('.')
sys.path.append('./MVMeshRecon')
import argparse
import torch

from training.holoscene_train_texture import HoloSceneTrainTextureRunner
import datetime

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]') 
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel') # this is not required in torch 2.0
    parser.add_argument("--ft_folder", type=str, default=None, help='If set, finetune model from the given folder path')
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--none_wandb", default=False, action="store_true", help="If set, do not use wandb")

    opt = parser.parse_args()

    gpu = 0

    seed_everything(42)

    trainrunner = HoloSceneTrainTextureRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    ft_folder = opt.ft_folder,
                                    description=opt.description,
                                    use_wandb=not opt.none_wandb,
                                    )

    trainrunner.run()
