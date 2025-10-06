import numpy as np
import base64
from PIL import Image
from io import BytesIO
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, 
    Buffer, BufferView, Accessor, 
    Image as GLTFImage, Texture, Sampler, Material, PbrMetallicRoughness,
    FLOAT, UNSIGNED_INT, SCALAR, VEC2, VEC3, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
)
import argparse
import sys
import trimesh
sys.path.append('.')
sys.path.append('./MVMeshRecon')
import os
from pyhocon import ConfigFactory
import glob
from pytorch3d.io import save_obj, load_objs_as_meshes
from utils.general import load_tex_dict_from_tex_mesh_p3d, solve_intersection
from utils.sim import start_simulation_app, sim_validation, sim_scene_texture, export_usd_texture
import pickle
import torch
import subprocess
from model.gs import GS, SplatfactoOnMeshUCModelConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="path to config file", type=str)
    parser.add_argument("--timestamp", required=True, help="timestamp", type=str)
    args = parser.parse_args()

    conf = args.conf
    timestamp = args.timestamp
    conf = ConfigFactory.parse_file(conf)
    expname = conf.get_string('train.expname')

    exps_folder_name = "exps"
    expdir = os.path.join('./', exps_folder_name, expname)

    # Handle 'latest' timestamp by finding the most recent timestamp directory
    if timestamp == 'latest':
        if os.path.exists(expdir):
            timestamps = os.listdir(expdir)
            if len(timestamps) == 0:
                raise ValueError(f"No timestamp directories found in {expdir}")
            else:
                timestamp = sorted(timestamps)[-1]
                print(f"Using latest timestamp: {timestamp}")
        else:
            raise ValueError(f"Experiment directory {expdir} does not exist")
    
    plots_dir = os.path.join(expdir, timestamp, 'plots')

    # Find all GS pt files
    gs_pt_paths = sorted(glob.glob(os.path.join(plots_dir, "gauss_obj_*.pt")))
    gs_pt_paths = sorted(gs_pt_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
    
    # Create save directory for USD files (separate from plots_dir to avoid overwriting)
    save_usd_dir = os.path.join(plots_dir, "usd_gs")
    os.makedirs(save_usd_dir, exist_ok=True)
    print(f"Saving USD files to {save_usd_dir}")

    # Load translation dictionary if it exists
    translation_dict = None
    translation_dict_path = os.path.abspath(os.path.join(plots_dir, 'translation_dict.pkl'))
    if os.path.exists(translation_dict_path):
        with open(translation_dict_path, 'rb') as f:
            translation_dict = pickle.load(f)
        print(f"Loaded translation dictionary with {len(translation_dict)} entries")

    # Process each GS pt file
    for gs_idx, gs_pt_path in enumerate(gs_pt_paths):
        print(f"Processing {gs_pt_path} ({gs_idx + 1}/{len(gs_pt_paths)})")
        
        # Load GS dictionary from pt file
        gs_dict = torch.load(gs_pt_path, map_location="cpu")
        
        # Process the loaded dictionary following holoscene_train_gaussian.py pattern
        sh_degree = gs_dict["sh_degree"]
        
        # Create seed_gs dictionary with proper key mapping
        seed_gs = {
            'means': gs_dict["means"],
            'opacities': gs_dict['opacities'],
            'features_dc': gs_dict['shs_0'],  # Map shs_0 to features_dc
            'features_rest': gs_dict['shs_rest'],
            'scales': gs_dict['scales'],
            'quats': gs_dict['quats'],
            'sh_degree': sh_degree
        }
        
        # Create GS model
        config = SplatfactoOnMeshUCModelConfig()
        gs_model = GS(config=config, seed_gs=seed_gs)
        
        # Apply translation if available
        if translation_dict is not None and gs_idx in translation_dict:
            # Convert numpy translation to torch tensor and apply to means
            translation = torch.from_numpy(translation_dict[gs_idx]).float()
            gs_model.means = gs_model.means + translation.reshape(1, 3).cuda()
            print(f"Applied translation {translation} to GS model {gs_idx}")
        
        # Export to PLY file
        ply_filename = f"gauss_obj_{gs_idx}.ply"
        ply_path = os.path.join(save_usd_dir, ply_filename)
        gs_model.export_gs(ply_path)
        print(f"Exported PLY to {ply_path}")
        
        # Convert PLY to USDZ using subprocess
        try:
            cmd = ["python", "/home/hongchix/mount/holoscene/src/threedgrut/export/scripts/ply_to_usd.py", ply_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully converted {ply_filename} to USDZ")
            if result.stdout:
                print(f"Conversion output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {ply_filename} to USDZ: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
        except FileNotFoundError:
            print(f"Error: ply_to_usd.py script not found at /home/hongchix/mount/holoscene/src/threedgrut/export/scripts/ply_to_usd.py")
    
    print(f"Completed processing {len(gs_pt_paths)} GS files. USD files saved to {save_usd_dir}")

