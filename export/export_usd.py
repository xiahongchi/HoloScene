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

    mesh_paths = sorted(glob.glob(os.path.join(plots_dir, "surface_*.obj")))
    mesh_paths = sorted(mesh_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
    print("mesh_paths: ", mesh_paths)

    tex_dict_list = []
    
    for mesh_i, mesh_path in enumerate(mesh_paths):
        mesh_p3d = load_objs_as_meshes([mesh_path])
        tex_dict = load_tex_dict_from_tex_mesh_p3d(mesh_p3d)
        tex_dict["texture_path"] = os.path.abspath(os.path.join(plots_dir, f"surface_{mesh_i}.png"))
        tex_dict_list.append(tex_dict)
    
    all_meshes = [
        trimesh.Trimesh(
            tex_dict["vertices"], tex_dict["faces"], process=False
        ) for tex_dict in tex_dict_list
    ]
    translation_dict_path = os.path.abspath(os.path.join(plots_dir, 'translation_dict.pkl'))
    if os.path.exists(translation_dict_path):
        with open(translation_dict_path, 'rb') as f:
            translation_dict = pickle.load(f)
        for mesh_i in range(1, len(all_meshes)):
            all_meshes[mesh_i].vertices = all_meshes[mesh_i].vertices + translation_dict[mesh_i].reshape(1, 3)
    
    # all_meshes = solve_intersection(all_meshes)
    tex_dict_list = [
        {
            "vertices": all_meshes[tex_dict_i].vertices,
            "faces": all_meshes[tex_dict_i].faces,
            "vt": tex_dict_list[tex_dict_i]["vt"],
            "ft": tex_dict_list[tex_dict_i]["ft"],
            "texture_map": tex_dict_list[tex_dict_i]["texture_map"],
            "texture_path": os.path.join(plots_dir, f"surface_{tex_dict_i}.png")
        }
        for tex_dict_i in range(len(tex_dict_list))
    ]

    save_usd_dir = os.path.join(plots_dir, "usd")
    os.makedirs(save_usd_dir, exist_ok=True)
    print(f"Saved USD files to {save_usd_dir}")

    start_simulation_app()
    export_usd_texture(tex_dict_list, save_usd_dir)

