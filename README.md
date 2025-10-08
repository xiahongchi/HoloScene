<h2 align="center">
  <b>HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video</b>

  <b><i>NeurIPS 2025 </i></b>
</h2>

<p align="center">
    <a href='https://arxiv.org/abs/2510.05560'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://xiahongchi.github.io/HoloScene/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

<p align="center">
    <a href="https://xiahongchi.github.io/">Hongchi Xia<sup>1</sup></a>,
    <a href="https://chih-hao-lin.github.io/">Chih-Hao Lin<sup>1</sup></a>,
    <a href="https://haoyuhsu.github.io/">Hao-Yu Hsu<sup>1</sup></a>,
    <br>
    <a href="https://www.linkedin.com/in/quentinleboutet/?originalSubdomain=de">Quentin Leboutet<sup>2</sup></a>,
    <a href="https://sites.google.com/view/katelyn-gao/home">Katelyn Gao<sup>2</sup></a>,
    <a href="https://www.linkedin.com/in/michael-paulitsch/?originalSubdomain=de">Michael Paulitsch<sup>2</sup></a>,
    <br>
    <a href="https://scholar.google.com/citations?user=QGlp5ywAAAAJ&hl=en">Benjamin Ummenhofer<sup>2</sup></a>,
    <a href="https://shenlong.web.illinois.edu/">Shenlong Wang<sup>1</sup></a>
</p>

<p align="center">
    <sup>1</sup>University of Illinois at Urbana-Champaign &nbsp;&nbsp;&nbsp;
    <sup>2</sup>Intel
</p>

<p align="center">
    <img src="https://xiahongchi.github.io/HoloScene/src/images/teaser.png" width=90%>
</p>

HoloScene leverages a comprehensive interactive scene-graph representation, encoding object geometry, appearance, and physical properties alongside hierarchical and inter-object relationships. Reconstruction is formulated as an energy-based optimization problem, integrating observational data, physical constraints, and generative priors into a unified, coherent objective. The resulting digital twins exhibit complete and precise geometry, physical stability, and realistic rendering from novel viewpoints.

## News

- [2025/10/08] Code is released. For more information, please visit our [project page](https://xiahongchi.github.io/HoloScene/)!

## Installation
```bash
conda create -n holoscene -y python=3.10
conda activate holoscene

pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
pip install git+https://github.com/nerfstudio-project/gsplat.git@24abe714105441f049b50be1fc8eb411d727e6e6
pip install git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8#egg=nvdiffrast
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install isaacsim==4.2.0.2  --extra-index-url https://pypi.nvidia.com
pip install isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

# Download Wonder3D+ checkpoints
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='flamehaze1115/Wonder3D_plus', local_dir='./ckpts')
"

# Download LaMa model
cd lama
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
cd ..

# Download Omnidata normal estimation model
wget "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt"

# Set environment variables (if export gaussian splat usd needed)
export CPATH=$CPATH:./tiny-cuda-nn-include:$CONDA_PREFIX/include
```

## Data
Please download the preprocessed data at following `./scripts/data_download.sh` and unzip in the `data_dir` folder. The resulting folder structure should be:
```
└── HoloScene
  └── data_dir
    ├── replica
        ├── room_0 
            ├── images 
            ├── transforms.json 
        ├── room_1 
        ├── room_2 
    ├── scannetpp
        ├── acd69a1746 
        ├── 67d702f2e8 
        ├── 7831862f02 
    ├── gibson
        ├── Beechwood_0_int 
        ├── Beechwood_1_int 
    ├── custom
        ├── siebelgame 
```

## Training and Running

### Example: Replica Room 0
```bash
conda activate holoscene

# Stage 0: Generate depth and normal priors

data_dir="data_dir/replica/room_0"

python marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir="${data_dir}/images" \
    --output_dir="${data_dir}/"

python marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir="${data_dir}/images" \
    --output_dir="${data_dir}/"

# Stage 1: Initial reconstruction
python training/exp_runner.py --conf confs/replica/room_0/replica_room_0.conf

# Stage 2: Post-processing
python training/exp_runner_post.py --conf confs/replica/room_0/replica_room_0_post.conf \
    --is_continue \
    --timestamp latest \
    --checkpoint 1000

# Stage 3: Texture refinement
python training/exp_runner_texture.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --is_continue \
    --timestamp latest \
    --checkpoint 1000

# Stage 4: Gaussian on mesh
python training/exp_runner_gaussian_on_mesh.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --is_continue \
    --timestamp latest \
    --checkpoint 1000

# Export results
python export/export_glb.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --timestamp latest

python export/export_usd.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --timestamp latest

python export/export_gs_usd.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --timestamp latest
```

## Acknowledgements
We thank the authors of:
* [MonoSDF](https://github.com/autonomousvision/monosdf)
* [RICO](https://github.com/kyleleey/RICO)
* [ObjectSDF++](https://github.com/QianyiWu/objectsdf_plus)
* [PhyRecon](https://github.com/PhyRecon/PhyRecon)
* [DRAWER](https://github.com/xiahongchi/DRAWER)
* [Wonder3D](https://github.com/xxlong0/Wonder3D)
* [LaMa](https://github.com/advimman/lama)
* [3DGrut](https://github.com/nv-tlabs/3dgrut/tree/main)

for their foundational work and open-source contributions.

## Citation

```bibtex
@misc{xia2025holoscene,
      title={HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video}, 
      author={Hongchi Xia and Chih-Hao Lin and Hao-Yu Hsu and Quentin Leboutet and Katelyn Gao and Michael Paulitsch and Benjamin Ummenhofer and Shenlong Wang},
      year={2025},
      eprint={2510.05560},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.05560}, 
}
```
