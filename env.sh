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

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='flamehaze1115/Wonder3D_plus', local_dir='./ckpts')
"

cd lama
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip

cd ..
wget "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt"

export CPATH=$CPATH:/home/hongchix/mount/holoscene/src/tiny-cuda-nn-include:$CONDA_PREFIX/include