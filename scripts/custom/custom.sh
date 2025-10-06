conda activate holoscene

data_dir="data_dir/custom/siebelgame"

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


python training/exp_runner.py --conf confs/custom/siebelgame/custom_siebelgame.conf

python training/exp_runner_post.py --conf confs/custom/siebelgame/custom_siebelgame_post.conf \
    --is_continue \
    --timestamp 2025_09_25_00_05_40 \
    --checkpoint 1000

python training/exp_runner_texture.py --conf confs/custom/siebelgame/custom_siebelgame_tex.conf \
    --is_continue \
    --timestamp 2025_09_28_13_40_51 \
    --checkpoint 1000

python training/exp_runner_gaussian_on_mesh.py --conf confs/custom/siebelgame/custom_siebelgame_tex.conf \
    --is_continue \
    --timestamp 2025_09_24_02_43_07 \
    --checkpoint 1000


python training/exp_runner.py --conf confs/replica_room_1.conf