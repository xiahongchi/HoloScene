conda activate holoscene

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


python training/exp_runner.py --conf confs/replica/room_0/replica_room_0.conf

python training/exp_runner_post.py --conf confs/replica/room_0/replica_room_0_post.conf \
    --is_continue \
    --timestamp latest \
    --checkpoint 1000

python training/exp_runner_texture.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --is_continue \
    --timestamp latest \
    --checkpoint 1000

python training/exp_runner_gaussian_on_mesh.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --is_continue \
    --timestamp latest \
    --checkpoint 1000

python export/export_glb.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --timestamp latest

python export/export_usd.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --timestamp latest

python export/export_gs_usd.py --conf confs/replica/room_0/replica_room_0_tex.conf \
    --timestamp latest