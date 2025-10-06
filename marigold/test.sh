data_dir="data/replica/room_1"

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
