mkdir -p data_dir

cd data_dir

mkdir -p custom
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/custom/siebelgame.zip
cd ..

mkdir -p replica
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/replica/room_0.zip
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/replica/room_1.zip
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/replica/room_2.zip
cd ..

mkdir -p scannetpp
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/scannetpp/acd69a1746.zip
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/scannetpp/67d702f2e8.zip
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/scannetpp/7831862f02.zip
cd ..

mkdir -p gibson
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/gibson/Beechwood_0_int.zip
wget https://huggingface.co/datasets/hongchi/HoloScene/resolve/main/gibson/Beechwood_1_int.zip
cd ..
