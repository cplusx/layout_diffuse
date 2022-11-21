# python fid_eval.py \
# --dataset celeb_mask \
# -s /home/ubuntu/disk2/data/face/CelebAMask-HQ/CelebA-HQ-img-train-256x256 \
# -d experiments/celeb_mask_ldm_partial_attn/sampling_at_00279_image

run_once () {
    res=$1
    epoch=$2
    python fid_eval.py \
    -s /home/ubuntu/disk2/data/face/CelebAMask-HQ/CelebA-HQ-img \
    --resize_s \
    -d experiments/celeb_mask_ldm_${res}_samples/epoch_${epoch}/image > tmp/data_efficiency_res_${res}_epoch_${epoch}.txt
}

# CUDA_VISIBLE_DEVICES=1 run_once 128 "00099" && run_once 128 "00199" && run_once 128 "00499" && run_once 128 "00999" && run_once 128 "01999" &

# CUDA_VISIBLE_DEVICES=2 run_once 256 "00049" && run_once 256 "00099" && run_once 256 "00249" && run_once 256 "00499" && run_once 256 "00999" &

# CUDA_VISIBLE_DEVICES=3 run_once 512 "00024" && run_once 512 "00049" && run_once 512 "00124" && run_once 512 "00249" && run_once 512 "00499" &

CUDA_VISIBLE_DEVICES=0 run_once 1024 "00012" && run_once 1024 "00024" && run_once 1024 "00062" && run_once 1024 "00124" && run_once 1024 "00249" &

CUDA_VISIBLE_DEVICES=1 run_once 2048 "00006" && run_once 2048 "00012" && run_once 2048 "00031" && run_once 2048 "00062" && run_once 2048 "00124" &

# seq 4 5 10| while read e; do
#     python fid_eval.py \
#     -s /home/ubuntu/disk2/data/face/CelebAMask-HQ/CelebA-HQ-img \
#     --resize_s \
#     -d experiments/celeb_mask_ldm_v2/epoch_0000$e/image > tmp/celeb_v2_fid_e_$e.txt
# done

# seq 14 5 30 | while read e; do
#     python fid_eval.py \
#     -s /home/ubuntu/disk2/data/face/CelebAMask-HQ/CelebA-HQ-img \
#     --resize_s \
#     -d experiments/celeb_mask_ldm_v2/epoch_000$e/image > tmp/celeb_v2_fid_e_$e.txt
# done