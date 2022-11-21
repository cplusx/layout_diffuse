run_once () {
    expt=$1
    appendix=$3
    echo "Score for ${expt}, epoch $2"
    python fid_eval.py \
    -s /home/ubuntu/disk2/data/COCO/train2017 \
    --resize_s \
    -d experiments/${expt}/epoch_000$2$appendix/raw_tensor_npy
    # -d experiments/${expt}/epoch_000$2$appendix/sample_image
}

expt="laion_ldm_cocostuff_layout_no_caption"
appendix="_plms_100_5.0"
epoch="09"
CUDA_VISIBLE_DEVICES=5 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &

expt="laion_ldm_cocostuff_layout_no_caption"
appendix="_plms_100_5.0"
epoch="29"
CUDA_VISIBLE_DEVICES=6 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &

expt="laion_ldm_cocostuff_layout_no_caption"
appendix="_plms_100_5.0"
epoch="59"
CUDA_VISIBLE_DEVICES=7 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &

# expt="laion_ldm_cocostuff_layout_caption_v9"
# appendix="_plms_200_5.0"
# epoch="59"
# CUDA_VISIBLE_DEVICES=6 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &

# expt="laion_ldm_cocostuff_layout_caption_ablation_no_instance_attn"
# appendix="_plms_100_5.0"
# epoch="09"
# CUDA_VISIBLE_DEVICES=6 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &

# expt="laion_ldm_cocostuff_layout_caption_ablation_no_instance_attn"
# appendix="_plms_100_5.0"
# epoch="29"
# CUDA_VISIBLE_DEVICES=5 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &

# expt="laion_ldm_cocostuff_layout_caption_ablation_no_instance_attn"
# appendix="_plms_100_5.0"
# epoch="59"
# CUDA_VISIBLE_DEVICES=7 run_once $expt $epoch $appendix >> tmp/fid_${expt}_${epoch}.txt 2>&1 &