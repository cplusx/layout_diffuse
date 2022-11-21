# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_caption_ablation_no_instance_attn/epoch_00009_plms_100_5.0/raw_tensor
# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_caption_ablation_no_instance_attn/epoch_00029_plms_100_5.0/raw_tensor
# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_caption_ablation_no_instance_attn/epoch_00059_plms_100_5.0/raw_tensor

# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_caption_v9/epoch_00059_plms_200_5.0/raw_tensor
seq 4 5 10 | while read e; do
    python scripts/convert_npz_to_npy.py -s experiments/celeb_mask_ldm_v2/epoch_0000$e/raw_tensor
done

seq 14 5 30 | while read e; do
    python scripts/convert_npz_to_npy.py -s experiments/celeb_mask_ldm_v2/epoch_000$e/raw_tensor
done
# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_caption_v9/epoch_00009/raw_tensor
# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_caption_v9/epoch_00029/raw_tensor

# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_no_caption/epoch_00009_plms_100_5.0/raw_tensor
# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_no_caption/epoch_00029_plms_100_5.0/raw_tensor
# python scripts/convert_npz_to_npy.py -s experiments/laion_ldm_cocostuff_layout_no_caption/epoch_00059_plms_100_5.0/raw_tensor
