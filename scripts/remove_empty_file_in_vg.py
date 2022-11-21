import os
from PIL import Image
from glob import glob

VG_DIR = '/home/ubuntu/disk2/data/VG/images'
# VG_DIR = 'experiments/laion_ldm_cocostuff_layout_caption_v9/epoch_00059_plms_200_5.0/sampled_256_cropped_224'
image_paths = glob(VG_DIR+'/**/*.jpg') + glob(VG_DIR+'/**/*.png')

for path in image_paths:
    try:
        Image.open(path)
    except:
        print(f'{path} failed, remove it')
        os.system(f'rm {path}')