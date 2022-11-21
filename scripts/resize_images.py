import argparse
import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(img_path, save_path, size, mode):
    print('save image to ', save_path)
    img = Image.open(img_path)
    img = img.resize((size, size), mode)
    img = img.save(save_path)

def read_resize_and_save(img_path, save_path, size, mode=Image.BICUBIC):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if img_path.endswith('.png') or img_path.endswith('.jpg'):
        process_image(img_path, save_path, size, mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str)
    parser.add_argument('--size', type=int)
   
    ''' parser configs '''
    args = parser.parse_args()
    size = args.size

    in_dir = args.indir
    out_dir = os.path.join(
        os.path.dirname(in_dir),
        os.path.basename(in_dir) + f'-{size}'
    )

    image_names = glob.glob(in_dir + '/*.jpg') + glob.glob(in_dir + '/*.png') + glob.glob(in_dir + '/**/*.jpg') + glob.glob(in_dir + '/**/*.png')

    for image_name in tqdm(image_names):
        save_img_path = image_name.replace(in_dir, out_dir)
        if image_name.endswith('.jpg'):
            save_img_path = save_img_path.replace('.jpg', '.png')
        if os.path.exists(save_img_path):
            continue
        try:
            read_resize_and_save(image_name, save_img_path, size, mode=Image.BICUBIC)
        except:
            print(image_name, 'is broken')