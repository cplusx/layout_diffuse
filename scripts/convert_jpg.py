import argparse
import os
import cv2
from tqdm import tqdm

def read_convert_and_save(img_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = cv2.imread(img_path)
    cv2.imwrite(save_path, image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str)
   
    ''' parser configs '''
    args = parser.parse_args()

    in_dir = args.indir
    out_dir = os.path.join(
        os.path.dirname(in_dir),
        os.path.basename(in_dir) + f'-jpg'
    )

    image_names = os.listdir(in_dir)

    for image_name in tqdm(image_names, desc='convert image to jpg'):
        if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
            continue
        img_path = os.path.join(in_dir, image_name)
        save_img_path = os.path.join(out_dir, image_name.replace('.png', '.jpg'))
        if os.path.exists(save_img_path):
            continue
        read_convert_and_save(img_path, save_img_path)