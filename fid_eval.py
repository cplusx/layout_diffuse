import argparse
import os
from cleanfid import fid
from PIL import ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 10

def resize_a_folder(folder_path, size):
    os.system(f'python scripts/resize_images.py --indir {folder_path} --size {size}')
    return folder_path + f'-{size}'

def convert_a_folder_to_jpg(folder_path):
    os.system(f'python scripts/convert_jpg.py --indir {folder_path}')
    return folder_path + f'-jpg'

def evaluate(args):
    src_path = args.src
    if args.cvt_jpg_s:
        src_path = convert_a_folder_to_jpg(src_path)
    if args.resize_s:
        src_path = resize_a_folder(args.src, args.target_size)

    dst_path = args.dst
    if args.cvt_jpg_d:
        dst_path = convert_a_folder_to_jpg(dst_path)
    if args.resize_d:
        dst_path = resize_a_folder(args.dst, args.target_size)

    fid_score = fid.compute_fid(src_path, dst_path)
    print('FID of : {}'.format(fid_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, default='', help='Ground truth images directory')
    parser.add_argument('--resize_s', action='store_true')
    parser.add_argument('--cvt_jpg_s', action='store_true')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')
    parser.add_argument('--resize_d', action='store_true')
    parser.add_argument('--cvt_jpg_d', action='store_true')
    parser.add_argument('--target_size', type=int, default=256)
   
    ''' parser configs '''
    args = parser.parse_args()

    evaluate(args)