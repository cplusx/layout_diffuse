import os
import numpy as np
import argparse
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', type=str, default='', help='source images directory')
args = parser.parse_args()

indir = args.src
outdir = indir+'_npy'
os.makedirs(outdir, exist_ok=True)

npz_files = glob.glob(indir + '/*.npz')
print(len(npz_files))
for npz_file in tqdm(npz_files):
    out_path = npz_file.replace(indir, outdir)
    out_path = out_path.replace('npz', 'npy')
    image = np.load(npz_file)['image']

    with open(out_path, 'wb') as OUT:
        np.save(OUT, image*255)