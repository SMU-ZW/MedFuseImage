#!/usr/bin/python

# import thread
import time
from PIL import Image
import glob
from tqdm import tqdm
import os

print('starting')
data_dir = '/projects/eclarson/stems/STEMC/EHR/MIMIC/physionet.org/content/mimic-cxr-jpg/get-zip'
version = '2.1.0'

paths_done = glob.glob(f'{data_dir}/{version}/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0/resized/**/*.jpg', recursive = True)
print('done', len(paths_done))

paths_all = glob.glob(f'{data_dir}/{version}/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0/files/**/*.jpg', recursive = True)
print('all', len(paths_all))



done_files = [os.path.basename(path) for path in paths_done]

paths = [path for path in paths_all if os.path.basename(path) not in done_files ]
print('left', len(paths))

file_root = f'{data_dir}/{version}/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0/files'
resize_root = f'{data_dir}/{version}/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0/resized'

def resize_images(path):
    try:
        basewidth = 512
        img = Image.open(path)

        rel_path = os.path.relpath(path, file_root)

        save_path = os.path.join(resize_root, rel_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Resize
        wpercent = basewidth / float(img.size[0])
        hsize = int(float(img.size[1]) * wpercent)
        img = img.resize((basewidth, hsize))

        # Save
        img.save(save_path)

        print(f"✅ Saved: {save_path}")
        
    except Exception as e:
        print(f"❌ Failed: {path} -> {e}")


from multiprocessing.dummy import Pool as ThreadPool

threads = 10

for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i: i+threads]
    pool = ThreadPool(len(paths_subset))
    pool.map(resize_images, paths_subset)
    pool.close()
    pool.join()
