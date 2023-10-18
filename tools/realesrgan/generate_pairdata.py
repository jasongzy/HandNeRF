import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def pad_img(img: np.ndarray, output_size: int = 512) -> np.ndarray:
    pad_size = output_size - np.array(img.shape[:2])
    assert (pad_size >= 0).all()
    img_new = cv2.copyMakeBorder(img, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    assert img_new.shape == (output_size, output_size, 3)
    return img_new


task = "new"
exp_name = "0-ROM02_complete"
src_dir = f"../../data/result/{task}/{exp_name}/comparison"
src_dir_gt = f"../../data/result/{task}/{exp_name}/comparison_full"
gt_dir = "datasets/HandNeRF/ROM02_complete_gt"
lq_dir = "datasets/HandNeRF/ROM02_complete"

os.system(f"rm -rf {gt_dir}")
os.system(f"rm -rf {lq_dir}")
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(lq_dir, exist_ok=True)

for img_path in tqdm(sorted(glob.glob(os.path.join(src_dir, "*.png")))):
    img_name = os.path.basename(img_path)
    if img_name.endswith("_gt.png"):
        # os.system(f"ln -s {img_path} {os.path.join(gt_dir, img_name.replace('_gt', ''))}")
        # os.system(f"cp {img_path} {os.path.join(gt_dir, img_name.replace('_gt', ''))}")
        pass
    else:
        # os.system(f"ln -s {img_path} {lq_dir}")
        # os.system(f"cp {img_path} {lq_dir}")
        img = cv2.imread(img_path)
        # img = pad_img(img, 256)
        img = pad_img(img, max(img.shape))
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(os.path.join(lq_dir, img_name), img)


for img_path in tqdm(sorted(glob.glob(os.path.join(src_dir_gt, "*.png")))):
    img_name = os.path.basename(img_path)
    if img_name.endswith("_gt.png"):
        # os.system(f"ln -s {img_path} {os.path.join(gt_dir, img_name.replace('_gt', ''))}")
        # os.system(f"cp {img_path} {os.path.join(gt_dir, img_name.replace('_gt', ''))}")
        img = cv2.imread(img_path)
        img = pad_img(img, 512)
        cv2.imwrite(os.path.join(gt_dir, img_name.replace("_gt", "")), img)
    else:
        pass
