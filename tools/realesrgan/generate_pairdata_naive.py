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


src_dir = "../../data/InterHand2.6M_30fps/merge/various_poses"
gt_dir = "datasets/HandNeRF/various_poses_gt"
lq_dir = "datasets/HandNeRF/various_poses"
training_view = ["400002", "400004", "400010", "400012", "400013", "400016", "400018", "400042", "400053", "400059"]

os.system(f"rm -rf {gt_dir}")
os.system(f"rm -rf {lq_dir}")
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(lq_dir, exist_ok=True)

for img_dir in tqdm(sorted(glob.glob(os.path.join(src_dir, "cam*")))):
    if img_dir[-6:] in training_view:
        for img_path in tqdm(sorted(glob.glob(os.path.join(img_dir, "*.jpg"))), leave=False):
            # img_name = os.path.basename(img_path)
            img_name = "_".join(img_path.split("/")[-2:])
            img = cv2.imread(img_path)
            mask = cv2.imread(
                os.path.join("/".join(img_path.split("/")[:-2]), "mask", "/".join(img_path.split("/")[-2:])).replace(
                    ".jpg", ".png"
                )
            )
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(mask.copy(), kernel)
            msk_dilate = cv2.dilate(mask.copy(), kernel)
            mask[(msk_dilate - msk_erode) == 1] = 1
            img[mask == 0] = 0
            img = pad_img(img, 512)
            cv2.imwrite(os.path.join(gt_dir, img_name), img)
            img_lq = cv2.GaussianBlur(img, (5, 5), 0, 0)
            img_lq = cv2.resize(img, (256, 256))
            cv2.imwrite(os.path.join(lq_dir, img_name), img_lq)
