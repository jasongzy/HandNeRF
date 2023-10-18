import argparse
import glob
import os

import imageio.v2 as imageio
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--capture", default=0, type=int)
    parser.add_argument("--cam", default=400002, type=int)
    args = parser.parse_args()

    data_root = os.path.join(args.basedir, "images", args.split, f"Capture{args.capture}")
    assert os.path.exists(data_root), f"Wrong path: {data_root}"
    print(f"Generating GIFs for {args.split}/Capture{args.capture} ...")

    for seq in tqdm(sorted(glob.glob(os.path.join(data_root, "*"))), dynamic_ncols=True):
        imgs = [imageio.imread(img) for img in sorted(glob.glob(os.path.join(seq, f"cam{args.cam}", "*.jpg")))]
        result_dir = os.path.join(args.basedir, "gif", args.split, f"Capture{args.capture}", seq.split("/")[-1])
        os.makedirs(result_dir, exist_ok=True)
        imageio.mimwrite(os.path.join(result_dir, f"cam{args.cam}.gif"), imgs, duration=1000 / 30)
    print("Done!")
