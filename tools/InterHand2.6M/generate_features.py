import argparse
import os
import sys
from glob import glob

sys.path.append("./dino")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.decomposition import PCA
from tqdm import tqdm

from dino.vision_transformer import vit_small

torch.set_grad_enabled(False)


class Dataset(data.Dataset):
    def __init__(self, data_root, use_mask=True):
        super().__init__()

        self.data_root = data_root
        self.use_mask = use_mask
        self.img_path = sorted(glob(os.path.join(data_root, "cam*", "*.jpg")))
        with open(os.path.join(data_root, "cams.txt"), "r") as f:
            cam_list = f.readlines()
        self.img_path = [x for x in self.img_path if any(f"cam{cam.strip()}" in x for cam in cam_list)]

    def get_mask(self, index):
        mask_path = self.img_path[index]
        mask_path = mask_path.replace(self.data_root, os.path.join(self.data_root, "mask"))
        mask_path = f"{mask_path[:-4]}.png"
        mask = cv2.imread(mask_path)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask != 0).astype(np.uint8)
        return mask

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path[index])
        file_dir, file_name = os.path.split(img_path)
        file_name = os.path.join(os.path.split(file_dir)[-1], file_name)
        img = cv2.imread(img_path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_mask:
            mask = self.get_mask(index)
            img[mask == 0] = 0
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        return img, file_name

    def __len__(self):
        return len(self.img_path)


def calc_pca(emb, dim=256):
    X = emb.flatten(0, -2).cpu().numpy()
    np.random.seed(6)
    pca = PCA(n_components=dim)
    pca.fit(X)
    X_pca = pca.transform(X).reshape(*emb.shape[:2], dim)
    X_pca = torch.from_numpy(X_pca).to(emb.device)
    return X_pca


def convT(feat: torch.Tensor, H=512, W=334, patch_size=8):
    """
    Args:
        feat: [B, H_feat * W_feat, C]
    Returns:
        [B, H, W, C]
    """
    C = feat.shape[-1]
    feat = feat.transpose(1, 2).contiguous()
    H_feat = H // patch_size
    W_feat = W // patch_size
    feat = feat.reshape(-1, C, H_feat, W_feat)
    kernel = torch.ones((C, 1, patch_size, patch_size), device=feat.device)
    feat_full = F.conv_transpose2d(
        feat, kernel, stride=patch_size, output_padding=(H - H_feat * patch_size, W - W_feat * patch_size), groups=C
    )
    assert feat_full.shape[1:] == (C, H, W)
    feat_full = feat_full.permute(0, 2, 3, 1).contiguous()
    return feat_full


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--capture", default=0, type=int)
    parser.add_argument("--seq", default="0051_dinosaur", type=str)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pth", default="dino/dino_deitsmall8_pretrain.pth", type=str)
    parser.add_argument("--no_mask", action="store_true", default=False)
    parser.add_argument("--clear_existed", action="store_true", default=False)
    args = parser.parse_args()

    data_root = os.path.join(args.basedir, "images", args.split, f"Capture{args.capture}", args.seq)
    assert os.path.exists(data_root), f"Wrong path: {data_root}"
    print(f"Generating image features for {args.split}/Capture{args.capture}/{args.seq} ...")

    FEAT_DIR = "dino"
    if args.clear_existed:
        os.system(f"rm -rf {os.path.join(data_root, FEAT_DIR)}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = Dataset(data_root, use_mask=~args.no_mask)
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = vit_small(patch_size=8, num_classes=0)
    model.load_state_dict(torch.load(args.pth, map_location="cpu"), strict=True)
    model = model.to(device)

    for batch, names in tqdm(data_loader, dynamic_ncols=True):
        batch = batch.to(device)
        # feat_batch = model(batch)
        feat_batch = model.get_intermediate_layers(batch, layer_indices=[1])
        feat_batch = feat_batch[0][:, 1:]
        feat_batch = calc_pca(feat_batch, dim=256)
        feat_batch = convT(feat_batch, H=512, W=334, patch_size=8)
        # feat_batch = feat_batch.half()
        feat_batch = feat_batch.cpu().numpy()
        for feat, name in zip(feat_batch, names):
            save_path = os.path.join(data_root, FEAT_DIR, os.path.split(name)[0])
            os.makedirs(save_path, exist_ok=True)
            # np.save(os.path.join(save_path, os.path.split(name)[-1].replace(".jpg", ".npy")), feat)
            np.savez_compressed(os.path.join(save_path, os.path.split(name)[-1].replace(".jpg", ".npz")), feat)

    print("Done!")
