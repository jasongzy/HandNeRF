import argparse
import glob
import os
import os.path as osp
import pickle
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
from basicsr.archs.arch_util import pixel_unshuffle
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.models import build_model
from basicsr.models.realesrgan_model import RealESRGANModel

warnings.filterwarnings("default", category=UserWarning)

ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), "../.."))

torch.set_grad_enabled(False)


class Dataset(data.Dataset):
    def __init__(self, data_root: str, use_mask=True):
        super().__init__()

        self.data_root = data_root
        self.use_mask = use_mask
        self.img_path = sorted(glob.glob(osp.join(data_root, "cam*", "*.jpg")))
        with open(os.path.join(data_root, "cams.txt"), "r") as f:
            cam_list = f.readlines()
        self.img_path = [x for x in self.img_path if any(f"cam{cam.strip()}" in x for cam in cam_list)]

    def get_mask(self, index: int):
        mask_path = self.img_path[index]
        mask_path = mask_path.replace(self.data_root, osp.join(self.data_root, "mask"))
        mask_path = f"{mask_path[:-4]}.png"
        mask = cv2.imread(mask_path)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask != 0).astype(np.uint8)
        return mask

    def __getitem__(self, index: int):
        img_path = osp.join(self.img_path[index])
        file_dir, file_name = osp.split(img_path)
        file_name = osp.join(osp.split(file_dir)[-1], file_name)
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


def define_model(pth_name: str = None) -> RealESRGANModel:
    with open(osp.join(ROOT_PATH, "lib/networks/sr", "opt.pkl"), "rb") as f:
        opt = pickle.load(f)
    # opt["root_path"] = ROOT_PATH
    opt["model_type"] = "RealESRGANModel_basicsr"
    # opt["path"]["pretrain_network_g"] = None
    model: RealESRGANModel = build_model(opt)
    if pth_name:
        model.load_network(model.net_g, osp.join(ROOT_PATH, pth_name))
    return model


def net_forward(self: RRDBNet, x: torch.Tensor):
    """Monkey patch for basicsr.archs.rrdbnet_arch.RRDBNet"""
    if self.scale == 2:
        feat = pixel_unshuffle(x, scale=2)
    elif self.scale == 1:
        feat = pixel_unshuffle(x, scale=4)
    else:
        feat = x
    feat = self.conv_first(feat)
    body_feat = self.conv_body(self.body(feat))
    feat = feat + body_feat
    # upsample
    feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
    # feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
    # out = self.conv_last(self.lrelu(self.conv_hr(feat)))
    # return out
    return feat


def infer_img(model: RealESRGANModel, img: torch.Tensor) -> torch.Tensor:
    """img: [B, 3, H, W] RGB values within [0, 1]"""
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        unsqueezed = True
    else:
        unsqueezed = False
    assert len(img.shape) == 4 and img.shape[1] == 3, f"Invalid image shape: {img.shape}"
    H, W = img.shape[2:]
    if H != W:
        new_size = max(H, W)
        img = F.pad(img, (0, new_size - W, 0, new_size - H), mode="constant")
        assert img.shape[2] == img.shape[3] == new_size
    else:
        new_size = None

    model.feed_data({"lq": img})
    # model.test()
    model.net_g.eval()
    feat = net_forward(model.net_g, model.lq)

    if new_size:
        sr_ratio = 1
        feat = feat[:, :, : H * sr_ratio, : W * sr_ratio]
    feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
    if unsqueezed:
        feat = feat.squeeze(0)
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--capture", default=0, type=int)
    parser.add_argument("--seq", default="0051_dinosaur", type=str)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pth",
        default="tools/realesrgan/experiments/finetune_ROM03_augment_complete/models/net_g_latest.pth",
        type=str,
    )
    parser.add_argument("--no_mask", action="store_true", default=False)
    parser.add_argument("--clear_existed", action="store_true", default=False)
    args = parser.parse_args()

    data_root = os.path.join(args.basedir, "images", args.split, f"Capture{args.capture}", args.seq)
    assert os.path.exists(data_root), f"Wrong path: {data_root}"
    print(f"Generating image features for {args.split}/Capture{args.capture}/{args.seq} ...")

    FEAT_DIR = "esrgan"
    if args.clear_existed:
        os.system(f"rm -rf {os.path.join(data_root, FEAT_DIR)}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = Dataset(data_root, use_mask=not args.no_mask)
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = define_model(pth_name=args.pth)

    for batch, names in tqdm(data_loader, dynamic_ncols=True):
        batch = batch.to(device)
        feat_batch = infer_img(model, batch)
        feat_batch = feat_batch.cpu().numpy()
        for feat, name in zip(feat_batch, names):
            save_path = os.path.join(data_root, FEAT_DIR, os.path.split(name)[0])
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, os.path.split(name)[-1].replace(".jpg", ".npy")), feat)
            # np.savez_compressed(os.path.join(save_path, os.path.split(name)[-1].replace(".jpg", ".npz")), feat)

    print("Done!")
