import glob
import os.path as osp
import pickle
import warnings

import torch
import torch.nn.functional as F
import yaml

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
from basicsr.models import build_model
from basicsr.models.realesrgan_model import RealESRGANModel
from basicsr.utils.options import ordered_yaml, parse_options

warnings.filterwarnings("default", category=UserWarning)

ROOT_PATH = osp.abspath(osp.dirname(__file__))


def define_model(pth_name="net_g_latest.pth") -> RealESRGANModel:
    # opt, args = parse_options(ROOT_PATH, is_train=False)
    # with open(osp.join(ROOT_PATH, "finetune_realesrgan_x2plus_pairdata.yml"), mode="r") as f:
    #     opt = yaml.load(f, Loader=ordered_yaml()[0])
    # opt["is_train"] = False
    with open(osp.join(ROOT_PATH, "opt.pkl"), "rb") as f:
        opt = pickle.load(f)
    opt["root_path"] = ROOT_PATH
    opt["model_type"] = "RealESRGANModel_basicsr"
    opt["path"]["pretrain_network_g"] = None

    model: RealESRGANModel = build_model(opt)
    model.load_network(model.net_g, osp.join(ROOT_PATH, pth_name))
    return model


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

    val_data = {"lq": img}
    model.feed_data(val_data)
    model.test()
    sr_img = model.output

    if new_size:
        sr_ratio = 2
        sr_img = sr_img[:, :, : H * sr_ratio, : W * sr_ratio]
    if unsqueezed:
        sr_img = sr_img.squeeze(0)
    return sr_img


if __name__ == "__main__":
    import cv2
    import numpy as np
    from basicsr.utils.img_util import img2tensor, imwrite, tensor2img
    from tqdm import tqdm

    model = define_model()

    img_dir = osp.join(ROOT_PATH, "../../../data/result/new/0-ROM03/Capture0-0025_three_count/comparison")
    for img_path in tqdm(sorted(glob.glob(osp.join(img_dir, "*.png"))), dynamic_ncols=True):
        img_name = osp.splitext(osp.basename(img_path))[0]
        save_img_path = osp.join(ROOT_PATH, "test", f"{img_name}.png")
        if "_gt" in img_name:
            continue

        img = cv2.imread(img_path)
        # pad_size = 256 - np.array(img.shape[:2])
        # img = cv2.copyMakeBorder(img, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        imwrite(img, save_img_path.replace(".png", "_raw.png"))
        img = img2tensor(img / 255.0, bgr2rgb=True, float32=True)

        sr_img = infer_img(model, img.cuda())

        sr_img = tensor2img(sr_img)
        imwrite(sr_img, save_img_path)
