import glob
import os.path as osp
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.models.realesrgan_model import RealESRGANModel
from basicsr.utils import make_exp_dirs
from basicsr.utils.img_util import img2tensor, imwrite, tensor2img
from utils import parse_options

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, args = parse_options(root_path, is_train=False)
    opt["root_path"] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    # resume_state = load_resume_state(opt)
    model: RealESRGANModel = build_model(opt)
    model.load_network(
        model.net_g,
        f"experiments/{opt['name']}/models/net_g_latest.pth",
    )

    # mkdir for experiments and logger
    make_exp_dirs(opt)

    # create train and validation dataloaders
    dataset_opt = opt["datasets"]["val"]
    val_set = build_dataset(dataset_opt)
    val_loader = build_dataloader(
        val_set, dataset_opt, num_gpu=opt["num_gpu"], dist=opt["dist"], sampler=None, seed=opt["manual_seed"]
    )

    # model.validation(val_loader, 0, None, opt["val"]["save_img"])
    # for val_data in tqdm(val_loader):
    img_dir = "../../data/result/new/0-ROM03/Capture0-0025_three_count_small/comparison"
    for img_path in tqdm(sorted(glob.glob(osp.join(img_dir, "*.png")))):
        img_name = osp.splitext(osp.basename(img_path))[0]
        save_img_path = osp.join("test", opt["name"], f"{img_name}.png")
        if "_gt" in img_name:
            continue

        img = cv2.imread(img_path)
        pad_size = 256 - np.array(img.shape[:2])
        img = cv2.copyMakeBorder(img, 0, pad_size[0], 0, pad_size[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        imwrite(img, save_img_path.replace(".png", "_raw.png"))
        img = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
        val_data = {"lq": img.unsqueeze(0), "lq_path": [img_path]}

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])
        if "gt" in visuals:
            gt_img = tensor2img([visuals["gt"]])
            del model.gt

        # tentative for out of GPU memory
        del model.lq
        del model.output
        torch.cuda.empty_cache()

        imwrite(sr_img, save_img_path)
