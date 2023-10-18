import argparse
import datetime
import logging
import os
import os.path as osp
import random
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.data.realesrgan_paired_dataset import RealESRGANPairedDataset
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.models import build_model
from basicsr.train import create_train_val_dataloader, init_tb_loggers, load_resume_state
from basicsr.utils import (
    AvgTimer,
    FileClient,
    MessageLogger,
    get_env_info,
    get_root_logger,
    get_time_str,
    imfrombytes,
    img2tensor,
    make_exp_dirs,
    mkdir_and_rename,
    set_random_seed,
)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import _postprocess_yml_value, copy_opt_file, dict2str, ordered_yaml
from basicsr.utils.registry import DATASET_REGISTRY
from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize

# from basicsr.train import train_pipeline


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, default="finetune_realesrgan_x2plus_pairdata.yml", help="Path to option YAML file."
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="pytorch", help="job launcher")
    parser.add_argument("--auto_resume", default=True, action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--force_yml", nargs="+", default=None, help="Force to update yml files. Examples: train:ema_decay=0.999"
    )
    args = parser.parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        args.launcher = "none"

    # parse yml to dict
    with open(args.opt, mode="r") as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # distributed settings
    if args.launcher == "none":
        opt["dist"] = False
        if opt["num_gpu"] == "auto":
            opt["num_gpu"] = 1
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)
    opt["rank"], opt["world_size"] = get_dist_info()

    # random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split("=")
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = "opt"
            for key in keys.split(":"):
                eval_str += f'["{key}"]'
            eval_str += "=value"
            # using exec function
            exec(eval_str)

    opt["auto_resume"] = args.auto_resume
    opt["is_train"] = is_train

    # debug setting
    if args.debug and not opt["name"].startswith("debug"):
        opt["name"] = "debug_" + opt["name"]

    if opt["num_gpu"] == "auto":
        opt["num_gpu"] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt["datasets"].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        if "scale" in opt:
            dataset["scale"] = opt["scale"]
        if dataset.get("dataroot_gt") is not None:
            dataset["dataroot_gt"] = osp.expanduser(dataset["dataroot_gt"])
        if dataset.get("dataroot_lq") is not None:
            dataset["dataroot_lq"] = osp.expanduser(dataset["dataroot_lq"])

    # paths
    for key, val in opt["path"].items():
        if (val is not None) and ("resume_state" in key or "pretrain_network" in key):
            opt["path"][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, "experiments", opt["name"])
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_states"] = osp.join(experiments_root, "training_states")
        opt["path"]["log"] = experiments_root
        opt["path"]["visualization"] = osp.join(experiments_root, "visualization")

        # change some options for debug mode
        if "debug" in opt["name"]:
            if "val" in opt:
                opt["val"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(root_path, "results", opt["name"])
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root
        opt["path"]["visualization"] = osp.join(results_root, "visualization")

    return opt, args


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt["root_path"] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"] and opt["rank"] == 0:
            mkdir_and_rename(osp.join(opt["root_path"], "tb_logger", opt["name"]))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt["path"]["experiments_root"])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    # logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt["datasets"]["train"].get("prefetch_mode")
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f"Use {prefetch_mode} prefetch dataloader")
        if opt["datasets"]["train"].get("pin_memory") is not True:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt["train"].get("warmup_iter", -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_timer.get_avg_time(), "data_time": data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if opt.get("val") is not None and (current_iter % opt["val"]["val_freq"] == 0):
                if len(val_loaders) > 1:
                    logger.warning("Multiple validation datasets are *only* supported by SRModel.")
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get("val") is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"])
    if tb_logger:
        tb_logger.close()


class KeepRNG:
    def __init__(self, enable=True):
        self.enable = enable

    def __enter__(self):
        self.rng_state_np = np.random.get_state()
        self.rng_state_torch = torch.get_rng_state()
        # self.rng_state_torch_cuda = torch.cuda.get_rng_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            np.random.set_state(self.rng_state_np)
            torch.set_rng_state(self.rng_state_torch)
            # torch.cuda.set_rng_state(self.rng_state_torch_cuda)


@DATASET_REGISTRY.register(suffix="basicsr")
class RealESRGANPairedDatasetAug(RealESRGANPairedDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.color_jitter = transforms.ColorJitter(brightness=(0.7, 1.5), contrast=0.3, saturation=0.3, hue=0)

    def common_transform(self, img1, img2):
        transform_ = lambda img: np.array(self.color_jitter(Image.fromarray(np.uint8(img * 255)))) / 255.0
        # with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else None):
        with KeepRNG():
            img1 = transform_(img1)
        img2 = transform_(img2)
        return img1, img2

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        scale = self.opt["scale"]

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]["lq_path"]
        img_bytes = self.file_client.get(lq_path, "lq")
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"])
            # color jitter
            img_gt, img_lq = self.common_transform(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # from torchvision.utils import save_image
        # save_image(img_gt, "aug_gt.png")
        # save_image(img_lq, "aug.png")

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}


if __name__ == "__main__":
    import sys

    sys.argv.append("--launcher")
    sys.argv.append("none")
    seed = random.randint(1, 10000)
    opt, _ = parse_options(osp.abspath(osp.dirname(__file__)))
    set_random_seed(seed)

    img1 = np.array(Image.open("test.png")) / 255.0
    img2 = np.array(Image.open("test_gt.png")) / 255.0
    img1, img2 = RealESRGANPairedDatasetAug(opt["datasets"]["train"]).common_transform(img1, img2)
    Image.fromarray((img1 * 255).astype(np.uint8)).save("test_aug.png")
    Image.fromarray((img2 * 255).astype(np.uint8)).save("test_gt_aug.png")
