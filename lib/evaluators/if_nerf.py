import os
import warnings

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance

from lib.config import cfg
from lib.networks.detector.intaghand import calculate_hand_error, get_model
from lib.utils.img_utils import enlarge_box, fill_img, np2torch_img, pad_img_to_square


class Evaluator:
    def __init__(self):
        self.metric_names = ("mse", "psnr", "ssim", "lpips", "depth_l1", "fid", "mpjpe", "mpvpe")
        self.metrics: dict[str, list] = {item: [] for item in self.metric_names}

        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
        self.lpips_module = lpips.LPIPS(net="alex", verbose=False)
        # warnings.filterwarnings("default", category=UserWarning)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
        self.pose_detector = get_model().to(self.device)

    def clear_all(self):
        for item in self.metrics.values():
            item.clear()
        self.fid.reset()

    def get_results(self):
        self.metrics["fid"] = [self.fid.compute().item()]
        return self.metrics

    def merge_results(self, results_list: list):
        if isinstance(results_list[0], self.__class__):
            for evaluator in results_list:
                for k, v in self.metrics.items():
                    v.extend(getattr(evaluator, k))
        else:
            assert isinstance(results_list[0], dict)
            for results in results_list:
                for k, v in self.metrics.items():
                    v.extend(results[k])

    @staticmethod
    def save_img_pairs(img_pred, img_gt, batch, dirname="comparison", preprocess=True):
        if preprocess:
            # RGB to BGR, float to uint8
            img_pred = img_pred[..., [2, 1, 0]] * 255
            img_gt = img_gt[..., [2, 1, 0]] * 255
        result_dir = os.path.join(cfg.result_dir, dirname)
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch["frame_index"].item()
        view_index = batch["cam_ind"].item()
        cv2.imwrite(f"{result_dir}/frame{frame_index:04d}_view{view_index:04d}.png", img_pred)
        cv2.imwrite(f"{result_dir}/frame{frame_index:04d}_view{view_index:04d}_gt.png", img_gt)

    @classmethod
    def get_depth_plot(cls, depth_map):
        import io

        import matplotlib.pyplot as plt
        from PIL import Image

        fig = plt.figure()
        plt.axis("off")
        plt.imshow(depth_map)
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format="png", bbox_inches="tight", pad_inches=0)
        io_buf.seek(0)
        depth_plot = np.array(Image.open(io_buf))
        io_buf.close()
        plt.close("all")
        return depth_plot

    @staticmethod
    def psnr_metric(img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    @classmethod
    def ssim_metric(cls, img_pred, img_gt, mask=None):
        if mask is not None:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            img_pred = img_pred[y : y + h, x : x + w]
            img_gt = img_gt[y : y + h, x : x + w]
        ssim = structural_similarity(img_pred, img_gt, channel_axis=-1, data_range=1.0)
        return ssim

    def lpips_metric(self, img_pred, img_gt, mask=None, pad_to_square=False):
        if mask is not None:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            img_pred = img_pred[y : y + h, x : x + w]
            img_gt = img_gt[y : y + h, x : x + w]
        if pad_to_square:
            img_pred = pad_img_to_square(img_pred)
            img_gt = pad_img_to_square(img_gt)
        lpips_value = self.lpips_module(np2torch_img(img_pred), np2torch_img(img_gt), normalize=True)
        lpips_value = float(lpips_value.squeeze().data)

        return lpips_value

    @staticmethod
    def depth_metric(depth_pred, depth_gt):
        depth_l1 = np.mean(np.abs(depth_pred.copy(), depth_gt.copy()))
        return depth_l1

    def joint_metric(self, img_pred, img_gt, mask=None, hand_type="right"):
        if mask is not None:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            x, y, w, h = enlarge_box((x, y, w, h), ratio=1.2, img_shape=img_pred.shape[:2])
            img_pred = img_pred[y : y + h, x : x + w]
            img_gt = img_gt[y : y + h, x : x + w]
        img_pred = cv2.resize(pad_img_to_square(img_pred), (256, 256))
        img_gt = cv2.resize(pad_img_to_square(img_gt), (256, 256))
        img_input = np.stack((img_pred, img_gt), axis=0)
        result, paramsDict, handDictList, otherInfo = self.pose_detector(np2torch_img(img_input, device=self.device))
        v3d_left, v3d_right = result["verts3d"]["left"], result["verts3d"]["right"]
        if hand_type == "right":
            joint_error, vert_error = calculate_hand_error(v3d_right[:1], v3d_right[1:], hand_type, self.device)
        elif hand_type == "left":
            joint_error, vert_error = calculate_hand_error(v3d_left[:1], v3d_left[1:], hand_type, self.device)
        else:
            joint_error_left, vert_error_left = calculate_hand_error(
                v3d_left[:1], v3d_left[1:], hand_type="left", device=self.device
            )
            joint_error_right, vert_error_right = calculate_hand_error(
                v3d_right[:1], v3d_right[1:], hand_type="right", device=self.device
            )
            joint_error = (joint_error_left + joint_error_right) / 2
            vert_error = (vert_error_left + vert_error_right) / 2
        return joint_error, vert_error

    def evaluate_(self, output, batch, plot_depth=False, joint_metrics=True):
        H, W = batch["H"].item(), batch["W"].item()
        mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
        mask_at_box = mask_at_box.reshape(H, W)
        rgb_pred = output["rgb_map"][0].detach().cpu().numpy()
        rgb_gt = batch["rgb"][0].detach().cpu().numpy()
        if "rgb_sr" in batch:
            rgb_gt = batch["rgb_sr"][0].detach().cpu().numpy()
        if rgb_gt.sum() == 0:
            return

        # mse = np.mean((rgb_pred - rgb_gt) ** 2)
        # psnr = self.psnr_metric(rgb_pred, rgb_gt)

        if "rgb_sr" in batch:
            # H_sr, W_sr = batch["H_sr"].item(), batch["W_sr"].item()
            mask_sr = batch["msk_sr"][0].detach().cpu().numpy() != 0
            img_pred = fill_img(rgb_pred, mask_sr)
            img_gt = fill_img(rgb_gt, mask_sr)
        else:
            img_pred = fill_img(rgb_pred, mask_at_box)
            img_gt = fill_img(rgb_gt, mask_at_box)

        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = self.psnr_metric(img_pred, img_gt)
        ssim = self.ssim_metric(img_pred, img_gt)
        lpips = self.lpips_metric(img_pred, img_gt)

        depth_pred = output["depth_map"][0].detach().cpu().numpy()
        depth_gt = batch["depth"][0].detach().cpu().numpy()
        depth_l1 = self.depth_metric(depth_pred, depth_gt)
        depth_map_pred = fill_img(depth_pred, mask_at_box)
        depth_map_gt = fill_img(depth_gt, mask_at_box)

        if joint_metrics:
            mpjpe, mpvpe = self.joint_metric(
                img_pred,
                img_gt,
                mask=batch.get("msk_sr", batch["msk"])[0].detach().cpu().numpy(),
                hand_type=cfg.test_dataset.hand_type,
            )
        else:
            mpjpe, mpvpe = None, None

        results = {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "depth_l1": depth_l1,
            "mpjpe": mpjpe,
            "mpvpe": mpvpe,
        }
        results.update(
            {"img_pred": img_pred, "img_gt": img_gt, "depth_map_pred": depth_map_pred, "depth_map_gt": depth_map_gt}
        )

        if plot_depth:
            depth_plot_pred = self.get_depth_plot(depth_map_pred)
            depth_plot_gt = self.get_depth_plot(depth_map_gt)
            results.update({"depth_plot_pred": depth_plot_pred, "depth_plot_gt": depth_plot_gt})

        return results

    def evaluate(self, output, batch, save_imgs=True, save_depth=False):
        results = self.evaluate_(output, batch, plot_depth=save_depth)
        if results is None:
            return
        # if self.writer is not None and step is not None:
        #     for k in self.metric_names:
        #         self.writer.add_scalar(f"test/{k}", results[k], step + 1)

        for k, v in self.metrics.items():
            if k in results:
                v.append(results[k])
        self.fid.update(np2torch_img(pad_img_to_square(results["img_gt"]), device=self.fid.device), real=True)
        self.fid.update(np2torch_img(pad_img_to_square(results["img_pred"]), device=self.fid.device), real=False)

        if save_imgs:
            self.save_img_pairs(results["img_pred"], results["img_gt"], batch)
        if save_depth:
            depth_plot_pred = cv2.cvtColor(results["depth_plot_pred"], cv2.COLOR_BGR2RGB)
            depth_plot_gt = cv2.cvtColor(results["depth_plot_gt"], cv2.COLOR_BGR2RGB)
            self.save_img_pairs(depth_plot_pred, depth_plot_gt, batch, dirname="depth", preprocess=False)

    def summarize(self, writer: SummaryWriter = None):
        result_dir = cfg.result_dir
        print(colored(f"the results are saved at {result_dir}", "yellow"))

        self.get_results()  # to compute FID

        metrics_mean = {k: np.mean(self.metrics[k]) for k in self.metric_names}
        print("PSNR: {:.4f}".format(metrics_mean["psnr"]))
        print("SSIM: {:.4f}".format(metrics_mean["ssim"]))
        print("LPIPS: {:.4f}".format(metrics_mean["lpips"]))
        print("Depth_L1: {:.4f}".format(metrics_mean["depth_l1"]))
        print("FID: {:.4f}".format(metrics_mean["fid"]))
        print("MPJPE: {:.4f}".format(metrics_mean["mpjpe"]))
        print("MPVPE: {:.4f}".format(metrics_mean["mpvpe"]))

        result_path = os.path.join(cfg.result_dir, "metrics.npy")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        np.save(result_path, {"mean": metrics_mean} | self.metrics)

        if writer is not None:
            for k in self.metric_names:
                if k in ("mse", "fid"):
                    continue
                for i, v in enumerate(self.metrics[k]):
                    writer.add_scalar(f"test/{k}", v, i)
            writer.add_hparams(
                hparam_dict=cfg.to_hparams_dict(),
                metric_dict={f"test-mean/{k}": v for k, v in metrics_mean.items()},
                run_name="hparams",
            )

        self.clear_all()
