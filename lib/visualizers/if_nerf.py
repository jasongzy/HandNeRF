import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

from lib.config import cfg


class Visualizer:
    def __init__(self):
        result_dir = cfg.result_dir
        print(colored(f"the results are saved at {result_dir}", "yellow"))

    def visualize_image(self, output, batch):
        rgb_pred = output["rgb_map"][0].detach().cpu().numpy()
        rgb_gt = batch["rgb"][0].detach().cpu().numpy()
        # print('mse: {}'.format(np.mean((rgb_pred - rgb_gt)**2)))

        mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
        H, W = batch["H"].item(), batch["W"].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        result_dir = os.path.join(cfg.result_dir, "comparison")
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch["frame_index"].item()
        view_index = batch["cam_ind"].item()
        cv2.imwrite(f"{result_dir}/frame{frame_index:04d}_view{view_index:04d}.png", (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(f"{result_dir}/frame{frame_index:04d}_view{view_index:04d}_gt.png", (img_gt[..., [2, 1, 0]] * 255))

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()

    def visualize_normal(self, output, batch):
        mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
        H, W = batch["H"].item(), batch["W"].item()
        mask_at_box = mask_at_box.reshape(H, W)
        surf_mask = mask_at_box.copy()
        surf_mask[mask_at_box] = output["surf_mask"][0].detach().cpu().numpy()

        normal_map = np.zeros((H, W, 3))
        normal_map[surf_mask] = output["surf_normal"][output["surf_mask"]].detach().cpu().numpy()

        normal_map[..., 1:] = normal_map[..., 1:] * -1
        norm = np.linalg.norm(normal_map, axis=2)
        norm[norm < 1e-8] = 1e-8
        normal_map = normal_map / norm[..., None]
        normal_map = (normal_map + 1) / 2

        plt.imshow(normal_map)
        plt.show()

    def visualize_acc(self, output, batch):
        acc_pred = output["acc_map"][0].detach().cpu().numpy()

        mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        acc = np.zeros((H, W))
        acc[mask_at_box] = acc_pred

        plt.imshow(acc)
        plt.show()

        # acc_path = os.path.join(cfg.result_dir, 'acc')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # acc_path = os.path.join(acc_path, f'{i:04d}_{cam_ind:02d}.jpg')
        # os.makedirs(os.path.dirname(acc_path), exist_ok=True)
        # plt.savefig(acc_path)

    def visualize_depth(self, output, batch):
        depth_pred = output["depth_map"][0].detach().cpu().numpy()

        mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        depth = np.zeros((H, W))
        depth[mask_at_box] = depth_pred

        plt.imshow(depth)
        plt.show()

        # depth_path = os.path.join(cfg.result_dir, 'depth')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # depth_path = os.path.join(depth_path, f'{i:04d}_{cam_ind:02d}.jpg')
        # os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        # plt.savefig(depth_path)

    def visualize(self, output, batch):
        self.visualize_image(output, batch)
