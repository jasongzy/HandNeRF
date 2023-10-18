from dataclasses import dataclass

import numpy as np
import torch

from .nerf_net_utils import raw2outputs
from .tpose_renderer import Renderer as TRenderer
from lib.config import cfg
from lib.networks.bw_deform.interhands_network import Network as TNetwork


@dataclass(frozen=True)
class IHRay:
    ray_o: torch.Tensor
    ray_d: torch.Tensor
    near_left: torch.Tensor
    far_left: torch.Tensor
    near_right: torch.Tensor
    far_right: torch.Tensor
    radii: torch.Tensor = None


class Renderer:
    def __init__(self, net: TNetwork):
        self.net = net
        self.Left = TRenderer(net.Left)
        self.Right = TRenderer(net.Right)

    def merge_hands_pixel(self, batch, ret):
        """Deprecated"""
        H, W = batch["left"]["H"].item(), batch["left"]["W"].item()
        img_pred = np.zeros((H, W, 3))
        img_pred_left = np.zeros((H, W, 3))
        depth_full_left = np.zeros((H, W))
        img_pred_right = np.zeros((H, W, 3))
        depth_full_right = np.zeros((H, W))
        rgb_pred_left = ret["left"]["rgb_map"][0].detach().cpu().numpy()
        depth_left = ret["left"]["depth_map"][0].detach().cpu().numpy()
        mask_at_box_left = batch["left"]["mask_at_box"][0].detach().cpu().numpy().reshape(H, W)
        rgb_pred_right = ret["right"]["rgb_map"][0].detach().cpu().numpy()
        depth_right = ret["right"]["depth_map"][0].detach().cpu().numpy()
        mask_at_box_right = batch["right"]["mask_at_box"][0].detach().cpu().numpy().reshape(H, W)

        mask_both = mask_at_box_left & mask_at_box_right
        mask_left_only = mask_at_box_left & ~mask_at_box_right
        mask_right_only = ~mask_at_box_left & mask_at_box_right

        img_pred_left[mask_at_box_left] = rgb_pred_left
        depth_full_left[mask_at_box_left] = depth_left
        threshold = 30
        depth_full_left[
            np.where(
                (img_pred_left[:, :, 0] <= threshold / 255)
                & (img_pred_left[:, :, 1] <= threshold / 255)
                & (img_pred_left[:, :, 2] <= threshold / 255)
            )
        ] = 1e5
        img_pred_right[mask_at_box_right] = rgb_pred_right
        depth_full_right[mask_at_box_right] = depth_right
        depth_full_right[
            np.where(
                (img_pred_right[:, :, 0] <= threshold / 255)
                & (img_pred_right[:, :, 1] <= threshold / 255)
                & (img_pred_right[:, :, 2] <= threshold / 255)
            )
        ] = 1e5

        # depth_pred = np.minimum(depth_full_left, depth_full_right)
        # depth_pred[np.where((depth_full_left[:, :] == 1e5))] = 0

        mask_left_front = mask_both & (depth_full_left <= depth_full_right)
        mask_right_front = mask_both & (depth_full_left > depth_full_right)

        img_pred[mask_left_only] = img_pred_left[mask_left_only]
        img_pred[mask_left_front] = img_pred_left[mask_left_front]
        img_pred[mask_right_only] = img_pred_right[mask_right_only]
        img_pred[mask_right_front] = img_pred_right[mask_right_front]

        return img_pred

    def get_nerf_outputs(self, ray: IHRay, batch, is_rhand: bool):
        """Wrapper of renderer.get_wsampling_points() & get_density_color() for one hand.
        Pixels are aligned (NaN as mask) for both hands.
        """
        if is_rhand:
            renderer = self.Right
            near = ray.near_right
            far = ray.far_right
            batch = batch["right"]
        else:
            renderer = self.Left
            near = ray.near_left
            far = ray.far_left
            batch = batch["left"]

        n_batch, n_pixel = near.shape
        n_sample = cfg.N_samples

        mask = ~torch.isnan(near)
        rgb_dim = cfg.rgb_dim if cfg.use_neural_renderer else 3
        if cfg.use_both_renderer:
            rgb_dim += 3
        raw_full = torch.zeros([n_batch, n_pixel, n_sample, rgb_dim + 1], dtype=near.dtype, device=near.device)
        z_vals_full = torch.zeros(
            [n_batch, n_pixel, n_sample + 1 if cfg.encoding == "mip" else n_sample],
            dtype=near.dtype,
            device=near.device,
        )
        pbw = torch.zeros([n_batch, 0, cfg.joints_num], device=near.device)
        tbw = torch.zeros([n_batch, 0, cfg.joints_num], device=near.device)
        if mask.any():
            wpts, z_vals = renderer.get_wsampling_points(
                ray.ray_o[mask].unsqueeze(0),
                ray.ray_d[mask].unsqueeze(0),
                near[mask].unsqueeze(0),
                far[mask].unsqueeze(0),
                ray.radii[mask].unsqueeze(0) if cfg.encoding == "mip" else None,
            )
            if cfg.encoding == "mip":
                wpts, covs = wpts
            raw_decoder = lambda wpts_val, viewdir_val, dists_val: renderer.net.forward(
                wpts_val, viewdir_val, dists_val, batch
            )
            ret = renderer.get_density_color(
                (wpts, covs) if cfg.encoding == "mip" else wpts,
                ray.ray_d[mask].unsqueeze(0),
                z_vals,
                raw_decoder,
            )
            if cfg.use_alpha_sdf in ("always", "residual"):
                from lib.utils.sdf_utils import get_sdf, sdf_to_alpha_hard

                sdf = get_sdf(batch["vertices"].squeeze(0), batch["faces"].squeeze(0), wpts)
                alpha_sdf = sdf_to_alpha_hard(sdf)
                if cfg.use_alpha_sdf == "residual" and not cfg.test_novel_pose:
                    ret["raw"][..., -1:] = ret["raw"][..., -1:] + alpha_sdf.reshape(n_batch, -1, 1)
                else:
                    ret["raw"][..., -1:] = alpha_sdf.reshape(n_batch, -1, 1)
            raw_full[mask] = ret["raw"].reshape(n_batch, -1, n_sample, rgb_dim + 1)
            z_vals_full[mask] = z_vals
            if "pbw" in ret:
                pbw = ret["pbw"].view(n_batch, -1, cfg.joints_num)
            if "tbw" in ret:
                tbw = ret["tbw"].view(n_batch, -1, cfg.joints_num)

        return raw_full, z_vals_full, pbw, tbw

    def get_pixel_value(self, ray: IHRay, batch) -> dict[str, torch.Tensor]:
        n_batch, n_pixel = ray.near_left.shape
        n_sample = cfg.N_samples

        raw_left, z_vals_left, pbw_left, tbw_left = self.get_nerf_outputs(ray, batch, is_rhand=False)
        raw_right, z_vals_right, pbw_right, tbw_right = self.get_nerf_outputs(ray, batch, is_rhand=True)

        raw = torch.cat([raw_left, raw_right], dim=2)
        raw = raw.view(-1, 2 * n_sample, raw.shape[-1])

        if cfg.encoding == "mip":
            z_vals_left = 0.5 * (z_vals_left[..., :-1] + z_vals_left[..., 1:])
            z_vals_right = 0.5 * (z_vals_right[..., :-1] + z_vals_right[..., 1:])
        z_vals = torch.cat([z_vals_left, z_vals_right], dim=2)
        z_vals = z_vals.view(-1, 2 * n_sample)
        z_vals, indices = torch.sort(z_vals, dim=-1, descending=False, stable=True)
        raw = torch.gather(raw, 1, indices.repeat(raw.shape[-1], 1, 1).permute(1, 2, 0))
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        if cfg.exact_depth:
            R = batch["R_w2c"][0].float()
            T = batch["T_w2c"][0].float()
            depth_map = ray.ray_o + (depth_map[..., None] * ray.ray_d)
            depth_map = torch.matmul(depth_map, R.T) + T.T
            depth_map = depth_map[..., -1]

        ret = {
            "rgb_map": rgb_map,
            "acc_map": acc_map,
            "depth_map": depth_map,
            "raw": raw.view(n_batch, -1, raw.shape[-1]),
        }

        if self.net.training and cfg.learn_bw and not cfg.aninerf_animation:
            pbw = torch.cat([pbw_left, pbw_right], dim=1)
            tbw = torch.cat([tbw_left, tbw_right], dim=1)
            ret.update({"pbw": pbw, "tbw": tbw})

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def render(self, batch, is_separate=False) -> dict[str, torch.Tensor]:
        if is_separate:  # training
            ret_left = self.Left.render(batch["left"])
            torch.cuda.empty_cache()
            ret_right = self.Right.render(batch["right"])
            ret = {"left": ret_left, "right": ret_right}
            # img_pred = self.merge_hands_pixel(batch, ret)
            return ret

        # volume rendering for each pixel
        n_pixel = batch["ray_o"].shape[1]
        CHUNK = cfg.chunk
        ret_list = []
        for i in range(0, n_pixel, CHUNK):
            ray_data = {}
            for k in ("ray_o", "ray_d"):
                ray_data[k] = batch[k][:, i : i + CHUNK]
            for k in ("near", "far"):
                for hand in ("left", "right"):
                    ray_data[f"{k}_{hand}"] = batch[hand][k][:, i : i + CHUNK]
            if cfg.encoding == "mip":
                ray_data["radii"] = batch["radii"][:, i : i + CHUNK]
            ray = IHRay(**ray_data)
            pixel_value = self.get_pixel_value(ray, batch)
            ret_list.append(pixel_value)
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in ret_list[0].keys()}

        if cfg.use_both_renderer:
            rgb_map, rgb_feat_map = torch.split(ret["rgb_map"], (3, cfg.rgb_dim), dim=-1)
        if cfg.use_neural_renderer:
            if cfg.neural_renderer_type == "eg3d_sr":
                batch["rgb_map_nerf"] = rgb_map
                batch["poses"] = (batch["left"]["poses"] + batch["right"]["poses"]) * 0.5
            if cfg.use_both_renderer:
                ret["rgb_map"] = self.Right.neural_rendering(rgb_feat_map, batch)
            else:
                ret["rgb_map"] = self.Right.neural_rendering(ret["rgb_map"], batch)
        if cfg.use_sr != "none" and cfg.sr_ratio != 1:
            ret["rgb_map"] = self.Right.super_resolution(ret["rgb_map"], batch)
        return ret
