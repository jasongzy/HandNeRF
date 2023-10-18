from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .nerf_net_utils import raw2outputs
from lib.config import cfg
from lib.networks.bw_deform.tpose_nerf_network import Network as TNetwork
from lib.networks.mipnerf import sample_along_rays
from lib.utils.img_utils import fill_img_torch


@dataclass(frozen=True)
class Ray:
    ray_o: torch.Tensor
    ray_d: torch.Tensor
    near: torch.Tensor
    far: torch.Tensor
    occupancy: torch.Tensor = None
    radii: torch.Tensor = None
    tnear: torch.Tensor = None
    tfar: torch.Tensor = None


class Renderer:
    def __init__(self, net: TNetwork):
        self.net = net

        if cfg.use_sr == "swinir":
            from lib.networks.sr.swinir import define_model, infer_img  # fmt: skip
            assert cfg.sr_ratio <= 4.0, "Maximum sr_ratio for SwinIR: 4.0"
            self.sr_model = define_model(
                model_path="lib/networks/sr/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                task="real_sr",
                scale=4,
                training_patch_size=64,
                large_model=True,
            )
            self.sr_infer = lambda x: infer_img(self.sr_model.eval(), x, scale=4, window_size=8)
        elif cfg.use_sr == "realesrgan":
            from lib.networks.sr.realesrgan import define_model, infer_img  # fmt: skip
            assert cfg.sr_ratio <= 2.0, "Maximum sr_ratio for Real-ESRGAN: 2.0"
            self.sr_model = define_model(pth_name=cfg.sr_ckpt)
            self.sr_infer = lambda x: infer_img(self.sr_model, x)
        elif cfg.use_sr != "none":
            raise ValueError(f"Invalid sr model: {cfg.use_sr}")

    @torch.no_grad()
    def get_wsampling_points(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        radii: torch.Tensor = None,
    ):
        if cfg.encoding == "mip":
            assert radii is not None
            z_vals, samples = sample_along_rays(
                ray_o.view(-1, 3),
                ray_d.view(-1, 3),
                radii.view(-1, 1),
                cfg.N_samples,
                near.view(-1, 1),
                far.view(-1, 1),
                randomized=cfg.perturb > 0.0 and self.net.training,
                lindisp=False,
                ray_shape="cone",
                diag=True,
            )
            n_batch = ray_o.shape[0]
            z_vals = z_vals.view(n_batch, -1, cfg.N_samples + 1)
            pts, covs = samples
            pts = pts.view(n_batch, -1, cfg.N_samples, 3)
            covs = covs.view(n_batch, -1, cfg.N_samples, 3)
            pts = (pts, covs)
        else:
            # calculate the steps for each ray
            t_vals = torch.linspace(0.0, 1.0, steps=cfg.N_samples).to(near)
            z_vals: torch.Tensor = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals
            if cfg.perturb > 0.0 and self.net.training:
                # get intervals between samples
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(upper)
                z_vals = lower + (upper - lower) * t_rand
            pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        if cfg.encoding == "mip":
            wpts, covs = wpts
            covs = covs.view(-1, 3)
        viewdir = viewdir[:, :, None].repeat(1, 1, wpts.shape[2], 1).contiguous()
        viewdir = viewdir.view(-1, 3)
        wpts = wpts.view(-1, 3)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        if cfg.encoding != "mip":
            dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(-1)

        ret = raw_decoder((wpts, covs) if cfg.encoding == "mip" else wpts, viewdir, dists)

        return ret

    def get_pixel_value(self, ray: Ray, batch) -> dict[str, torch.Tensor]:
        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray.ray_o, ray.ray_d, ray.near, ray.far, ray.radii)
        if cfg.encoding == "mip":
            wpts, covs = wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        # compute the color and density
        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net.forward(wpts_val, viewdir_val, dists_val, batch)
        # viewing direction = ray_d (has been normalized in the dataset)
        ret = self.get_density_color((wpts, covs) if cfg.encoding == "mip" else wpts, ray.ray_d, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        raw = ret["raw"].reshape(-1, n_sample, ret["raw"].shape[-1])
        if cfg.use_alpha_sdf in ("always", "residual") or (
            cfg.use_alpha_sdf == "train_init"
            and self.net.training
            and batch["iter_step"] <= batch["max_iter_step"] // 16 + 1  # first 25 epoch for max_epoch=400
        ):
            from lib.utils.sdf_utils import get_sdf, sdf_to_alpha_hard

            sdf = get_sdf(batch["vertices"].squeeze(0), batch["faces"].squeeze(0), wpts)
            alpha_sdf = sdf_to_alpha_hard(sdf)
            if cfg.use_alpha_sdf == "residual" and not cfg.test_novel_pose:
                raw[..., -1:] = raw[..., -1:] + alpha_sdf.reshape(-1, n_sample, 1)
                alpha_predict = raw[..., -1:]
            else:
                alpha_predict = raw[..., -1:].clone()
                raw[..., -1:] = alpha_sdf.reshape(-1, n_sample, 1)
        else:
            alpha_sdf = None
        ret.update({"z_vals": z_vals.view(n_batch, n_pixel, -1)})
        if cfg.encoding == "mip":
            z_vals = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
        z_vals = z_vals.view(-1, n_sample)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, cfg.white_bkgd)

        if self.net.training and alpha_sdf is not None:
            ret.update({"alpha_sdf": alpha_sdf.view(n_batch, -1, 1), "alpha": alpha_predict.view(n_batch, -1, 1)})
        if self.net.training and cfg.depth_loss == "gnll":
            depth_var = torch.sum(weights * (z_vals - depth_map.unsqueeze(-1)) ** 2, -1) + 1e-5
            depth_var = depth_var.view(n_batch, n_pixel)
            ret.update({"depth_var": depth_var})

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        weights = weights.view(n_batch, n_pixel, -1)
        depth_map = depth_map.view(n_batch, n_pixel)

        if cfg.exact_depth:
            R = batch["R_w2c"][0].float()
            T = batch["T_w2c"][0].float()
            depth_map = ray.ray_o + (depth_map[..., None] * ray.ray_d)
            depth_map = torch.matmul(depth_map, R.T) + T.T
            depth_map = depth_map[..., -1]

        ret.update(
            {
                "rgb_map": rgb_map,
                "acc_map": acc_map,
                "weights": weights,
                "depth_map": depth_map,
                "raw": raw.view(n_batch, -1, ret["raw"].shape[-1]),
            }
        )

        if "pbw" in ret:
            pbw = ret["pbw"].view(n_batch, -1, cfg.joints_num)
            ret.update({"pbw": pbw})
        if "tbw" in ret:
            tbw = ret["tbw"].view(n_batch, -1, cfg.joints_num)
            ret.update({"tbw": tbw})

        if self.net.training and cfg.use_hard_surface_loss_canonical:
            wpts_can, z_vals_can = self.get_wsampling_points(ray.ray_o, ray.ray_d, ray.tnear, ray.tfar, ray.radii)
            if cfg.encoding == "mip":
                wpts_can, covs_can = wpts_can
            raw_decoder_can = lambda wpts_val, viewdir_val, dists_val: self.net.forward(
                wpts_val, viewdir_val, dists_val, batch, is_canonical=True
            )
            torch.cuda.empty_cache()
            ret_can = self.get_density_color(
                (wpts_can, covs_can) if cfg.encoding == "mip" else wpts_can, ray.ray_d, z_vals_can, raw_decoder_can
            )
            raw_can = ret_can["raw"].reshape(-1, n_sample, ret_can["raw"].shape[-1])
            if cfg.encoding == "mip":
                z_vals_can = 0.5 * (z_vals_can[..., :-1] + z_vals_can[..., 1:])
            z_vals_can = z_vals_can.view(-1, n_sample)
            _, _, acc_map_can, weights_can, _ = raw2outputs(raw_can, z_vals_can, cfg.white_bkgd)
            ret.update(
                {
                    "acc_map_can": acc_map_can.view(n_batch, n_pixel),
                    "weights_can": weights_can.view(n_batch, n_pixel, -1),
                }
            )

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def neural_rendering(self, rgb_feat_map: torch.Tensor, batch) -> torch.Tensor:
        H, W = batch["H"].item(), batch["W"].item()
        if "coord" in batch.keys():
            coord = batch["coord"].squeeze()
        else:  # eval/test
            mask_at_box = batch["mask_at_box"]
            coord = torch.argwhere(mask_at_box.reshape(H, W) == 1)

        if cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn", "cnn_sr"):
            feat_map = fill_img_torch(rgb_feat_map, coord, H, W)
            rgb_map = self.net.neural_renderer(feat_map)
        elif cfg.neural_renderer_type == "spconv":
            feat_map = rgb_feat_map.to(coord.device).view(-1, rgb_feat_map.shape[-1])
            rgb_map = self.net.neural_renderer(feat_map, coord, (H, W))
        elif cfg.neural_renderer_type == "transformer":
            rgb_map = self.net.neural_renderer(
                rgb_feat_map.to(coord.device),
                coord.unsqueeze(0).float(),
                num_neighbors=int(cfg.N_rand / 50),
                window_size=10,
            )
        elif cfg.neural_renderer_type == "transcnn":
            feat_map = self.net.neural_renderer.transformer(
                rgb_feat_map.to(coord.device),
                coord.unsqueeze(0).float(),
                num_neighbors=int(cfg.N_rand / 50),
                window_size=10,
            )
            feat_map = fill_img_torch(feat_map, coord, H, W)
            rgb_map = self.net.neural_renderer.cnn(feat_map)
        elif cfg.neural_renderer_type == "eg3d_sr":
            rgb_map = fill_img_torch(batch["rgb_map_nerf"], coord, H, W)
            feat_map = fill_img_torch(rgb_feat_map, coord, H, W)
            rgb_map = self.net.neural_renderer(rgb_map, feat_map, batch["poses"][0, 1:, :].flatten())

        if cfg.neural_renderer_type in ("cnn", "spconv", "transcnn"):
            rgb_map = rgb_map.squeeze().permute(1, 2, 0)
            rgb_map = rgb_map[coord[:, 0], coord[:, 1]].unsqueeze(0)
            rgb_map = rgb_map.to(rgb_feat_map.device)
            # rgb_vis = torch.zeros((H, W, 3), device=coord.device)
            # rgb_vis[coord[:, 0], coord[:, 1]] = rgb_map
            # rgb_vis = rgb_vis.detach().cpu().numpy()
        elif cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr"):
            rgb_map = rgb_map.squeeze().permute(1, 2, 0)
            if "msk_sr" in batch:
                rgb_map = rgb_map[batch["msk_sr"].squeeze() != 0]
            rgb_map = rgb_map.unsqueeze(0)

        return rgb_map

    def super_resolution(self, rgb_map: torch.Tensor, batch) -> torch.Tensor:
        H, W = batch["H"].item(), batch["W"].item()
        H_sr, W_sr = batch["H_sr"].item(), batch["W_sr"].item()
        mask_at_box = batch["mask_at_box"]
        coord = torch.argwhere(mask_at_box.reshape(H, W) == 1)
        rgb_map_full = fill_img_torch(rgb_map, coord, H, W)
        # if cfg.use_sr == "realesrgan":
        #     rgb_map_full_left = rgb_map_full.clone() * batch["left"]["msk"]
        #     rgb_map_full_right = rgb_map_full.clone() * batch["right"]["msk"]
        #     rgb_map_full = torch.concat([rgb_map_full_left, rgb_map_full_right])
        #     rgb_map_full = self.sr_infer(rgb_map_full)
        #     rgb_map_full_left = rgb_map_full[:1] * batch["left"]["msk_sr"]
        #     rgb_map_full_right = rgb_map_full[1:] * batch["right"]["msk_sr"]
        #     rgb_map_full = rgb_map_full_left + rgb_map_full_right
        rgb_map_full = self.sr_infer(rgb_map_full)
        if rgb_map_full.shape[2:] != (H_sr, W_sr):
            rgb_map_full = F.interpolate(rgb_map_full, size=(H_sr, W_sr), mode="area")
        rgb_map_full = rgb_map_full.squeeze().permute(1, 2, 0)
        rgb_map = rgb_map_full[batch["msk_sr"].squeeze() != 0]
        rgb_map = rgb_map.unsqueeze(0)
        rgb_map = torch.clamp(rgb_map, min=0, max=1)
        return rgb_map

    def render(self, batch) -> dict[str, torch.Tensor]:
        # volume rendering for each pixel
        n_pixel = batch["ray_o"].shape[1]
        CHUNK = cfg.chunk
        ret_list = []
        for i in range(0, n_pixel, CHUNK):
            ray_data = {}
            for k in ("ray_o", "ray_d", "near", "far"):
                ray_data[k] = batch[k][:, i : i + CHUNK]
            if cfg.encoding == "mip":
                ray_data["radii"] = batch["radii"][:, i : i + CHUNK]
            if self.net.training and cfg.use_hard_surface_loss_canonical:
                for k in ("tnear", "tfar"):
                    ray_data[k] = batch[k][:, i : i + CHUNK]
            ray = Ray(**ray_data)
            pixel_value = self.get_pixel_value(ray, batch)
            ret_list.append(pixel_value)
            # torch.cuda.empty_cache()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in ret_list[0].keys()}

        if cfg.use_both_renderer or (self.net.training and cfg.use_distill and not cfg.use_neural_renderer):
            rgb_map, rgb_feat_map = torch.split(ret["rgb_map"], (3, cfg.rgb_dim), dim=-1)
            if cfg.use_distill:
                ret["rgb_map"] = rgb_map
                ret["rgb_feat_map"] = rgb_feat_map

        if cfg.use_neural_renderer:
            if cfg.neural_renderer_type == "eg3d_sr":
                batch["rgb_map_nerf"] = rgb_map
            if cfg.use_both_renderer:
                ret["rgb_map_nerf"] = rgb_map.clone()
                ret["rgb_map"] = self.neural_rendering(rgb_feat_map.detach(), batch)
            else:
                if cfg.use_distill:
                    ret["rgb_feat_map"] = ret["rgb_map"]
                ret["rgb_map"] = self.neural_rendering(ret["rgb_map"], batch)

        if not self.net.training and cfg.use_sr != "none" and cfg.sr_ratio != 1:
            ret["rgb_map"] = self.super_resolution(ret["rgb_map"], batch)

        return ret
