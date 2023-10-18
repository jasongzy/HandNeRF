import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.networks.bw_deform.tpose_nerf_network import Network as TNetwork
from lib.networks.renderer import tpose_renderer
from lib.utils.blend_utils import *
from lib.utils.net_utils import loss_sanity_check


class NetworkWrapper(nn.Module):
    def __init__(self, net: TNetwork):
        super().__init__()

        self.net = net
        self.renderer = tpose_renderer.Renderer(self.net)

        self.bw_crit = F.smooth_l1_loss
        self.img2mse = F.mse_loss
        if cfg.use_depth:
            if cfg.depth_loss == "l2":
                self.depth_crit = F.mse_loss
            elif cfg.depth_loss == "l1":
                self.depth_crit = F.l1_loss
            elif cfg.depth_loss == "smooth_l1":
                self.depth_crit = F.smooth_l1_loss
            elif cfg.depth_loss == "gnll":
                self.depth_crit = nn.GaussianNLLLoss(eps=1e-3)
            elif cfg.depth_loss == "pearson":
                from lib.losses import NegPearson

                self.depth_crit = NegPearson()
            else:
                raise ValueError(f"Invalid depth_loss: {cfg.depth_loss}")

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.novel_pose_deform.parameters():
            param.requires_grad = True

        if cfg.use_poses:
            if cfg.poses_layer_policy == "freeze":
                for param in self.net.novel_pose_deform.poses_layer.parameters():
                    param.requires_grad = False
            if cfg.learn_bw and cfg.bw_policy == "freeze":
                for param in self.net.novel_pose_deform.bw.parameters():
                    param.requires_grad = False

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)

    def get_loss(self, batch, ret=None):
        scalar_stats = {}
        loss = 0

        assert cfg.bw_use_P2T or cfg.bw_use_T2P or cfg.use_depth, "no loss can be used"

        if cfg.learn_bw:
            if cfg.bw_use_P2T:
                if cfg.bw_ray_sampling:
                    wpts, _ = self.renderer.get_wsampling_points(
                        batch["ray_o"], batch["ray_d"], batch["near"], batch["far"], batch.get("radii", None)
                    )
                    if cfg.encoding == "mip":
                        wpts, covs = wpts
                        covs = covs.view(1, -1, 3)
                    wpts = wpts.view(1, -1, 3)
                else:
                    assert cfg.encoding != "mip", "Mip-NeRF is not supported for non-ray sampling"
                    wpts = get_sampling_points(batch["wbounds"], cfg.N_rand * cfg.N_samples)
                ppts = world_points_to_pose_points(wpts, batch["R"], batch["Th"])
                pbw0, tbw0 = ppts_to_tpose(self.net, (ppts, covs) if cfg.encoding == "mip" else ppts, batch)
                bw_loss0 = self.bw_crit(pbw0, tbw0)
                loss += cfg.weight_bw * bw_loss0
                scalar_stats["bw_loss0"] = bw_loss0
                torch.cuda.empty_cache()

            if cfg.bw_use_T2P:
                if cfg.bw_ray_sampling:
                    tpts_w, _ = self.renderer.get_wsampling_points(
                        batch["ray_o"], batch["ray_d"], batch["tnear"], batch["tfar"], batch.get("radii", None)
                    )
                    if cfg.encoding == "mip":
                        tpts_w, covs = tpts_w
                        covs = covs.view(1, -1, 3)
                    tpts_w = tpts_w.view(1, -1, 3)
                    tpts = world_points_to_pose_points(tpts_w, batch["R"], batch["Th"])
                else:
                    tpts = get_sampling_points(batch["tbounds"], cfg.N_rand * cfg.N_samples)
                pbw1, tbw1 = tpose_to_ppts(self.net, (tpts, covs) if cfg.encoding == "mip" else tpts, batch)
                bw_loss1 = self.bw_crit(pbw1, tbw1)
                loss += cfg.weight_bw * bw_loss1
                scalar_stats["bw_loss1"] = bw_loss1
                torch.cuda.empty_cache()

        if cfg.use_depth:
            if cfg.depth_loss == "gnll":
                depth_loss = self.depth_crit(ret["depth_map"], batch["depth"], ret["depth_var"])
            else:
                depth_loss = self.depth_crit(ret["depth_map"], batch["depth"])
            scalar_stats["depth_loss"] = depth_loss
            loss += 0.1 * cfg.weight_depth * depth_loss
        if cfg.learn_displacement:
            displ_loss = torch.norm(ret["displ"], dim=2).mean()
            scalar_stats["displ_loss"] = displ_loss
            loss += cfg.weight_displ * displ_loss

        loss_sanity_check(scalar_stats)
        scalar_stats["loss"] = loss
        return loss, scalar_stats

    def forward(self, batch):
        if cfg.use_depth or cfg.learn_displacement:
            ret = self.renderer.render(batch)
            torch.cuda.empty_cache()
        else:
            ret = None
        loss, scalar_stats = self.get_loss(batch, ret)
        image_stats = {}
        return ret, loss, scalar_stats, image_stats


def ppts_to_tpose(net: TNetwork, pose_pts, batch):
    if cfg.encoding == "mip":
        pose_pts, covs = pose_pts

    # to filter bw pairs that satisfy:
    # 1. the sampled pose_pts are near to SMPL surface
    ppts_idx = net.get_idx_near_surface(pose_pts, batch["pbw"], batch["pbounds"])
    pose_pts = pose_pts[ppts_idx].unsqueeze(0)

    pbw = net.novel_pose_deform.get_bw(pose_pts, batch, is_tpose=False)
    tpose_pts = pose_points_to_tpose_points(pose_pts, pbw, batch["A"])
    tbw = net.deform.get_bw(tpose_pts, batch, is_tpose=True)

    with torch.no_grad():
        # 2. the deformed tpose_pts are inside tbounds
        inside = net.get_idx_inside_bounds(tpose_pts, batch["tbounds"])
        # 3. density of the deformed tpose_pts is large enough
        density = net.tpose_human.calculate_density(
            (tpose_pts, covs[ppts_idx].unsqueeze(0)) if cfg.encoding == "mip" else tpose_pts, bounds=batch["tbounds"]
        ).squeeze(1)
        density[~inside] = 0
        density_idx = net.get_idx_large_density(density)

    pbw = pbw.transpose(1, 2)[density_idx]
    tbw = tbw.transpose(1, 2)[density_idx]

    return pbw, tbw


def tpose_to_ppts(net: TNetwork, tpose_pts, batch):
    if cfg.encoding == "mip":
        tpose_pts, covs = tpose_pts

    # to filter bw pairs that satisfy:
    # 1. the sampled tpose_pts are near to SMPL surface
    tpts_idx = net.get_idx_near_surface(tpose_pts, batch["tbw"], batch["tbounds"])
    tpose_pts = tpose_pts[tpts_idx].unsqueeze(0)
    with torch.no_grad():
        # 2. density of the sampled tpose_pts is large enough
        density = net.tpose_human.calculate_density(
            (tpose_pts, covs[tpts_idx].unsqueeze(0)) if cfg.encoding == "mip" else tpose_pts, bounds=batch["tbounds"]
        ).squeeze(1)
        density_idx = net.get_idx_large_density(density)
    tpose_pts = tpose_pts[density_idx].unsqueeze(0)

    tbw = net.deform.get_bw(tpose_pts, batch, is_tpose=True)
    pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch["A"])
    pbw = net.novel_pose_deform.get_bw(pose_pts, batch, is_tpose=False)

    # 3. the deformed pose_pts are inside pbounds
    inside = net.get_idx_inside_bounds(pose_pts, batch["pbounds"])

    pbw = pbw.transpose(1, 2)[inside]
    tbw = tbw.transpose(1, 2)[inside]

    return pbw, tbw
