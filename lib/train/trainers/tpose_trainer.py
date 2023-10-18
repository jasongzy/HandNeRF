import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.networks.bw_deform.tpose_nerf_network import Network as TNetwork
from lib.networks.renderer.tpose_renderer import Renderer as TRenderer
from lib.utils.img_utils import fill_img_torch, pad_img_to_square_torch
from lib.utils.net_utils import loss_sanity_check


class NetworkWrapper(nn.Module):
    def __init__(self, net: TNetwork):
        super().__init__()

        self.net = net
        self.renderer = TRenderer(self.net)

        self.bw_crit = F.smooth_l1_loss
        self.img2mse = F.mse_loss

        if cfg.use_perceptual_loss:
            from lib.losses import VGGPerceptualLoss

            self.perceptual_loss = VGGPerceptualLoss(resize=True)

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

        if cfg.use_alpha_loss and cfg.use_alpha_sdf not in ("train_init", "always", "residual"):
            raise ValueError("alpha_loss can only be used when use_alpha_sdf=train_init or always or residual")

    @staticmethod
    def get_decay_coef(iter_step: int, end_step: int, start_step: int = 0):
        if iter_step < start_step or iter_step > end_step:
            decay_coef = 0
        else:
            decay_coef = 1 - ((iter_step - 1 - start_step) / end_step)
            if decay_coef < 0:
                decay_coef = 0
            elif decay_coef > 1:
                decay_coef = 1
        return decay_coef

    def get_loss(self, batch, ret):
        scalar_stats = {}
        loss = 0

        if cfg.prior_knowledge_loss_decay:
            iter_decay_coef = self.get_decay_coef(batch["iter_step"], batch["max_iter_step"])
        else:
            iter_decay_coef = 1.0
        if cfg.use_depth:
            weight_depth = cfg.weight_depth * iter_decay_coef
        if cfg.use_mask_loss:
            weight_mask = cfg.weight_mask * iter_decay_coef
        if cfg.use_color_var_loss:
            weight_color_var = cfg.weight_color_var * self.get_decay_coef(
                batch["iter_step"], batch["max_iter_step"] / 2, start_step=2000
            )
        if cfg.use_alpha_loss:
            weight_alpha = cfg.weight_alpha * iter_decay_coef

        mask = batch["mask_at_box"]
        if cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr"):
            img_loss = self.img2mse(ret["rgb_map"], batch["rgb_sr"])
        else:
            img_loss = self.img2mse(ret["rgb_map"][mask], batch["rgb"][mask])
        scalar_stats["img_loss"] = img_loss
        loss += img_loss

        if cfg.use_alpha_loss and "alpha" in ret:
            alpha_loss = F.mse_loss(ret["alpha"], ret["alpha_sdf"])
            scalar_stats["alpha_loss"] = alpha_loss
            loss += weight_alpha * alpha_loss

        if "pbw" in ret:
            bw_loss = self.bw_crit(ret["pbw"], ret["tbw"])
            scalar_stats["bw_loss"] = bw_loss
            loss += cfg.weight_bw * bw_loss

        if cfg.use_perceptual_loss:
            if cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr"):
                fill_and_pad = lambda x: pad_img_to_square_torch(
                    fill_img_torch(
                        x, torch.argwhere(batch["msk_sr"].squeeze(0) == 1), batch["H_sr"].item(), batch["W_sr"].item()
                    )
                )
                perceptual_loss = self.perceptual_loss(fill_and_pad(ret["rgb_map"]), fill_and_pad(batch["rgb_sr"]))
            else:
                fill_and_pad = lambda x: pad_img_to_square_torch(
                    fill_img_torch(x, batch["coord"].squeeze(), batch["H"].item(), batch["W"].item())
                )
                perceptual_loss = self.perceptual_loss(fill_and_pad(ret["rgb_map"]), fill_and_pad(batch["rgb"]))
            scalar_stats["perceptual_loss"] = perceptual_loss
            loss += cfg.weight_perceptual * perceptual_loss

        if cfg.use_depth:
            if cfg.depth_loss == "gnll":
                depth_loss = self.depth_crit(ret["depth_map"], batch["depth"], ret["depth_var"])
            else:
                depth_loss = self.depth_crit(ret["depth_map"], batch["depth"])
            scalar_stats["depth_loss"] = depth_loss
            loss += weight_depth * depth_loss

        if cfg.learn_displacement:
            displ_loss = torch.norm(ret["displ"], dim=2).mean()
            scalar_stats["displ_loss"] = displ_loss
            loss += cfg.weight_displ * displ_loss

        if cfg.use_viewdir and cfg.use_color_range_loss:
            color_range_loss = F.mse_loss(ret["rgb_dummy"], ret["raw"][..., :-1])
            scalar_stats["color_range_loss"] = color_range_loss
            loss += cfg.weight_color_range * color_range_loss

        if cfg.use_distill:
            distill_loss = F.mse_loss(ret["rgb_feat_map"], F.normalize(batch["feat_pretrain"], dim=-1))
            scalar_stats["distill_loss"] = distill_loss
            loss += cfg.weight_distill * distill_loss

        if cfg.use_both_renderer:
            img_loss_nerf = self.img2mse(ret["rgb_map_nerf"][mask], batch["rgb"][mask])
            scalar_stats["img_loss_nerf"] = img_loss_nerf
            loss += img_loss_nerf

        if cfg.use_mask_loss:
            mask_loss = F.mse_loss(
                torch.clamp(ret["acc_map"][mask], min=0.0, max=1.0),
                batch["mask_ray"][mask].float(),
            )
            scalar_stats["mask_loss"] = mask_loss
            loss += weight_mask * mask_loss

        if cfg.use_sparsity_loss:
            sparsity_loss = torch.mean(torch.abs((ret["raw"][..., -1])))  # apply L1 regularization to density (alpha)
            scalar_stats["sparsity_loss"] = sparsity_loss
            loss += cfg.weight_sparsity * sparsity_loss

        if cfg.use_distortion_loss:
            from lib.losses import DistortionLoss

            distortion_loss_fn = DistortionLoss(is_mip=cfg.encoding == "mip")
            distortion_loss = distortion_loss_fn(
                ret["weights"].unsqueeze(-1), ret["z_vals"], batch["near"].unsqueeze(-1), batch["far"].unsqueeze(-1)
            ).mean()
            scalar_stats["distortion_loss"] = distortion_loss
            loss += cfg.weight_distortion * distortion_loss

        if cfg.use_hard_surface_loss or cfg.use_hard_surface_loss_canonical:
            penalize_01 = lambda x: torch.mean(-torch.log(torch.exp(-torch.abs(x)) + torch.exp(-torch.abs(1 - x))))
            HARD_SURFACE_OFFSET = -float(penalize_01(torch.Tensor([0])))  # log(1 + 1/e)
            if cfg.use_hard_surface_loss:
                hard_surface_loss = penalize_01(ret["weights"][mask]) + HARD_SURFACE_OFFSET
                scalar_stats["hard_surface_loss"] = hard_surface_loss
                loss += cfg.weight_hard_surface * hard_surface_loss
            if cfg.use_hard_surface_loss_canonical:
                hard_surface_loss_can = penalize_01(ret["weights_can"]) + HARD_SURFACE_OFFSET
                sharp_edge_loss_can = penalize_01(ret["acc_map_can"]) + HARD_SURFACE_OFFSET
                hard_surface_loss_can += sharp_edge_loss_can
                scalar_stats["hard_surface_loss_can"] = hard_surface_loss_can
                loss += cfg.weight_hard_surface * hard_surface_loss_can

        if cfg.use_color_var_loss:
            # color_var_loss = -torch.log(torch.tanh(torch.var(ret["rgb_map"][mask]) * 100))
            color_var_loss = F.smooth_l1_loss(torch.var(ret["rgb_map"][mask]), torch.var(batch["rgb"][mask]))
            scalar_stats["color_var_loss"] = color_var_loss
            loss += weight_color_var * color_var_loss

        loss_sanity_check(scalar_stats)
        scalar_stats["loss"] = loss
        return loss, scalar_stats

    def forward(self, batch):
        ret = self.renderer.render(batch)
        loss, scalar_stats = self.get_loss(batch, ret)
        image_stats = {}
        return ret, loss, scalar_stats, image_stats
