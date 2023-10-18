import torch
import torch.nn as nn
import torch.nn.functional as F

from .aninerf_animation_trainer import NetworkWrapper as ATrainer
from .tpose_trainer import NetworkWrapper as TTrainer
from lib.config import cfg
from lib.networks.bw_deform.interhands_network import Network as IHNetwork
from lib.networks.renderer.interhands_renderer import Renderer as IHRenderer


class NetworkWrapper(nn.Module):
    def __init__(self, net: IHNetwork):
        super().__init__()

        self.net = net
        self.renderer = IHRenderer(self.net)
        if cfg.aninerf_animation:
            self.Left = ATrainer(self.net.Left)
            self.Right = ATrainer(self.net.Right)
        else:
            self.Left = TTrainer(self.net.Left)
            self.Right = TTrainer(self.net.Right)

    def forward(self, batch):
        if not cfg.aninerf_animation or (cfg.use_depth or cfg.learn_displacement):
            ret = self.renderer.render(batch, is_separate=True)
        else:
            ret = {"left": None, "right": None}
        loss = 0

        loss_left, scalar_stats_left = self.Left.get_loss(batch["left"], ret["left"])
        loss_right, scalar_stats_right = self.Right.get_loss(batch["right"], ret["right"])

        loss = loss_left + loss_right
        scalar_stats = {k: scalar_stats_left[k] + scalar_stats_right[k] for k in scalar_stats_left.keys()}

        if cfg.use_hands_align_loss:
            assert not cfg.hands_share_params
            hands_align_loss = torch.zeros_like(loss)
            for hand in ("left", "right"):
                trainer = self.Right if hand == "left" else self.Left
                weights = ret[hand]["weights"]
                raw = ret[hand]["raw"].reshape(*weights.shape, -1)
                rgb = raw[..., :-1]
                # alpha = raw[..., -1:]
                alpha = ret[hand]["tsamples_alpha"].reshape(*weights.shape)
                select_mask = weights == torch.max(weights, dim=-1)[0].unsqueeze(-1)
                select_mask &= weights > 0
                select_mask &= alpha >= 0.5
                select_mask &= (rgb > (50 / 255)).any(-1)
                coord = batch[hand]["coord"].squeeze()
                msk = batch[hand]["msk"].squeeze()
                ray_mask = msk[coord[:, 0], coord[:, 1]] != 0
                select_mask &= ray_mask.unsqueeze(-1).unsqueeze(0)
                if select_mask.sum().item() == 0:
                    continue
                tsamples_xyz = ret[hand]["tsamples_xyz"].reshape(*weights.shape, -1)
                tsamples_xyz = tsamples_xyz[select_mask].unsqueeze(0)
                if cfg.encoding == "mip":
                    tsamples_covs = ret[hand]["tsamples_covs"].reshape(*weights.shape, -1)
                    tsamples_covs = tsamples_covs[select_mask].unsqueeze(0)
                tsamples_viewdir = ret[hand]["tsamples_viewdir"].reshape(*weights.shape, -1)
                tsamples_viewdir = tsamples_viewdir[select_mask].unsqueeze(0)
                latent_index = batch[hand]["latent_index"]
                tbounds = batch[hand]["tbounds"]
                if hand == "left":
                    tbounds[:, :, 0] = -tbounds[:, :, 0]
                    tbounds[:, 0, 0], tbounds[:, 1, 0] = tbounds[:, 1, 0].clone(), tbounds[:, 0, 0].clone()
                _, rgb_other, _ = trainer.renderer.net.tpose_human.calculate_density_rgb(
                    (tsamples_xyz, tsamples_covs) if cfg.encoding == "mip" else tsamples_xyz,
                    tsamples_viewdir,
                    latent_index,
                    bounds=tbounds,
                )
                rgb_other = trainer.renderer.net.rgb_actvn(rgb_other)
                hands_align_loss += F.mse_loss(rgb_other, rgb[select_mask].T.unsqueeze(0))
            scalar_stats["hands_align_loss"] = hands_align_loss
            loss += cfg.weight_hands_align * hands_align_loss

        return ret, loss, scalar_stats, {}
