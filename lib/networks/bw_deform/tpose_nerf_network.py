import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.networks.embedder as embedder
from lib.config import cfg
from lib.utils.base_utils import normalize_batch
from lib.utils.blend_utils import (
    pose_dirs_to_tpose_dirs,
    pose_points_to_tpose_points,
    pts_sample_blend_weights,
    world_dirs_to_pose_dirs,
    world_points_to_pose_points,
)
from lib.utils.net_utils import ResMLP, adaptive_num_emb, load_network, remove_net_prefix


class Network(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.tpose_human = TPoseHuman()
        if "both" in (cfg.train_dataset.hand_type, cfg.test_dataset.hand_type) and cfg.hands_share_params:
            pose_num = 2 * cfg.num_train_frame + 1
        else:
            pose_num = cfg.num_train_frame + 1
        self.deform = DeformationField(pose_num=pose_num)

        if cfg.rgb_prior == [0, 0, 0]:
            self.rgb_actvn = torch.sigmoid
        else:
            self.register_buffer("rgb_prior", torch.Tensor(cfg.rgb_prior).reshape(3, 1))
            self.rgb_actvn = lambda rgb: torch.clamp(F.elu(rgb) + self.rgb_prior, min=0.0, max=1.0)
        if cfg.density_activation == "relu":
            self.density_actvn = F.relu
        elif cfg.density_activation == "softplus":
            self.density_actvn = F.softplus
        else:
            raise ValueError(f"Invalid density_activation: {cfg.density_activation}")
        self.density_bias: float = cfg.density_bias

        if cfg.use_neural_renderer:
            import lib.networks.renderer.neural_renderer as nr  # fmt: skip
            if cfg.neural_renderer_type == "cnn":
                self.neural_renderer = nr.CNNRenderer(cfg.rgb_dim)
            elif cfg.neural_renderer_type == "spconv":
                self.neural_renderer = nr.SPCNNRenderer(cfg.rgb_dim)
            elif cfg.neural_renderer_type == "transformer":
                self.neural_renderer = nr.TransRenderer(cfg.rgb_dim)
            elif cfg.neural_renderer_type == "transcnn":
                self.neural_renderer = nr.TransCNNRenderer(cfg.rgb_dim)
            elif cfg.neural_renderer_type == "cnn_sr":
                self.neural_renderer = nr.CNNRenderer_SR(cfg.rgb_dim, sr_ratio=cfg.sr_ratio)
            elif cfg.neural_renderer_type == "eg3d_sr":
                self.neural_renderer = nr.EG3D_SR(cfg.rgb_dim, sr_ratio=cfg.sr_ratio)

            else:
                raise ValueError(f"Invalid neural_renderer_type: {cfg.neural_renderer_type}")

        if cfg.aninerf_animation:
            if "both" in (cfg.train_dataset.hand_type, cfg.test_dataset.hand_type) and cfg.hands_share_params:
                novel_pose_num = 2 * cfg.num_eval_frame
            else:
                novel_pose_num = cfg.num_eval_frame
            self.novel_pose_deform = DeformationField(pose_num=novel_pose_num)

            if cfg.get("init_aninerf", "no_pretrain") != "no_pretrain":
                assert cfg.aninerf_animation
                init_model_path = cfg.trained_model_dir.replace(cfg.exp_name, cfg.init_aninerf)
                adaptive_num_emb(
                    self,
                    init_model_path,
                    latent_dim=cfg.latent_dim,
                    net_both_hands=False,
                    verbose=cfg.local_rank == 0,
                )
                if cfg.is_interhand and cfg.train_dataset.hand_type == "both":
                    assert args.get("hand_type") in ("left", "right"), f'Invalid hand_type: {args.get("hand_type")}'
                    if args["hand_type"] == "right":
                        prefix = "Right."
                    elif args["hand_type"] == "left":
                        prefix = "Left."
                else:
                    prefix = ""
                remove_hand_prefix = lambda model: remove_net_prefix(model, prefix) if prefix else model
                load_network(
                    self, init_model_path, strict=False, preprocess=remove_hand_prefix, verbose=cfg.local_rank == 0
                )
                if cfg.use_poses:
                    only = []
                    if cfg.poses_layer_policy in ("freeze", "finetune"):
                        only.append("poses_layer")
                    if cfg.learn_bw and cfg.bw_policy in ("freeze", "finetune"):
                        only.append("bw")
                    if only:
                        if cfg.local_rank == 0:
                            print(f"initialize params for {only} of novel_pose_deform")
                        load_network(
                            self.novel_pose_deform,
                            init_model_path,
                            only=only,
                            preprocess=lambda model: remove_net_prefix(remove_hand_prefix(model), "deform."),
                            verbose=cfg.local_rank == 0,
                        )

        if cfg.use_poses and not cfg.learn_bw and not cfg.learn_displacement:
            raise ValueError("poses can only be used when learn_bw or learn_displacement=True")

        if cfg.use_both_renderer and not cfg.use_neural_renderer:
            raise ValueError("use_neural_renderer must be True when use_both_renderer=True")

        if cfg.shading_mode == "ao" and cfg.use_neural_renderer:
            raise ValueError("shading_mode=ao is not supported with neural renderer")

    @staticmethod
    @torch.no_grad()
    def get_idx_near_surface(
        pts: torch.Tensor, bw_grid: torch.Tensor, bounds: torch.Tensor, norm_threshold: float = cfg.norm_th
    ) -> torch.Tensor:
        """Get the indices of points near enough to the SMPL surface.

        Args:
            pts: [n_batch, n_point, 3]
            bw_grid: [n_batch, depth, height, width, n_joint + 1]
            bounds: [n_batch, 2, 3]
        Returns:
            idx: [n_batch, n_point]
        """
        init_bw = pts_sample_blend_weights(pts, bw_grid, bounds)
        norm = init_bw[:, -1]
        idx = norm < norm_threshold
        # ensure not empty
        if not torch.any(idx):
            idx[torch.arange(norm.shape[0]), norm.argmin(dim=1)] = True
        return idx

    @staticmethod
    @torch.no_grad()
    def get_idx_inside_bounds(pts: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: [n_batch, n_point, 3]
            bounds: [n_batch, 2, 3]
        Returns:
            idx: [n_batch, n_point]
        """
        inside = (pts > bounds[:, :1]) & (pts < bounds[:, 1:])
        inside = torch.sum(inside, dim=2) == 3
        # ensure not empty
        if not torch.any(inside):
            inside[torch.arange(pts.shape[0]), pts.argmin(dim=1)] = True
        # outside = ~inside
        return inside

    @staticmethod
    @torch.no_grad()
    def get_idx_large_density(density: torch.Tensor, density_threshold: float = cfg.train_th) -> torch.Tensor:
        """
        Args:
            density: [n_batch, n_point]
        Returns:
            idx: [n_batch, n_point]
        """
        density = density.detach()
        density_idx = density > density_threshold
        # ensure not empty
        if not torch.any(density_idx):
            density_idx[torch.arange(density.shape[0]), density.argmax(dim=1)] = True
        return density_idx

    @staticmethod
    def get_full_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D]
            mask: [...] (mask.sum() == N)
        Returns:
            [..., D]
        """
        x_full = torch.zeros([*mask.shape, x.shape[-1]], dtype=x.dtype, device=x.device)
        x_full[mask] = x
        return x_full

    def forward(
        self,
        wpts: torch.Tensor | tuple[torch.Tensor],
        viewdir: torch.Tensor,
        dists: torch.Tensor,
        batch: dict,
        is_canonical=False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            wpts: [n_point, 3] or tuple(wpts, covs: [n_point, 3])
            viewdir: [n_point, 3]
            dists: [n_point]
        Returns:
            dict{"raw": [1, n_point, rgb_dim + 1]}
        """
        if cfg.render_type == "canonical":
            is_canonical = True

        if cfg.encoding == "mip":
            wpts, covs = wpts
        # transform points from the world space to the pose space
        wpts = wpts.unsqueeze(0)
        pose_pts = world_points_to_pose_points(wpts, batch["R"], batch["Th"])

        # filter out points that are too far away from SMPL surface
        ppts_idx = self.get_idx_near_surface(pose_pts, batch["pbw"], batch["pbounds"])
        pose_pts = pose_pts[ppts_idx].unsqueeze(0)
        viewdir = viewdir[ppts_idx.squeeze(0)].unsqueeze(0)
        dists = dists[ppts_idx.squeeze(0)]
        if cfg.encoding == "mip":
            covs = covs[ppts_idx.squeeze(0)].unsqueeze(0)

        # transform points from the pose space to the tpose space
        if (not self.training and is_canonical) or cfg.encoding == "uvd":
            tpose_pts = pose_pts
            displ = None
        elif cfg.aninerf_animation:
            tpose_pts, pbw, displ = self.novel_pose_deform(pose_pts, batch)
        else:
            tpose_pts, pbw, displ = self.deform(pose_pts, batch)

        tbounds = batch["tbounds"]
        # filter out points outside tbounds
        # tpose_pts = torch.clip(tpose_pts, tbounds[:, 0:1], tbounds[:, 1:2])
        inside = self.get_idx_inside_bounds(tpose_pts, tbounds)
        inside_coord = torch.argwhere(ppts_idx)[inside.squeeze(0)]
        inside_full = torch.full_like(ppts_idx, False)
        inside_full[inside_coord[:, 0], inside_coord[:, 1]] = True
        ppts_idx = inside_full
        tpose_pts = tpose_pts[inside].unsqueeze(0)
        viewdir = viewdir[inside].unsqueeze(0)
        dists = dists[inside.squeeze(0)]
        if displ is not None:
            displ = displ[inside].unsqueeze(0)
        if cfg.encoding == "mip":
            covs = covs[inside].unsqueeze(0)

        if cfg.is_interhand and not batch["is_rhand"] and cfg.encoding != "uvd":
            tpose_pts_orig = tpose_pts.clone()
            tpose_pts[:, :, 0] = -tpose_pts[:, :, 0]
            # tbounds_orig = tbounds.clone()
            tbounds[:, :, 0] = -tbounds[:, :, 0]
            tbounds[:, 0, 0], tbounds[:, 1, 0] = tbounds[:, 1, 0].clone(), tbounds[:, 0, 0].clone()
        else:
            tpose_pts_orig = tpose_pts
            # tbounds_orig = tbounds

        if cfg.use_pose_viewdir or cfg.use_tpose_viewdir:
            viewdir = world_dirs_to_pose_dirs(viewdir, batch["R"])
        if cfg.use_tpose_viewdir and not is_canonical:
            viewdir = pose_dirs_to_tpose_dirs(viewdir, pbw, batch["A"])
            if cfg.use_amp and self.training:
                viewdir = viewdir.half()

        density, rgb, rgb_feat = self.tpose_human.calculate_density_rgb(
            (tpose_pts, covs) if cfg.encoding == "mip" else tpose_pts,
            viewdir,
            batch["latent_index"],
            bounds=tbounds,
            extra=batch if cfg.encoding == "uvd" else None,
        )

        # set density (unactivated) to 0 for points outside tbounds
        # inside = self.get_idx_inside_bounds(tpose_pts_orig, tbounds_orig)
        density = density.squeeze(1)
        # density[~inside] = 0

        density2alpha = lambda density, dists, act_fn=F.relu, bias=0.0: 1.0 - torch.exp(-act_fn(density + bias) * dists)
        alpha = density2alpha(density.squeeze(0), dists, act_fn=self.density_actvn, bias=self.density_bias)
        rgb = rgb_feat.squeeze(0) if rgb is None else self.rgb_actvn(rgb.squeeze(0))
        # rgb[rgb.isnan()] = 0.0
        # rgb[rgb.isinf()] = 1.0
        if cfg.use_both_renderer or (self.training and cfg.use_distill and not cfg.use_neural_renderer):
            rgb = torch.cat((rgb, rgb_feat.squeeze(0)), dim=0)
        raw = torch.cat((rgb, alpha.unsqueeze(0)), dim=0)
        raw = raw.transpose(0, 1)
        raw_full = self.get_full_tensor(raw, ppts_idx)
        ret = {"raw": raw_full}

        if self.training and not is_canonical:
            # the deformation field is only optimized for points with larger density
            density_idx = self.get_idx_large_density(density)
            if cfg.learn_bw and not cfg.aninerf_animation:
                pbw = (pbw.transpose(1, 2))[inside].unsqueeze(0).transpose(1, 2)
                # calculate neural blend weights of points in the tpose space
                tbw = self.deform.get_bw(tpose_pts, batch, is_tpose=True, init_pts=tpose_pts_orig)
                pbw = pbw.transpose(1, 2)[density_idx].unsqueeze(0)
                tbw = tbw.transpose(1, 2)[density_idx].unsqueeze(0)
                ret.update({"pbw": pbw, "tbw": tbw})
            if cfg.learn_displacement:
                # tdispl = self.deform.get_displacement(tpose_pts, batch, is_tpose=True)
                displ = displ[density_idx].unsqueeze(0)
                ret["displ"] = displ
            if cfg.use_viewdir and cfg.use_color_range_loss:
                # color of the same point should not change too much due to viewing directions
                viewdir_dummy = torch.randn(viewdir.shape, dtype=viewdir.dtype, device=viewdir.device)
                viewdir_dummy = viewdir_dummy / torch.norm(viewdir_dummy, dim=-1, keepdim=True)
                _, rgb_dummy, rgb_dummy_feat = self.tpose_human.calculate_density_rgb(
                    (tpose_pts, covs) if cfg.encoding == "mip" else tpose_pts,
                    viewdir_dummy,
                    batch["latent_index"],
                    bounds=tbounds,
                )
                if rgb_dummy is None:
                    rgb_dummy = rgb_dummy_feat.squeeze(0).transpose(0, 1)
                else:
                    rgb_dummy = self.rgb_actvn(rgb_dummy.squeeze(0)).transpose(0, 1)
                rgb_dummy_full = self.get_full_tensor(rgb_dummy, ppts_idx)
                ret["rgb_dummy"] = rgb_dummy_full
            if cfg.train_dataset.hand_type == "both" and cfg.use_hands_align_loss:
                ret["tsamples_xyz"] = self.get_full_tensor(tpose_pts.detach(), ppts_idx)
                ret["tsamples_viewdir"] = self.get_full_tensor(viewdir.detach(), ppts_idx)
                ret["tsamples_alpha"] = self.get_full_tensor(alpha.detach().unsqueeze(0).unsqueeze(-1), ppts_idx)
                if cfg.encoding == "mip":
                    ret["tsamples_covs"] = self.get_full_tensor(covs.detach(), ppts_idx)

        return ret

    def calculate_density(self, wpts: torch.Tensor | tuple[torch.Tensor], batch: dict):
        """Outdated"""
        if cfg.encoding == "mip":
            wpts, covs = wpts
        # transform points from the world space to the pose space
        wpts = wpts.unsqueeze(0)
        pose_pts = world_points_to_pose_points(wpts, batch["R"], batch["Th"])

        # filter out points that are too far away from SMPL surface
        ppts_idx = self.get_idx_near_surface(pose_pts, batch["pbw"], batch["pbounds"])
        pose_pts = pose_pts[ppts_idx].unsqueeze(0)
        if cfg.encoding == "mip":
            covs = covs[ppts_idx.squeeze(0)].unsqueeze(0)

        tpose_pts, _, _ = self.deform(pose_pts, batch)

        tbounds = batch["tbounds"]
        if cfg.is_interhand and not batch["is_rhand"]:
            tpose_pts_orig = tpose_pts.clone()
            tpose_pts[:, :, 0] = -tpose_pts[:, :, 0]
            tbounds_orig = tbounds.clone()
            tbounds[:, :, 0] = -tbounds[:, :, 0]
            tbounds[:, 0, 0], tbounds[:, 1, 0] = tbounds[:, 1, 0].clone(), tbounds[:, 0, 0].clone()
        else:
            tpose_pts_orig = tpose_pts
            tbounds_orig = tbounds

        density = self.tpose_human.calculate_density(
            (tpose_pts, covs) if cfg.encoding == "mip" else tpose_pts, bounds=tbounds
        )

        # set density (unactivated) to 0 for points outside tbounds
        inside = self.get_idx_inside_bounds(tpose_pts_orig, tbounds_orig)
        density = density.squeeze(1)
        density[~inside] = 0

        density_full = torch.zeros([wpts.shape[1]]).to(wpts)
        density_full[ppts_idx[0]] = density[0]

        return density_full


class TPoseHuman(nn.Module):
    def __init__(self):
        super().__init__()

        if cfg.shading_mode in ("latent", "ao"):
            if "both" in (cfg.train_dataset.hand_type, cfg.test_dataset.hand_type) and cfg.hands_share_params:
                pose_num = 2 * cfg.num_train_frame
            else:
                pose_num = cfg.num_train_frame
            self.rgb_latent = nn.Embedding(pose_num, cfg.latent_dim)

        if cfg.encoding == "nerf":
            in_dim = embedder.xyz_dim
        elif cfg.encoding == "mip":
            from lib.networks.mipnerf import IntegratedPositionalEncoder  # fmt: skip
            self.posi_enc = IntegratedPositionalEncoder(max_deg=cfg.xyz_res)
            in_dim = embedder.xyz_dim - 3
        elif cfg.encoding == "triplane":
            from lib.networks.encodings import TriplaneEncoding  # fmt: skip
            self.xyz_encoding = TriplaneEncoding(
                resolution=cfg.triplane_res, num_components=cfg.triplane_dim, reduce=cfg.triplane_reduce
            )
            in_dim = self.xyz_encoding.get_out_dim()
        elif cfg.encoding == "hash":
            from lib.networks.encodings import HashEncoding  # fmt: skip
            self.xyz_encoding = HashEncoding()
            in_dim = self.xyz_encoding.get_out_dim()
        elif cfg.encoding == "uvd":
            in_dim = embedder.xyz_dim
        else:
            raise ValueError(f"Invalid encoding method: {cfg.encoding}")

        W = cfg.xyz_hidden_dim
        self.xyz_linears = ResMLP(in_dim, W, W, cfg.xyz_layer_num, cfg.xyz_skips, cfg.xyz_activation)
        self.density_fc = nn.Conv1d(W, 1, 1)

        self.feature_fc = nn.Conv1d(W, W, 1)
        if cfg.shading_mode == "latent":
            self.latent_fc = nn.Conv1d(W + cfg.latent_dim, W, 1)
        elif cfg.shading_mode == "ao":
            self.ao_layer = ResMLP(W + cfg.latent_dim, 1, [256, 128], 3, [], ["relu", "sigmoid"])

        if cfg.encoding == "hash":
            from lib.networks.encodings import SphericalHarmonicsEncoding  # fmt: skip
            self.view_encoding = SphericalHarmonicsEncoding()
            view_dim = self.view_encoding.get_out_dim()
        else:
            self.view_encoding = embedder.view_embedder
            view_dim = embedder.view_dim

        self.view_fc = nn.Sequential(
            nn.Conv1d(W + view_dim if cfg.use_viewdir else W, W // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(W // 2, cfg.rgb_dim, 1),
            nn.Sigmoid() if cfg.view_activation == "sigmoid" else nn.ReLU(inplace=True),
        )
        if not cfg.use_neural_renderer or cfg.use_both_renderer:
            self.rgb_fc = nn.Conv1d(cfg.rgb_dim, 3, 1)

    def calculate_density(
        self,
        nf_pts: torch.Tensor | tuple[torch.Tensor],
        bounds: torch.Tensor = None,
        return_feat=False,
        extra=None,
    ):
        """
        Args:
            nf_pts: [n_batch, n_point, 3] or tuple(nf_pts, covs: [n_batch, n_point, 3])
            bounds: [n_batch, 2, 3], for rescaling, bounds[:, 0] is the min and bounds[:, 1] is the max
        Returns:
            density (unactivated): [n_batch, 1, n_point]
        """
        if cfg.encoding == "mip":
            nf_pts, covs = nf_pts
        if cfg.pe_scaling or cfg.encoding in ("triplane", "hash"):
            assert bounds is not None, "bounds of coordinates must be provided"
            if cfg.encoding == "triplane":
                target_min, target_max = -1, 1
            elif cfg.encoding == "hash":
                target_min, target_max = 0, 1
            else:
                target_min, target_max = -torch.pi, torch.pi
            # nf_pts = nf_pts * (torch.pi * 4.0)
            nf_pts, norm_ratio = normalize_batch(
                nf_pts,
                target_min,
                target_max,
                x_min=bounds[:, 0:1],
                x_max=bounds[:, 1:2],
                # x_min=torch.Tensor((-0.25, -0.25, -0.25)).to(nf_pts.device).view(1, 1, 3),
                # x_max=torch.Tensor((0.25, 0.25, 0.25)).to(nf_pts.device).view(1, 1, 3),
                dim=-2,
                return_ratio=True,
            )
            if cfg.encoding == "mip":
                # covs = covs * ((torch.pi * 4.0) ** 2)
                covs = covs * (norm_ratio**2)
        if cfg.encoding == "nerf":
            nf_pts = embedder.xyz_embedder(nf_pts)
        elif cfg.encoding == "mip":
            nf_pts = self.posi_enc((nf_pts, covs))
        elif cfg.encoding in ("triplane", "hash"):
            nf_pts = self.xyz_encoding(nf_pts)
        elif cfg.encoding == "uvd":
            from lib.utils.data_utils import get_uvd

            mesh_vertices = extra["mano_vertices"].squeeze(0)
            mesh_faces = extra["mano_faces"].squeeze(0)
            mesh_face_uv = extra["mano_face_uv"].squeeze(0)
            pts_uv, pts_d, _ = get_uvd(nf_pts.squeeze(0).float(), mesh_vertices, mesh_faces, mesh_face_uv)
            nf_pts = torch.cat([pts_uv, pts_d], -1).unsqueeze(0).to(nf_pts)
            nf_pts = embedder.xyz_embedder(nf_pts)
        net: torch.Tensor = self.xyz_linears(nf_pts.transpose(1, 2))
        density: torch.Tensor = self.density_fc(net)
        return (density, net) if return_feat else density

    def calculate_density_rgb(
        self,
        nf_pts: torch.Tensor | tuple[torch.Tensor],
        viewdir: torch.Tensor,
        latent_index: torch.Tensor,
        bounds: torch.Tensor = None,
        extra=None,
    ):
        """
        Args:
            nf_pts: [n_batch, n_point, 3] or tuple(nf_pts, covs: [n_batch, n_point, 3])
            viewdir: [n_batch, n_point, 3]
            latent_index: int
        Returns:
            density (unactivated): [n_batch, 1, n_point]
            rgb (unactivated): [n_batch, 3, n_point]
            rgb_feat: [n_batch, rgb_dim, n_point]
        """
        density, net = self.calculate_density(nf_pts, bounds, return_feat=True, extra=extra)
        features = self.feature_fc(net)

        if cfg.shading_mode in ("latent", "ao"):
            assert (
                int(latent_index) < self.rgb_latent.num_embeddings
            ), f"expect latent_index < {self.rgb_latent.num_embeddings} but got {int(latent_index)}"
            latent = self.rgb_latent(latent_index)
            latent = latent.unsqueeze(-1).expand(*latent.shape, net.shape[2])
            features_latent = torch.cat((features, latent), dim=1)
            if cfg.shading_mode == "latent":
                features = self.latent_fc(features_latent)
            elif cfg.shading_mode == "ao":
                ao = self.ao_layer(features_latent)

        if cfg.use_viewdir:
            if cfg.pe_scaling:
                viewdir = viewdir * torch.pi  # viewdir is normalized to a unit vector, so we scale it to [-pi, pi]
            viewdir = self.view_encoding(viewdir)
            viewdir = viewdir.transpose(1, 2)
            features = torch.cat((features, viewdir), dim=1)
        net: torch.Tensor = self.view_fc(features)
        if cfg.use_neural_renderer and not cfg.use_both_renderer:
            rgb = None
        else:
            rgb: torch.Tensor = self.rgb_fc(net)

        if cfg.shading_mode == "ao":
            rgb = rgb * ao

        return density, rgb, net


class NeuralBlendWeight(nn.Module):
    def __init__(self):
        super().__init__()

        assert cfg.learn_bw

        in_dim = embedder.xyz_dim + cfg.latent_dim
        W = cfg.bw_hidden_dim
        self.bw_linears = ResMLP(in_dim, W, W, cfg.bw_layer_num, cfg.bw_skips, cfg.bw_activation)
        self.bw_fc = nn.Conv1d(W, cfg.joints_num, 1)

    def forward(self, pts_pose_feature: torch.Tensor, smpl_bw: torch.Tensor):
        """
        Args:
            pts_pose_feature: [n_batch, in_dim, n_point]
            smpl_bw: [n_batch, n_joint, n_point]
        Returns:
            bw: [n_batch, n_joint, n_point]
        """
        net = self.bw_linears(pts_pose_feature)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw


class DeformationField(nn.Module):
    def __init__(self, pose_num: int = cfg.num_train_frame):
        super().__init__()

        if cfg.learn_bw:
            self.bw = NeuralBlendWeight()

        if cfg.use_poses:
            poses_dim = 3 if cfg.poses_format == "axis_angle" else 4
            self.poses_layer = ResMLP(
                (cfg.joints_num - 1) * poses_dim,
                cfg.latent_dim,
                cfg.poses_hidden_dim,
                cfg.poses_layer_num,
                cfg.poses_skips,
                cfg.poses_activation,
            )
        elif cfg.learn_bw or cfg.learn_displacement:
            self.bw_latent = nn.Embedding(pose_num, cfg.latent_dim)

        if cfg.learn_displacement:
            self.displacement_layer = ResMLP(
                embedder.xyz_dim + cfg.latent_dim,
                3,
                cfg.displ_hidden_dim,
                cfg.displ_layer_num,
                cfg.displ_skips,
                cfg.displ_activation,
            )
            self.displacement_layer.linears[-1].bias.data.fill_(0)

    def get_point_feature(self, pts: torch.Tensor, pose_repr: torch.Tensor, reflect=False):
        """
        Args:
            pts: [n_batch, n_point, 3]
            pose_repr: poses_vector or latent_index
        Returns:
            features: [n_batch, embedder.xyz_dim + cfg.latent_dim, n_point]
        """
        if reflect:
            pts = pts.clone()
            pts[:, :, 0] = -pts[:, :, 0]
        if cfg.pe_scaling:
            pts = pts * 4.0 * torch.pi  # pts are basically in [-0.25, 0.25], so we scale them to [-pi, pi]
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        if cfg.use_poses:
            pose_repr = self.poses_layer(pose_repr.unsqueeze(-1)).squeeze(-1)
        else:
            assert (
                int(pose_repr) < self.bw_latent.num_embeddings
            ), f"expect latent_index < {self.bw_latent.num_embeddings} but got {int(pose_repr)}"
            pose_repr = self.bw_latent(pose_repr)
        pose_repr = pose_repr.unsqueeze(-1).expand(*pose_repr.shape, pts.shape[2])
        features = torch.cat((pts, pose_repr), dim=1)
        return features

    def get_bw_(
        self,
        pts: torch.Tensor,
        bw_grid: torch.Tensor,
        bounds: torch.Tensor,
        pose_repr: torch.Tensor = None,
        init_pts: torch.Tensor = None,
        reflect=False,
    ) -> torch.Tensor:
        """
        Args:
            pts: [n_batch, n_point, 3]
            bw_grid: [n_batch, depth, height, width, n_joint + 1]
            bounds: [n_batch, 2, 3]
            pose_repr: poses_vector or latent_index
        Returns:
            bw: [n_batch, n_joint, n_point]
        """
        init_bw = pts_sample_blend_weights(pts if init_pts is None else init_pts, bw_grid, bounds)
        init_bw = init_bw[:, : cfg.joints_num]
        if cfg.learn_bw:
            feature = self.get_point_feature(pts, pose_repr, reflect)
            bw = self.bw(feature, init_bw)
        else:
            bw = init_bw
        return bw

    def get_bw(self, pts: torch.Tensor, batch: dict, is_tpose=False, init_pts: torch.Tensor = None):
        """Wrapper of get_bw_()

        Args:
            pts: [n_batch, n_point, 3]
        Returns:
            bw: [n_batch, n_joint, n_point]
        """
        if cfg.use_poses:
            pose_repr = batch["poses"][:, 1:, :].reshape(batch["poses"].shape[0], -1)
            tpose_repr = batch["tposes"][:, 1:, :].reshape(batch["tposes"].shape[0], -1)
        else:
            pose_repr = batch["bw_latent_index"] if cfg.aninerf_animation else batch["latent_index"] + 1
            tpose_repr = (
                torch.zeros_like(batch["bw_latent_index"])
                if cfg.aninerf_animation
                else torch.zeros_like(batch["latent_index"])
            )
        bw = self.get_bw_(
            pts,
            batch["tbw"] if is_tpose else batch["pbw"],
            batch["tbounds"] if is_tpose else batch["pbounds"],
            tpose_repr if is_tpose else pose_repr,
            init_pts,
            reflect=cfg.is_interhand and not batch["is_rhand"],
        )
        return bw

    def get_displacement_(self, pts: torch.Tensor, pose_repr: torch.Tensor, reflect=False) -> torch.Tensor:
        """
        Args:
            pts: [n_batch, n_point, 3]
            pose_repr: poses_vector or latent_index
        Returns:
            displacement: [n_batch, n_point, 3]
        """
        assert cfg.learn_displacement
        feature = self.get_point_feature(pts, pose_repr, reflect)
        displacement = self.displacement_layer(feature)
        displacement = displacement.permute(0, 2, 1).contiguous()
        displacement *= 0.05
        if reflect:
            displacement[:, :, 0] = -displacement[:, :, 0]
        return displacement

    def get_displacement(self, pts: torch.Tensor, batch: dict, is_tpose=False):
        """Wrapper of get_displacement_()

        Args:
            pts: [n_batch, n_point, 3]
        Returns:
            displacement: [n_batch, n_point, 3]
        """
        if cfg.use_poses:
            pose_repr = batch["poses"][:, 1:, :].reshape(batch["poses"].shape[0], -1)
            # tpose_repr = batch["tposes"][:, 1:, :].reshape(batch["tposes"].shape[0], -1)
        else:
            pose_repr = batch["bw_latent_index"] if cfg.aninerf_animation else batch["latent_index"] + 1
            # tpose_repr = (
            #     torch.zeros_like(batch["bw_latent_index"])
            #     if cfg.aninerf_animation
            #     else torch.zeros_like(batch["latent_index"])
            # )
        # displacement = self.get_displacement_(pts, tpose_repr if is_tpose else pose_repr)
        displacement = self.get_displacement_(pts, pose_repr, reflect=cfg.is_interhand and not batch["is_rhand"])
        return displacement

    def forward(self, pose_pts: torch.Tensor, batch: dict):
        """
        pose_pts: [n_batch, n_point, 3]
        """
        pbw = self.get_bw(pose_pts, batch)
        tpose_pts = pose_points_to_tpose_points(pose_pts, pbw, batch["A"])
        if cfg.use_amp and self.training:
            tpose_pts = tpose_pts.half()

        if cfg.use_bs:
            bs = pts_sample_blend_weights(tpose_pts, batch["bs"], batch["tbounds"])
            tpose_pts = tpose_pts - bs.permute(0, 2, 1)

        if cfg.learn_displacement and (not cfg.test_novel_pose or cfg.aninerf_animation):
            displacement = self.get_displacement(pose_pts, batch)
            tpose_pts = tpose_pts + displacement
        else:
            displacement = None

        return tpose_pts, pbw, displacement
