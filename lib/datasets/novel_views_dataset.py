import os
import pickle
from math import ceil

import cv2
import numpy as np
import torch.utils.data as data

from lib.config import cfg
from lib.utils.blend_utils import get_bs_from_shape_and_pose, get_bw_from_vertices, get_vertices_from_pose
from lib.utils.if_nerf.if_nerf_data_utils import get_bounds, get_near_far, get_radii, get_rays, get_rigid_transformation


def sample_ray(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return ray_o, ray_d, near, far, coord, mask_at_box


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def create_spiral_poses(radii, focus_depth=3.5, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spheric_poses(radius, elevation=-np.pi / 5, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, -0.9 * t], [0, 0, 1, t], [0, 0, 0, 1]])

        rot_phi = lambda phi: np.array(
            [[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]]
        )

        rot_theta = lambda th: np.array(
            [[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]]
        )

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, elevation, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class Dataset(data.Dataset):
    def __init__(self, data_root: str, ratio: float = cfg.ratio, hand_type: str = ""):
        super().__init__()

        self.H = int(cfg.H * ratio)
        self.W = int(cfg.W * ratio)
        self.hand_type = hand_type
        self.online_pose = cfg.get("test_pose") is not None
        if self.online_pose:
            import smplx  # fmt: skip
            mano_path = f"tools/InterHand2.6M/MANO_{self.hand_type.upper()}.pkl"
            with open(mano_path, "rb") as f:
                self.mano_data = pickle.load(f, encoding="latin1")
            self.mano_model: smplx.MANO = smplx.create(mano_path, use_pca=False, is_rhand=self.hand_type == "right")
            self.pose_mean = self.mano_model.pose_mean.unsqueeze(0).detach().numpy()

        with open(os.path.join(data_root, "frames.txt"), "r") as f:
            frame_list = f.readlines()
        frame_list = list(map(lambda x: int(x.strip()), frame_list))
        if cfg.render_type in ("static", "canonical"):
            if cfg.render_frame != -1:
                render_frame = cfg.render_frame
                self.latent_index = frame_list.index(render_frame)
            else:
                render_frame = frame_list[0]
                self.latent_index = 0
            if self.latent_index >= cfg.num_train_frame:
                raise ValueError(
                    f"frame {render_frame} is not trained: expect frame number < {cfg.num_train_frame} but got {self.latent_index}"
                )
            self.frame_index = render_frame
        elif cfg.render_type == "animation":
            self.frame_index = frame_list[: cfg.num_train_frame] + frame_list[cfg.num_train_frame - 2 :: -1]
            render_frame = self.frame_index[0]
            self.latent_index = list(range(cfg.num_train_frame)) + list(range(cfg.num_train_frame)[-2::-1])
        else:
            raise ValueError(f"Invalid render_type: {cfg.render_type}")

        self.data_root = data_root
        self.lbs_root = os.path.join(self.data_root, "lbs", self.hand_type)
        self.joints = np.load(os.path.join(self.lbs_root, "joints.npy")).astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, "parents.npy"))
        self.faces = np.load(os.path.join(self.lbs_root, "faces.npy")).astype(np.float32)

        annots = np.load(os.path.join(data_root, "annots.npy"), allow_pickle=True).item()
        self.K = np.array(next(iter(annots["cams"]["K"].values())))
        self.K[:2] = self.K[:2] * ratio
        self.num_cams = len(cfg.training_view)

        tpose = np.load(os.path.join(self.lbs_root, "tvertices.npy")).astype(np.float32)
        self.tbounds = get_bounds(tpose)
        self.tbw = np.load(os.path.join(self.lbs_root, "tbw.npy")).astype(np.float32)
        tparams = np.load(
            os.path.join(self.data_root, cfg.params, self.hand_type, "tpose.npy"), allow_pickle=True
        ).item()
        self.tposes = tparams["poses"].astype(np.float32).reshape(-1, 3)
        if cfg.is_interhand and self.hand_type == "left":
            self.tposes[:, 1] = -self.tposes[:, 1]
            self.tposes[:, 2] = -self.tposes[:, 2]

        wpts = self.prepare_input(render_frame)[0]
        self.wpts_mean = np.mean(wpts * 1000, axis=0)

        if cfg.render_view in ("surround", "forward", "fixed"):
            poses = create_spheric_poses(1000, -np.pi / 2, n_poses=cfg.render_cams)  # c2w, mm
        elif cfg.render_view == "spiral":
            poses = create_spiral_poses(np.array([50, 100, 30]), 1000, n_poses=cfg.render_cams)
        else:
            raise ValueError(f"Invalid render_view: {cfg.render_view}")
        poses = np.concatenate([poses[:, :, 0:1], poses[:, :, 2:3], poses[:, :, 1:2], poses[:, :, 3:4]], -1)
        poses = np.concatenate([poses[:, 0:1, :], poses[:, 2:3, :], poses[:, 1:2, :]], 1)
        if cfg.render_view == "fixed":
            poses = poses[poses.shape[0] // 2, :, :][None]
        if cfg.render_type == "animation":
            if cfg.render_view == "forward":
                assert 0 <= cfg.render_forward_range[0] < cfg.render_forward_range[1] <= 1
                view_range = list(map(lambda x: int(cfg.render_cams * x), cfg.render_forward_range))
                poses = np.concatenate(
                    [poses[view_range[0] : view_range[1], :, :], poses[view_range[1] : view_range[0] : -1, :, :]],
                    axis=0,
                )
            if poses.shape[0] < len(frame_list):
                n = ceil(len(frame_list) / poses.shape[0])
                if cfg.local_rank == 0:
                    print(
                        f"Warning: extend the number of views from {poses.shape[0]} to {n * poses.shape[0]}, to fulfil {len(frame_list)} frames."
                    )
                poses = np.tile(poses, (n, 1, 1))

        # np.save("./tools/InterHand2.6M/novel_cams.npy", poses)
        self.poses = poses
        self.R_c2w = poses[:, :, 0:3]
        self.set_T_c2w()

    def set_T_c2w(self, wpts_mean=None):
        if wpts_mean is not None:
            self.wpts_mean = wpts_mean
        if cfg.render_view in ("surround", "forward", "fixed"):
            self.poses[:, 1, -1] = 0
            self.T_c2w = self.poses[:, :, -1] + self.wpts_mean
        elif cfg.render_view == "spiral":
            self.T_c2w = self.poses[:, :, -1] + np.array([0, self.wpts_mean[1], 0])

    def prepare_input(self, i):
        # transform smpl from the world coordinate to the smpl coordinate
        if cfg.render_type == "canonical":
            tparams_path = os.path.join(self.data_root, "params", self.hand_type, "tpose.npy")
            tparams = np.load(tparams_path, allow_pickle=True).item()
        if cfg.render_type == "canonical" and cfg.render_frame == -1:
            # follow the world position of t-pose
            params = tparams
        else:
            if self.online_pose:
                params = cfg.test_pose[0]
                # params["poses"] += self.pose_mean
            else:
                params_path = os.path.join(self.data_root, "params", self.hand_type, f"{i}.npy")
                params = np.load(params_path, allow_pickle=True).item()
            if cfg.render_type == "canonical":
                # follow the world position of frame i
                params["poses"] = tparams["poses"]

        Rh = params["Rh"].astype(np.float32)
        Th = params["Th"].astype(np.float32)

        # prepare sp input of param pose
        Rh = cv2.Rodrigues(Rh)[0].astype(np.float32)

        if cfg.render_type == "canonical":
            vertices_path = os.path.join(self.lbs_root, "tvertices.npy")
            pxyz = np.load(vertices_path).astype(np.float32)
            wxyz = (np.dot(pxyz, np.linalg.inv(Rh)) + Th).astype(np.float32)
        else:
            # read xyz in the world coordinate system
            if self.online_pose:
                wxyz = get_vertices_from_pose(
                    self.mano_model,
                    params["poses"] - self.pose_mean,
                    # params["Rh"],
                    # params["shapes"],
                    # transl=params.get("transl"),
                    # params["mano_param"]["pose"],
                    params["mano_param"]["pose"][:3],
                    params["mano_param"]["shape"],
                    transl=params["mano_param"]["trans"],
                )
            else:
                vertices_path = os.path.join(self.data_root, "vertices", self.hand_type, f"{i}.npy")
                wxyz = np.load(vertices_path).astype(np.float32)
            pxyz = np.dot(wxyz - Th, Rh).astype(np.float32)

        # calculate the skeleton transformation
        poses = params["poses"].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = get_rigid_transformation(poses, joints, parents)

        if cfg.render_type == "canonical":
            pbw = np.load(os.path.join(self.lbs_root, "tbw.npy")).astype(np.float32)
        else:
            if self.online_pose:
                pbw = get_bw_from_vertices(self.mano_data, pxyz, is_lhand=self.hand_type == "left")
            else:
                pbw = np.load(os.path.join(self.lbs_root, f"bweights/{i}.npy")).astype(np.float32)

        if cfg.use_bs:
            if self.online_pose:
                bs = get_bs_from_shape_and_pose(self.mano_data, params, pxyz, is_lhand=self.hand_type == "left")
            else:
                bs = np.load(os.path.join(self.lbs_root, f"bs/{i}.npy")).astype(np.float32)
        else:
            bs = None

        return wxyz, pxyz, A, pbw, Rh, Th, poses, bs

    def __getitem__(self, index):
        if cfg.render_type in ("static", "canonical"):
            latent_index = self.latent_index
            frame_index = self.frame_index
        elif cfg.render_type == "animation":
            latent_index = self.latent_index[index % len(self.latent_index)]
            frame_index = self.frame_index[index % len(self.frame_index)]
        bw_latent_index = latent_index

        # c2w to w2c
        R = np.linalg.inv(self.R_c2w[index])
        T = -(np.dot(R, self.T_c2w[index].reshape(3, 1))) / 1000.0

        wpts, ppts, A, pbw, Rh, Th, poses, bs = self.prepare_input(frame_index)

        if cfg.is_interhand and self.hand_type == "left":
            poses[:, 1] = -poses[:, 1]
            poses[:, 2] = -poses[:, 2]

        pbounds = get_bounds(ppts)
        wbounds = get_bounds(wpts)
        ray_o, ray_d, near, far, coord, mask_at_box = sample_ray(self.H, self.W, self.K, R, T, wbounds)

        msk = np.ones((self.H, self.W)).astype(np.uint8)
        occupancy = msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            # "rgb": rgb,
            "occupancy": occupancy,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "mask_at_box": mask_at_box,
        }

        # blend weight
        meta = {
            "A": A,
            "pbw": pbw,
            "tbw": self.tbw,
            "pbounds": pbounds,
            "wbounds": wbounds,
            "tbounds": self.tbounds,
            "poses": poses,
            "tposes": self.tposes,
            "is_rhand": self.hand_type == "right",
        }
        ret.update(meta)

        # transformation
        meta = {"R": Rh, "Th": Th, "H": self.H, "W": self.W}
        ret.update(meta)

        meta = {"K": self.K, "R_w2c": R, "T_w2c": T}
        ret.update(meta)

        if (
            (cfg.train_dataset.hand_type == "both" or cfg.test_dataset.hand_type == "both")
            and self.hand_type == "left"
            and cfg.hands_share_params
        ):
            latent_index += cfg.num_train_frame
            bw_latent_index += cfg.num_train_frame
        if cfg.test_novel_pose:
            if "h36m" in self.data_root or cfg.is_interhand:
                latent_index = 0
            else:
                latent_index = cfg.num_train_frame - 1
        meta = {
            "latent_index": latent_index,
            "bw_latent_index": bw_latent_index,
            "frame_index": frame_index,
            "view_index": index,
        }
        ret.update(meta)

        if cfg.encoding == "mip":
            radii = get_radii(self.H, self.W, self.K, R, T)
            radii = radii[coord[:, 0], coord[:, 1]]
            ret["radii"] = radii

        if (cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr")) or cfg.use_sr != "none":
            H_sr, W_sr = int(self.H * cfg.sr_ratio), int(self.W * cfg.sr_ratio)
            ret.update({"msk_sr": np.ones((H_sr, W_sr)).astype(np.uint8), "H_sr": H_sr, "W_sr": W_sr})

        if cfg.use_alpha_sdf in ("always", "residual"):
            from lib.utils.sdf_utils import repair_mesh

            vertices, faces = repair_mesh(wpts, self.faces)
            ret.update({"vertices": vertices, "faces": faces})

        if cfg.use_bs:
            ret["bs"] = bs

        return ret

    def __len__(self):
        return self.R_c2w.shape[0]
