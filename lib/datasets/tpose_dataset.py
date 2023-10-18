import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch.utils.data as data

from lib.config import cfg
from lib.utils.base_utils import project
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, ratio: float = cfg.ratio, hand_type: str = ""):
        super().__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        self.ratio = ratio
        self.hand_type = hand_type

        if not os.path.exists(ann_file):
            full_path = os.path.join(data_root, ann_file)
            if os.path.exists(full_path):
                ann_file = full_path
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots["cams"]

        if len(cfg.test_view) == 0:
            if cfg.is_interhand:
                test_view = [id for id in list(self.cams["K"].keys()) if id not in cfg.training_view]
                if len(test_view) == 0:
                    test_view = [list(self.cams["K"].keys())[0]]
            else:
                test_view = [i for i in range(len(self.cams["K"])) if i not in cfg.training_view]
                if len(test_view) == 0:
                    test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == "train" else test_view
        self.views = view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose or cfg.aninerf_animation:
            if not cfg.is_interhand or cfg.test_rest_frames:
                i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
                ni = cfg.num_eval_frame

        if cfg.is_interhand:
            self.num_cams = len(view)
            self.ims = np.array(
                [np.array(ims_data["ims"]) for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]]
            ).ravel()
            self.ims = np.array(list(filter(lambda x: x.split("/")[0].split("cam")[1] in view, self.ims)))
            self.cam_inds = np.array(list(map(lambda x: x.split("/")[0].split("cam")[1], self.ims))).ravel()
        else:
            self.ims = np.array(
                [np.array(ims_data["ims"])[view] for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]]
            ).ravel()
            self.cam_inds = np.array(
                [np.arange(len(ims_data["ims"]))[view] for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]]
            ).ravel()
            self.num_cams = len(view)
        assert len(self.ims) > 0, "No image found with the given settings"

        self.lbs_root = os.path.join(self.data_root, "lbs", self.hand_type)
        self.joints = np.load(os.path.join(self.lbs_root, "joints.npy")).astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, "parents.npy"))
        self.faces = np.load(os.path.join(self.lbs_root, "faces.npy")).astype(np.float32)
        self.nrays = cfg.N_rand

        if cfg.encoding == "uvd":
            from lib.utils.data_utils import read_mano_uv_obj

            vt, ft, f = read_mano_uv_obj(f"tools/InterHand2.6M/MANO_UV_{hand_type}.obj")
            # vt: uv coordinates of the vertices of the MANO mesh                #(891, 2), range: [0, 1]
            # ft: MANO mesh face indices for vt                                  #(1538, 3), range: [0, 890]
            # f: MANO mesh face indices for the vertices of the MANO mesh        #(1538, 3), range: [0, 777]
            # mesh_faces = torch.tensor(f)                                    #NOTE: this is same as mano_layer[hand_type].faces
            self.mesh_face_uv = vt[ft]  # [1538, 3, 2]   #Neural Actor encoder.py line 1244

        if cfg.vis_tpose_mesh or cfg.vis_posed_mesh:
            self.mesh_samples = {}

    def get_mask_(self, img_name: str):
        msk_path = os.path.join(self.data_root, "mask_cihp", img_name)[:-4] + ".png"
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, "mask", img_name)[:-4] + ".png"
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, img_name.replace("images", "mask"))[:-4] + ".png"
        msk = imageio.imread(msk_path)
        if len(msk.shape) == 3:
            msk = msk[..., 0]
        return msk

    def get_mask(self, index: int):
        msk_cihp = self.get_mask_(self.ims[index])
        if self.hand_type == "left":
            msk_cihp[msk_cihp != cfg.left_mask_value] = 0
        elif self.hand_type == "right":
            msk_cihp[msk_cihp != cfg.right_mask_value] = 0
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if self.split == "train" and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 1

        return msk, orig_msk

    def prepare_input(self, i: int):
        # read xyz in the world coordinate system
        wxyz = np.load(os.path.join(self.data_root, cfg.vertices, self.hand_type, f"{i}.npy")).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params, self.hand_type, f"{i}.npy")
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params["Rh"].astype(np.float32)
        Th = params["Th"].astype(np.float32)

        # prepare sp input of param pose
        Rh = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, Rh).astype(np.float32)

        # calculate the skeleton transformation
        poses = params["poses"].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

        pbw = np.load(os.path.join(self.lbs_root, f"bweights/{i}.npy"))
        pbw = pbw.astype(np.float32)

        if cfg.use_bs:
            bs = np.load(os.path.join(self.lbs_root, f"bs/{i}.npy"))
            bs = bs.astype(np.float32)
        else:
            bs = None

        return wxyz, pxyz, A, pbw, Rh, Th, poses, bs

    def prepare_inside_pts(self, pts: np.ndarray, frame_index: int):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        # views = self.cams["K"].keys()
        views = self.views
        for view in views:
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([np.array(self.cams["R"][view]), np.array(self.cams["T"][view]) / 1000.0], axis=1)
            pts2d = project(pts3d_, np.array(self.cams["K"][view]), RT)

            msk = self.get_mask_(f"cam{view}/image{frame_index}.jpg")
            msk = (msk != 0).astype(np.uint8)
            msk = cv2.dilate(msk.copy(), np.ones((5, 5), np.uint8))
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        msk, orig_msk = self.get_mask(index)

        H, W = img.shape[:2]
        assert (H, W) == (cfg.H, cfg.W), f"{(H, W)} != {(cfg.H, cfg.W)}: {img_path}"
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        img_raw = img.copy()
        msk_raw = orig_msk.copy()

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams["K"][cam_ind])
        if not cfg.is_interhand:
            D = np.array(self.cams["D"][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams["R"][cam_ind])
        T = np.array(self.cams["T"][cam_ind]) / 1000.0

        # reduce the image resolution by ratio
        H, W = int(H * self.ratio), int(W * self.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2] * self.ratio

        if cfg.mask_bkgd:
            img[msk == 0] = 0

        if cfg.is_interhand:
            i = int(img_path.split("/")[-1][5:-4])
            frame_index = i
        elif self.human in ("CoreView_313", "CoreView_315"):
            i = int(os.path.basename(img_path).split("_")[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        tpose = np.load(os.path.join(self.lbs_root, "tvertices.npy")).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, "tbw.npy"))
        tbw = tbw.astype(np.float32)
        tparams = np.load(
            os.path.join(self.data_root, cfg.params, self.hand_type, "tpose.npy"), allow_pickle=True
        ).item()
        tposes = tparams["poses"].astype(np.float32).reshape(-1, 3)

        wpts, ppts, A, pbw, Rh, Th, poses, bs = self.prepare_input(i)

        if cfg.is_interhand and self.hand_type == "left":
            poses[:, 1] = -poses[:, 1]
            poses[:, 2] = -poses[:, 2]
            tposes[:, 1] = -tposes[:, 1]
            tposes[:, 2] = -tposes[:, 2]

        if cfg.poses_format == "axis_angle":
            pass
        elif cfg.poses_format == "axis_angle_norm":
            from lib.utils.blend_utils import axis_angle_to_norm  # fmt: skip
            poses = axis_angle_to_norm(poses)
            tposes = axis_angle_to_norm(tposes)
        elif cfg.poses_format == "quaternion":
            from lib.utils.blend_utils import axis_angle_to_quaternion  # fmt: skip
            poses = axis_angle_to_quaternion(poses)
            tposes = axis_angle_to_quaternion(tposes)
        else:
            raise ValueError(f"Unknown poses format: {cfg.poses_format}")

        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        if self.split == "train" and cfg.train_dataset.hand_type == "both":
            other_hand_mask = self.get_mask_(self.ims[index])
            if self.hand_type == "left":
                other_hand_mask[other_hand_mask != cfg.right_mask_value] = 0
            elif self.hand_type == "right":
                other_hand_mask[other_hand_mask != cfg.left_mask_value] = 0
            other_hand_mask = (other_hand_mask != 0).astype(np.uint8)
            other_hand_mask = cv2.resize(other_hand_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            other_hand_mask = None

        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split, forbidden_mask=other_hand_mask
        )

        # img_sample = np.ones((H, W, 3))
        # for uv in coord:
        #     img_sample[uv[0], uv[1]] = img[uv[0], uv[1]]
        # img_sample = (img_sample * 255.0).astype(np.uint8)

        if cfg.erode_edge:
            orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            "rgb": rgb,
            "msk": msk,
            "occupancy": occupancy,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "near": near,
            "far": far,
            "mask_at_box": mask_at_box,
            "coord": coord,
        }

        # blend weight
        meta = {
            "A": A,
            "pbw": pbw,
            "tbw": tbw,
            "pbounds": pbounds,
            "wbounds": wbounds,
            "tbounds": tbounds,
            "poses": poses,
            "tposes": tposes,
            "is_rhand": self.hand_type == "right",
        }
        ret.update(meta)

        # transformation
        meta = {"R": Rh, "Th": Th, "H": H, "W": W}
        ret.update(meta)

        meta = {"K": K, "R_w2c": R, "T_w2c": T}
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
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
            "cam_ind": int(cam_ind),
        }
        ret.update(meta)

        if cfg.use_depth or self.split != "train":
            depth_map_path = os.path.join(self.data_root, "depth", self.ims[index].replace(".jpg", ".npy"))
            depth_map = np.load(depth_map_path).squeeze()
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                depth_map[msk == 0] = 0
            depth = depth_map[coord[:, 0], coord[:, 1]]
            ret["depth"] = depth

        if cfg.use_distill and self.split == "train":
            feat_path = os.path.join(self.data_root, cfg.distill, self.ims[index].replace(".jpg", ".npy"))
            if os.path.isfile(feat_path):
                feat_pretrain = np.load(feat_path)
            else:
                feat_path = os.path.join(self.data_root, cfg.distill, self.ims[index].replace(".jpg", ".npz"))
                feat_pretrain = np.load(feat_path)["arr_0"]
            feat_pretrain = cv2.resize(feat_pretrain, (W, H), interpolation=cv2.INTER_NEAREST)
            feat_pretrain = feat_pretrain[coord[:, 0], coord[:, 1]]
            ret["feat_pretrain"] = feat_pretrain

        if cfg.encoding == "mip":
            radii_full = if_nerf_dutils.get_radii(H, W, K, R, T)
            radii = radii_full[coord[:, 0], coord[:, 1]]
            ret["radii"] = radii

        if self.split != "train":
            ret["img"] = img
            # ret["msk"] = msk
            ret["depth_map"] = depth_map

        if cfg.use_mask_loss:
            mask_ray = orig_msk[coord[:, 0], coord[:, 1]]
            ret["mask_ray"] = mask_ray

        if self.split == "train" and (
            (cfg.aninerf_animation and cfg.learn_bw and cfg.bw_ray_sampling and cfg.bw_use_T2P)
            or cfg.use_hard_surface_loss_canonical
        ):
            tpose_w = (np.dot(tpose, np.linalg.inv(Rh)) + Th).astype(np.float32)
            tbounds_w = if_nerf_dutils.get_bounds(tpose_w)
            _, _, _, tnear, tfar, _, _ = if_nerf_dutils.sample_ray_h36m(
                np.zeros_like(img), np.ones_like(msk), K, R, T, tbounds_w, self.nrays, self.split
            )
            ret["tnear"] = tnear
            ret["tfar"] = tfar

        if cfg.vis_tpose_mesh or cfg.vis_posed_mesh:
            if frame_index not in self.mesh_samples:
                voxel_size = cfg.voxel_size
                x = np.arange(wbounds[0, 0], wbounds[1, 0] + voxel_size[0], voxel_size[0])
                y = np.arange(wbounds[0, 1], wbounds[1, 1] + voxel_size[1], voxel_size[1])
                z = np.arange(wbounds[0, 2], wbounds[1, 2] + voxel_size[2], voxel_size[2])
                mesh_pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).astype(np.float32)
                mesh_inside = self.prepare_inside_pts(mesh_pts, frame_index)
                self.mesh_samples[frame_index] = {"pts": mesh_pts, "inside": mesh_inside}
            ret["mesh_pts"] = self.mesh_samples[frame_index]["pts"]
            ret["mesh_inside"] = self.mesh_samples[frame_index]["inside"]

        if (cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr")) or (
            self.split != "train" and cfg.use_sr != "none"
        ):
            H_sr, W_sr = int(H * cfg.sr_ratio), int(W * cfg.sr_ratio)
            rgb_sr = cv2.resize(img_raw, (W_sr, H_sr), interpolation=cv2.INTER_AREA)
            msk_sr = cv2.resize(msk_raw, (W_sr, H_sr), interpolation=cv2.INTER_NEAREST)
            # rgb_sr[msk_sr == 0] = 0
            # msk_sr = cv2.dilate(msk_sr, np.ones((H_sr // 8, W_sr // 8), np.uint8), 1)
            rgb_sr = rgb_sr[msk_sr != 0].reshape(-1, 3).astype(np.float32)
            ret.update({"rgb_sr": rgb_sr, "msk_sr": msk_sr, "H_sr": H_sr, "W_sr": W_sr})

        if cfg.use_alpha_sdf in ("always", "residual") or (self.split == "train" and cfg.use_alpha_sdf == "train_init"):
            from lib.utils.sdf_utils import repair_mesh

            vertices, faces = repair_mesh(wpts, self.faces)
            ret.update({"vertices": vertices, "faces": faces})

        if cfg.use_bs:
            ret["bs"] = bs

        if cfg.encoding == "uvd":
            ret.update({"mano_vertices": ppts, "mano_faces": self.faces.astype(int), "mano_face_uv": self.mesh_face_uv})

        return ret

    def __len__(self):
        return len(self.ims)
