import numpy as np
import torch.utils.data as data

from .novel_views_dataset import Dataset as NVDataset
from .tpose_dataset import Dataset as TDataset
from lib.config import cfg
from lib.utils.if_nerf.if_nerf_data_utils import get_near_far, get_radii, get_rays


def sample_ray_hands(H, W, K, R, T, bounds1, bounds2):
    ray_o, ray_d = get_rays(H, W, K, R, T)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near1, far1, mask_at_box1 = get_near_far(bounds1, ray_o, ray_d)
    near2, far2, mask_at_box2 = get_near_far(bounds2, ray_o, ray_d)

    near1_full = np.full((H * W), np.nan)
    near1_full[mask_at_box1] = near1
    far1_full = np.full((H * W), np.nan)
    far1_full[mask_at_box1] = far1
    near2_full = np.full((H * W), np.nan)
    near2_full[mask_at_box2] = near2
    far2_full = np.full((H * W), np.nan)
    far2_full[mask_at_box2] = far2

    mask_at_box = mask_at_box1 | mask_at_box2
    near1 = near1_full[mask_at_box].astype(np.float32)
    far1 = far1_full[mask_at_box].astype(np.float32)
    near2 = near2_full[mask_at_box].astype(np.float32)
    far2 = far2_full[mask_at_box].astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return ray_o, ray_d, near1, far1, near2, far2, coord, mask_at_box


class Dataset(data.Dataset):
    def __init__(self, dataset_module=TDataset, split="train", ratio=cfg.ratio, **args):
        super().__init__()

        self.dataset_module = dataset_module
        self.split = split

        if dataset_module == TDataset:
            self.Left: TDataset = dataset_module(
                args["data_root"], args["human"], args["ann_file"], split, ratio=ratio, hand_type="left"
            )
            self.Right: TDataset = dataset_module(
                args["data_root"], args["human"], args["ann_file"], split, ratio=ratio, hand_type="right"
            )
        elif dataset_module == NVDataset:
            self.Left: NVDataset = dataset_module(data_root=args["data_root"], ratio=ratio, hand_type="left")
            self.Right: NVDataset = dataset_module(data_root=args["data_root"], ratio=ratio, hand_type="right")
            wpts_mean = np.stack([self.Left.wpts_mean, self.Right.wpts_mean], axis=0).mean(axis=0)
            for dataset in (self.Left, self.Right):
                dataset: NVDataset
                # dataset.wpts_mean = wpts_mean
                dataset.set_T_c2w(wpts_mean)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        left_data = self.Left.__getitem__(index)
        right_data = self.Right.__getitem__(index)

        H = left_data["H"]
        W = left_data["W"]
        K = left_data["K"]
        R = left_data["R_w2c"]
        T = left_data["T_w2c"]

        meta = {
            "H": H,
            "W": W,
            "K": K,
            "R_w2c": R,
            "T_w2c": T,
            "frame_index": left_data["frame_index"],
        }
        if "view_index" in left_data:
            meta["view_index"] = left_data["view_index"]

        if self.split != "train":
            ray_o, ray_d, near_left, far_left, near_right, far_right, coord, mask_at_box = sample_ray_hands(
                H,
                W,
                K,
                R,
                T,
                left_data["wbounds"],
                right_data["wbounds"],
            )
            # msk = np.ones((meta["H"], meta["W"])).astype(np.uint8)
            msk = left_data["msk"] | right_data["msk"]
            occupancy = msk[coord[:, 0], coord[:, 1]]
            data_hands = {
                "occupancy": occupancy,
                "ray_o": ray_o,
                "ray_d": ray_d,
                "mask_at_box": mask_at_box,
                "msk": msk,
            }
            if cfg.encoding == "mip":
                radii = get_radii(H, W, K, R, T)
                radii = radii[coord[:, 0], coord[:, 1]]
                data_hands["radii"] = radii
            # left_data.update(data_hands)
            left_data.update({"near": near_left, "far": far_left})
            # right_data.update(data_hands)
            right_data.update({"near": near_right, "far": far_right})
            meta.update(data_hands)
            if self.dataset_module == TDataset:
                img = left_data["img"] + right_data["img"]
                H, W = img.shape[:2]
                rgb = img[mask_at_box.reshape(H, W)]
                depth_map = left_data["depth_map"] + right_data["depth_map"]
                depth = depth_map[mask_at_box.reshape(H, W)]
                meta.update({"rgb": rgb, "depth": depth, "cam_ind": left_data["cam_ind"]})
                if "msk_sr" in left_data:
                    msk_sr = left_data["msk_sr"] | right_data["msk_sr"]
                    img_sr = np.zeros((msk_sr.shape[0], msk_sr.shape[1], 3), dtype=img.dtype)
                    img_sr[left_data["msk_sr"] != 0] = left_data["rgb_sr"]
                    img_sr[right_data["msk_sr"] != 0] = right_data["rgb_sr"]
                    rgb_sr = img_sr[msk_sr != 0]
                    meta.update(
                        {"rgb_sr": rgb_sr, "msk_sr": msk_sr, "H_sr": left_data["H_sr"], "W_sr": left_data["W_sr"]}
                    )

        ret = {"left": left_data, "right": right_data}
        # compatible with single hand
        ret.update(meta)
        return ret

    def __len__(self):
        return self.Left.__len__()
