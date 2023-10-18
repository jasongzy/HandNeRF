import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import smplx
import torch

torch.set_grad_enabled(False)

CAMERAS = pd.read_csv("cameras.csv")
CAMERAS[["cam_ids"]] = CAMERAS[["cam_ids"]].astype(str)


def get_T_cam2world(cam_pos, cam_rot) -> np.ndarray:
    T_cam2world = np.append(np.linalg.inv(cam_rot), cam_pos.reshape(3, 1), axis=1)
    return T_cam2world


def get_T_world2cam(cam_pos, cam_rot) -> np.ndarray:
    T_cam2world = np.append(cam_rot, -np.dot(cam_rot, cam_pos.reshape(3, 1)), axis=1)
    return T_cam2world


def get_cams(basedir: str, split: str, capture_id: int):
    capture_id = str(capture_id)
    filepath = os.path.join(basedir, f"annotations/{split}/InterHand2.6M_{split}_camera.json")
    with open(filepath) as f:
        cameras = json.load(f)
    campos_dict = {}
    camrot_dict = {}
    focal_dict = {}
    princpt_dict = {}
    for item in ("campos", "camrot", "focal", "princpt"):
        for cam_id in cameras[capture_id][item].keys():
            item_var = np.array(cameras[capture_id][item][cam_id], dtype=np.float32)
            item_dict_var = eval(item + "_dict")
            if cam_id not in item_dict_var:
                item_dict_var[cam_id] = item_var
    K_dict = {}
    R_dict = {}
    T_dict = {}
    for cam_id in campos_dict.keys():
        K_dict[cam_id] = np.array(
            [
                [focal_dict[cam_id][0], 0, princpt_dict[cam_id][0]],
                [0, focal_dict[cam_id][1], princpt_dict[cam_id][1]],
                [0, 0, 1],
            ]
        ).tolist()
        T_cam2world = get_T_world2cam(campos_dict[cam_id], camrot_dict[cam_id])
        R_dict[cam_id] = T_cam2world[:, :3].tolist()
        T_dict[cam_id] = T_cam2world[:, 3:].tolist()  # mm
    cams = {"K": K_dict, "R": R_dict, "T": T_dict}
    return cams


def get_img_paths(basedir: str, split: str, capture_id: int, seq_name: str, cam_ids, keep_black_frames: bool = False):
    all_ims = []
    frame_ids = []
    for cam_id in cam_ids:
        data_root = os.path.join(basedir, "images", split, f"Capture{capture_id}", seq_name, f"cam{cam_id}")
        ims = glob.glob(os.path.join(data_root, "*.jpg"))
        ims = list(map(lambda x: x.split(f"{seq_name}/")[1], ims))
        ims = np.array(sorted(ims))
        if len(frame_ids) == 0:
            frame_ids = list(map(lambda x: int(x.split("/")[-1][5:-4]), ims))
        else:
            frame_ids_new = list(map(lambda x: int(x.split("/")[-1][5:-4]), ims))
            # assert (
            #     frame_ids == frame_ids_new
            # ), f"not all frames are available in cam{cam_id}: {set(frame_ids) - set(frame_ids_new)}"
            if frame_ids != frame_ids_new:
                frame_ids = sorted(list(set(frame_ids) & set(frame_ids_new)))
        all_ims.append(ims)

    if keep_black_frames:
        num_img = min(len(ims) for ims in all_ims)
        all_ims = [ims[:num_img] for ims in all_ims]
        all_ims = np.stack(all_ims, axis=1)
        return all_ims, frame_ids

    print("Filtering out black frames...")
    cameras_valid = list(CAMERAS[(CAMERAS.portrait == 1) & (CAMERAS.RGB == 1)].cam_ids)
    flag = True
    for view in all_ims:
        for frame in view:
            cam = frame.split("/")[0][3:]
            if cam not in cameras_valid:
                continue
            frame_id = int(frame.split("/")[-1][5:-4])
            if (frame_id in frame_ids) and (
                cv2.imread(os.path.join(basedir, "images", split, f"Capture{capture_id}", seq_name, frame)) == 0
            ).all():
                if flag:
                    print("invalid frames: ", end="")
                    flag = False
                print(frame_id, end=" ")
                frame_ids.remove(frame_id)
    if not flag:
        print("")
    all_ims_valid = []
    for view in all_ims:
        view_valid = list(filter(lambda x: int(x.split("/")[-1][5:-4]) in frame_ids, view))
        all_ims_valid.append(view_valid)
    all_ims_valid = np.array(all_ims_valid).transpose()
    return all_ims_valid, frame_ids


def generate_mask(basedir: str, split: str, capture_id: int, seq_name: str):
    data_root = os.path.join(basedir, "images", split, f"Capture{capture_id}", seq_name)
    if os.path.exists(os.path.join(data_root, "images.npy")) and os.path.exists(
        os.path.join(data_root, "annotations.npy")
    ):
        images = np.load(os.path.join(data_root, "images.npy"), allow_pickle=True)
        annotations = np.load(os.path.join(data_root, "annotations.npy"), allow_pickle=True)
    else:
        annopath = os.path.join(basedir, f"annotations/{split}/InterHand2.6M_{split}_data-001.json")
        with open(annopath) as f:
            data = json.load(f)
        images = data["images"]
        images = list(filter(lambda x: x["capture"] == capture_id and x["seq_name"] == seq_name, images))
        np.save(os.path.join(data_root, "images.npy"), images)
        images_ids = list(map(lambda x: x["id"], images))
        annotations = data["annotations"]
        annotations = list(filter(lambda x: x["image_id"] in images_ids, annotations))
        np.save(os.path.join(data_root, "annotations.npy"), annotations)

    for img_info in images:
        img = cv2.imread(os.path.join(basedir, "images", split, img_info["file_name"]))
        image_id = img_info["id"]
        anno = list(filter(lambda x: x["image_id"] == image_id, annotations))[0]
        bbox = anno["bbox"]
        bbox = list(map(int, bbox))

        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        mask[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]] = 255
        img_crop = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        mask2 = cv2.inRange(img_crop, np.array([0, 0, 0]), np.array([20, 20, 20]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask2 = 255 - cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        new_path = img_info["file_name"].split("/")[-2:]
        new_name = new_path[1].replace(".jpg", ".png")
        new_path = os.path.join(data_root, "mask", new_path[0])
        os.makedirs(new_path, exist_ok=True)
        cv2.imwrite(os.path.join(new_path, new_name), mask2)


def batch_rodrigues(rot_vecs, dtype=torch.float32):
    """https://github.com/zju3dv/EasyMocap/blob/master/easymocap/smplmodel/lbs.py#L280

    Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def convert_from_standard_smpl(poses, shapes, trans, joints):
    """
    https://github.com/zju3dv/EasyMocap/blob/master/easymocap/smplmodel/body_model.py#L265

    https://github.com/zju3dv/EasyMocap/blob/master/doc/02_output.md#attention-for-smplsmpl-x-users

    Rh 就是 MANO pose 的前三个参数值；
    Th 与 SMPL 所定义的 trans 不一致：
        trans 是 world to pose 额外平移矢量取负
        由于 global_orient (Rh) 的旋转中心是手腕根节点，而 pose 坐标系的原点却位于手掌中上部
        因此只要 Rh 非 0，那么即使 trans 为 0，world to pose 也存在一个非 0 的平移量
        这导致利用原始的 pose 和 trans 参数进行坐标系变换相当繁琐
    这里先用 poses，shapes 和 trans 获得了标准 SMPL 模型的关节世界坐标
    然后根据手腕根节点的世界坐标（以及 trans）计算 Th
    以便于之后使用简单的公式进行 world to pose 变换：X_p=Rh(X_w-Th)
    """
    if "torch" not in str(type(poses)):
        poses = torch.tensor(poses, dtype=torch.float32)
        shapes = torch.tensor(shapes, dtype=torch.float32)
        trans = torch.tensor(trans, dtype=torch.float32)

    bn = poses.shape[0]
    # process shapes
    if shapes.shape[0] < bn:
        shapes = shapes.expand(bn, -1)

    # N x 3
    j0 = joints[:, 0, :]
    Rh = poses[:, :3].clone()
    # N x 3 x 3
    rot = batch_rodrigues(Rh)
    Tnew = j0 - torch.einsum("bij,bj->bi", rot, j0) + torch.einsum("bij,bi->bj", torch.linalg.inv(rot), trans)
    poses[:, :3] = 0
    res = dict(
        poses=poses.detach().cpu().numpy(),
        shapes=shapes.detach().cpu().numpy(),
        Rh=Rh.detach().cpu().numpy(),
        Th=Tnew.detach().cpu().numpy(),
    )
    return res


def generate_mano(
    basedir: str, split: str, capture_id: int, seq_name: str, mano_dir: str, hand_type: str, frame_ids: list
):
    mano_layer = {
        "right": smplx.create(os.path.join(mano_dir, "MANO_RIGHT.pkl"), use_pca=False, is_rhand=True),
        "left": smplx.create(os.path.join(mano_dir, "MANO_LEFT.pkl"), use_pca=False, is_rhand=False),
    }
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer["left"].shapedirs[:, 0, :] - mano_layer["right"].shapedirs[:, 0, :])) < 1:
        # print("Fix shapedirs bug of MANO")
        mano_layer["left"].shapedirs[:, 0, :] *= -1

    with open(os.path.join(basedir, "annotations", split, f"InterHand2.6M_{split}_MANO_NeuralAnnot.json")) as f:
        mano_params = json.load(f)
    data_root = os.path.join(basedir, "images", split, f"Capture{capture_id}", seq_name)
    params_dir = os.path.join(data_root, "params")
    vertices_dir = os.path.join(data_root, "vertices")
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(vertices_dir, exist_ok=True)
    invalid_frame_ids = []
    for frame_idx in frame_ids:
        hand_list = ["left", "right"] if hand_type == "both" else [hand_type]
        for hand in hand_list:
            mano_param = mano_params[str(capture_id)][str(frame_idx)][hand]
            if mano_param is None:
                info = f"{hand} hand not found for frame {frame_idx}"
                if len(hand_list) == 1:
                    raise RuntimeError(info)
                else:
                    print(info)
                    invalid_frame_ids.append(frame_idx)
                    continue

            os.makedirs(os.path.join(params_dir, hand), exist_ok=True)
            os.makedirs(os.path.join(vertices_dir, hand), exist_ok=True)

            # get MANO 3D mesh coordinates (world coordinate)
            mano_pose = torch.FloatTensor(mano_param["pose"]).view(-1, 3)
            root_pose = mano_pose[0].view(1, 3)
            hand_pose = mano_pose[1:, :].view(1, -1)
            shape = torch.FloatTensor(mano_param["shape"]).view(1, -1)
            trans = torch.FloatTensor(mano_param["trans"]).view(1, 3)
            output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].numpy()  # meter
            pose_mean = mano_layer[hand].pose_mean.unsqueeze(0).detach().numpy()

            np.save(os.path.join(vertices_dir, hand, f"{frame_idx}.npy"), mesh)

            # params = {}
            # params["poses"] = np.array(mano_param["pose"]).reshape(1, -1)
            # params["shapes"] = np.array(mano_param["shape"]).reshape(1, -1)
            params = convert_from_standard_smpl(
                np.array(mano_param["pose"]).reshape(1, -1) + pose_mean,
                np.array(mano_param["shape"]).reshape(1, -1),
                np.array(mano_param["trans"]).reshape(1, -1),
                output.joints,
            )
            # params["Th"] = np.array(mano_param["trans"]).reshape(1, 3)  # Th != trans

            # test world to pose
            # mesh_pose = mano_layer[hand](hand_pose=hand_pose, betas=shape).vertices[0].detach().numpy()
            # assert ((mesh_pose - np.dot(mesh - params["Th"], cv2.Rodrigues(root_pose.numpy())[0])) < 1e-5).all()
            np.save(os.path.join(params_dir, hand, f"{frame_idx}.npy"), params)

            tpath = os.path.join(params_dir, hand, "tpose.npy")
            # if not os.path.exists(tpath):
            if frame_ids.index(frame_idx) == 0:
                pose_canonical = pose_mean.copy()
                pose_canonical[0, 2] = np.pi / 2
                params_canonical = convert_from_standard_smpl(
                    pose_canonical, np.array(mano_param["shape"]).reshape(1, -1), np.zeros([1, 3]), output.joints
                )
                np.save(tpath, params_canonical)

    return invalid_frame_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--capture", default=0, type=int)
    parser.add_argument("--seq", default="0051_dinosaur", type=str)
    parser.add_argument("--keep_landscape", action="store_true", default=False)
    parser.add_argument("--keep_black_frames", action="store_true", default=False)
    parser.add_argument("--mano_dir", default=".", type=str)
    parser.add_argument("--hand_type", default="right", type=str, choices=["left", "right", "both"])
    parser.add_argument("--clear_existed", action="store_true", default=False)
    args = parser.parse_args()

    data_root = os.path.join(args.basedir, "images", args.split, f"Capture{args.capture}", args.seq)
    assert os.path.exists(data_root), f"Wrong path: {data_root}"
    print(f"Generating annots for {args.split}/Capture{args.capture}/{args.seq} ...")

    if args.clear_existed:
        os.system(f"rm {os.path.join(data_root, '*.npy')}")
        os.system(f"rm {os.path.join(data_root, '*.txt')}")
        os.system(f"rm -rf {os.path.join(data_root, 'params')}")
        os.system(f"rm -rf {os.path.join(data_root, 'vertices')}")

    cams = get_cams(args.basedir, args.split, args.capture)
    # print(cams["K"].keys())

    if not args.keep_landscape:
        # keep portrait (vertical), remove landscape (horizontal)
        # id_remove = []
        # for cam_id in cams["K"].keys():
        #     H, W = 512, 334
        #     ims = glob.glob(os.path.join(data_root, f"cam{cam_id}", "*.jpg"))
        #     shape = cv2.imread(ims[0]).shape
        #     if shape[0] != H or shape[1] != W:
        #         id_remove.append(cam_id)
        id_remove = list(CAMERAS[CAMERAS.portrait == 0].cam_ids)
        for k in ("K", "R", "T"):
            for cam_id in id_remove:
                if cam_id in cams[k]:
                    del cams[k][cam_id]
        # print(cams["K"].keys())
    with open(os.path.join(data_root, "cams.txt"), "w") as f:
        f.write("\n".join(list(cams["K"].keys())))

    img_paths, frame_ids = get_img_paths(
        args.basedir, args.split, args.capture, args.seq, cams["K"].keys(), args.keep_black_frames
    )

    # generate_mask(args.basedir, args.split, args.capture, args.seq)
    invalid_frame_ids = generate_mano(
        args.basedir, args.split, args.capture, args.seq, args.mano_dir, args.hand_type, frame_ids
    )

    if invalid_frame_ids:
        for invalid_frame in invalid_frame_ids:
            frame_ids.remove(invalid_frame)
        img_paths = np.array(
            list(filter(lambda x: int(x[0].split("/")[-1][5:-4]) not in invalid_frame_ids, list(img_paths)))
        )
        assert len(frame_ids) == img_paths.shape[0]
    print(f"number of valid frames: {len(frame_ids)}")
    with open(os.path.join(data_root, "frames.txt"), "w") as f:
        f.write("\n".join(list(map(str, frame_ids))))

    annot = {"cams": cams}
    ims = []
    for img_path in img_paths:
        data = {"ims": img_path.tolist()}
        ims.append(data)
    annot["ims"] = ims

    np.save(os.path.join(data_root, "annots.npy"), annot)
    # np.save(os.path.join(data_root, "annots_python2.npy"), annot, fix_imports=True)

    print("Done!")
