import argparse
import json
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["EGL_DEVICE_ID"] = "0"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# import matplotlib.pyplot as plt
import pyrender
import smplx
import torch
import trimesh
from tqdm import tqdm

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--capture", default=0, type=int)
    parser.add_argument("--seq", default="0000_neutral_relaxed", type=str)
    parser.add_argument("--mano_dir", default=".", type=str)
    parser.add_argument("--hand_type", default="right", type=str, choices=["left", "right", "both"])
    parser.add_argument("--clear_existed", action="store_true", default=False)
    args = parser.parse_args()

    data_root = os.path.join(args.basedir, "images", args.split, f"Capture{args.capture}", args.seq)
    assert os.path.exists(data_root), f"Wrong path: {data_root}"
    print(f"Rendering mesh for {args.split}/Capture{args.capture}/{args.seq} ...")
    annot_root = osp.join(args.basedir, "annotations")

    if args.clear_existed:
        os.system(f"rm -rf {os.path.join(data_root, 'depth')}")
        os.system(f"rm -rf {os.path.join(data_root, 'mask')}")

    cam_list = os.listdir(data_root)
    cam_list = list(filter(lambda x: x.startswith("cam") and len(x) == 9, cam_list))
    cam_list = sorted(list(map(lambda x: x[3:], cam_list)))

    # smplx_path = 'smplx/models'
    # mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
    mano_layer = {
        "right": smplx.create(osp.join(args.mano_dir, "MANO_RIGHT.pkl"), use_pca=False, is_rhand=True),
        "left": smplx.create(osp.join(args.mano_dir, "MANO_LEFT.pkl"), use_pca=False, is_rhand=False),
    }
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer["left"].shapedirs[:, 0, :] - mano_layer["right"].shapedirs[:, 0, :])) < 1:
        # print("Fix shapedirs bug of MANO")
        mano_layer["left"].shapedirs[:, 0, :] *= -1

    with open(osp.join(annot_root, args.split, f"InterHand2.6M_{args.split}_MANO_NeuralAnnot.json")) as f:
        mano_params = json.load(f)
    with open(osp.join(annot_root, args.split, f"InterHand2.6M_{args.split}_camera.json")) as f:
        cam_params = json.load(f)

    for cam_idx in tqdm(cam_list, desc="cam", dynamic_ncols=True):
        save_path = osp.join(data_root, "mask", f"cam{cam_idx}")
        os.makedirs(save_path, exist_ok=True)
        save_path_depth = osp.join(data_root, "depth", f"cam{cam_idx}")
        os.makedirs(save_path_depth, exist_ok=True)

        img_path_list = sorted(glob(osp.join(data_root, f"cam{cam_idx}", "*.jpg")))
        for img_path in tqdm(img_path_list, leave=False, dynamic_ncols=True):
            frame_idx = img_path.split("/")[-1][5:-4]
            frame_idx = int(frame_idx)
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape

            prev_depth = None
            hand_list = ["left", "right"] if args.hand_type == "both" else [args.hand_type]
            scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
            nm = {}
            for hand_type in hand_list:
                # get mesh coordinate
                mano_param = mano_params[str(args.capture)][str(frame_idx)][hand_type]
                if mano_param is None:
                    info = f"{hand_type} hand not found for frame {frame_idx}"
                    if len(hand_list) == 1:
                        raise RuntimeError(info)
                    else:
                        # print(info)
                        continue

                # get MANO 3D mesh coordinates (world coordinate)
                mano_pose = torch.FloatTensor(mano_param["pose"]).view(-1, 3)
                root_pose = mano_pose[0].view(1, 3)
                hand_pose = mano_pose[1:, :].view(1, -1)
                shape = torch.FloatTensor(mano_param["shape"]).view(1, -1)
                trans = torch.FloatTensor(mano_param["trans"]).view(1, 3)
                output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                mesh = output.vertices[0].numpy() * 1000  # meter to milimeter

                # apply camera extrinsics
                cam_param = cam_params[str(args.capture)]
                t, R = (
                    np.array(cam_param["campos"][str(cam_idx)], dtype=np.float32).reshape(3),
                    np.array(cam_param["camrot"][str(cam_idx)], dtype=np.float32).reshape(3, 3),
                )
                t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
                mesh = np.dot(R, mesh.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)

                # mesh
                mesh = mesh / 1000  # milimeter to meter
                mesh = trimesh.Trimesh(mesh, mano_layer[hand_type].faces)
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                mesh.apply_transform(rot)
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
                )
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
                mesh_node = scene.add(mesh, "mesh")
                nm[mesh_node] = 100 if hand_type == "left" else 200

            # add camera intrinsics
            focal = np.array(cam_param["focal"][cam_idx], dtype=np.float32).reshape(2)
            princpt = np.array(cam_param["princpt"][cam_idx], dtype=np.float32).reshape(2)
            camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
            scene.add(camera)

            # renderer
            renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height, point_size=1.0)

            # light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array([0, -1, 1])
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([0, 1, 1])
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([1, 1, 2])
            scene.add(light, pose=light_pose)

            # render
            # rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=nm)
            renderer.delete()
            rgb = rgb[:, :, :3].astype(np.float32)
            depth = depth[:, :, None]
            valid_mask = depth > 0
            if prev_depth is None:
                render_mask = valid_mask
                img = rgb * render_mask + img * (1 - render_mask)
                prev_depth = depth
            else:
                render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth == 0)
                img = rgb * render_mask + img * (1 - render_mask)
                prev_depth = depth * render_mask + prev_depth * (1 - render_mask)

            # save image
            cv2.imwrite(osp.join(save_path, img_path.split("/")[-1].replace(".jpg", ".png")), rgb * render_mask)

            # save depth
            # fig = plt.figure()
            # plt.axis("off")
            # plt.imshow(prev_depth)
            # fig.savefig("depth.png", format='png', bbox_inches='tight', pad_inches = 0)
            np.save(osp.join(save_path_depth, img_path.split("/")[-1].replace(".jpg", ".npy")), prev_depth)

    print("Done!")
