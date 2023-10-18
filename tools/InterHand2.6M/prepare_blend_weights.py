"""
Prepare blend weights of grid points
"""

import argparse
import glob
import os
import pickle
from functools import lru_cache

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from psbody.mesh import Mesh


@lru_cache(maxsize=1)
def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def get_smpl_faces(pkl_path):
    smpl = read_pickle(pkl_path)
    faces = smpl["f"]
    return faces


def get_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def process_shapedirs(shapedirs, vert_ids, bary_coords):
    arr = []
    for i in range(3):
        t = barycentric_interpolation(shapedirs[:, i, :][vert_ids], bary_coords)
        arr.append(t[:, np.newaxis, :])
    arr = np.concatenate(arr, axis=1)
    return arr


def batch_rodrigues(poses):
    """poses: N x 3"""
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(rot_mats, joints, parents):
    """
    rot_mats: n_joint x 3 x 3
    joints: n_joint x 3
    parents: n_joint
    """
    num_joint = joints.shape[0]
    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([num_joint, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([num_joint, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms


def get_transform_params(smpl, params):
    """obtain the transformation parameters for linear blend skinning"""
    v_template = np.array(smpl["v_template"])

    # add shape blend shapes
    shapedirs = np.array(smpl["shapedirs"])
    if HAND_TYPE == "left":
        shapedirs[:, 0, :] *= -1
    betas = params["shapes"]
    v_shaped = v_template + np.sum(shapedirs * betas[None], axis=2)

    # add pose blend shapes
    poses = params["poses"].reshape(-1, 3)
    # n_joint x 3 x 3
    rot_mats = batch_rodrigues(poses)

    # obtain the joints
    joints = smpl["J_regressor"].dot(v_shaped)

    # obtain the rigid transformation
    parents = smpl["kintree_table"][0]
    A = get_rigid_transformation(rot_mats, joints, parents)

    # apply global transformation
    R = cv2.Rodrigues(params["Rh"][0])[0]
    Th = params["Th"]

    return A, R, Th, joints


def get_grid_points(xyz: np.ndarray, bounds_padding=0.05, vsize=0.025):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= bounds_padding
    max_xyz += bounds_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    voxel_size = [vsize, vsize, vsize]
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    return pts


def get_bweights(param_path, vertices_path):
    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(SMPL_PATH)
    # mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(SMPL_PATH)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    # create grid points in the pose space
    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    vert_ids, norm = smpl_mesh.closest_vertices(pts, use_cgal=True)
    bweights = smpl["weights"][vert_ids]

    # closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    # vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype("int32"))
    # bweights = barycentric_interpolation(smpl["weights"][vert_ids], bary_coords)

    # # calculate the distance to the smpl surface
    # norm = np.linalg.norm(pts - closest_points, axis=1)

    # A = np.dot(bweights, A.reshape(joints.shape[0], -1)).reshape(-1, 4, 4)
    # can_pts = pts - A[:, :3, 3]
    # R_inv = np.linalg.inv(A[:, :3, :3])
    # can_pts = np.sum(R_inv * can_pts[:, None], axis=2)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], joints.shape[0] + 1).astype(np.float32)

    return bweights


def prepare_blend_weights(frame_ids):
    bweight_dir = os.path.join(LBS_ROOT, "bweights")
    os.makedirs(bweight_dir, exist_ok=True)
    for i in tqdm(frame_ids, desc="blend weights", leave=False, dynamic_ncols=True):
        param_path = os.path.join(PARAM_DIR, f"{i}.npy")
        vertices_path = os.path.join(VERTICES_DIR, f"{i}.npy")
        bweights = get_bweights(param_path, vertices_path)
        bweight_path = os.path.join(bweight_dir, f"{i}.npy")
        np.save(bweight_path, bweights)


def get_tpose_blend_weights():
    i = frame_ids[0]

    params = np.load(os.path.join(PARAM_DIR, f"{i}.npy"), allow_pickle=True).item()
    # vertices = np.load(os.path.join(VERTICES_DIR, f"{i}.npy"))
    faces = get_smpl_faces(SMPL_PATH)
    np.save(os.path.join(LBS_ROOT, "faces.npy"), faces)
    # mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(SMPL_PATH)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    parent_path = os.path.join(LBS_ROOT, "parents.npy")
    np.save(parent_path, smpl["kintree_table"][0])
    joint_path = os.path.join(LBS_ROOT, "joints.npy")
    np.save(joint_path, joints)

    # transform points from the world space to the pose space
    # pxyz = np.dot(vertices - Th, R)

    # transform from frame 0 to canonical T-pose
    bweights = smpl["weights"]
    # A = np.dot(bweights, A.reshape(joints.shape[0], -1)).reshape(-1, 4, 4)
    # txyz = pxyz - A[:, :3, 3]
    # R_inv = np.linalg.inv(A[:, :3, :3])
    # txyz = np.sum(R_inv * txyz[:, None], axis=2)  # v_shaped_posed

    txyz = smpl["v_template"]
    np.save(os.path.join(LBS_ROOT, "tvertices.npy"), txyz)

    smpl_mesh = Mesh(txyz, faces)

    # create grid points in the pose space
    pts = get_grid_points(txyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype("int32"))
    bweights = barycentric_interpolation(smpl["weights"][vert_ids], bary_coords)

    # calculate the distance to the smpl surface
    norm = np.linalg.norm(pts - closest_points, axis=1)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], joints.shape[0] + 1).astype(np.float32)
    bweight_path = os.path.join(LBS_ROOT, "tbw.npy")
    np.save(bweight_path, bweights)

    return bweights


def prepare_tpose_blendshape(frame_ids):
    bs_dir = os.path.join(LBS_ROOT, "bs")
    os.makedirs(bs_dir, exist_ok=True)
    smpl = read_pickle(SMPL_PATH)
    shapedirs = np.array(smpl["shapedirs"])
    if HAND_TYPE == "left":
        shapedirs[:, 0, :] *= -1
    posedirs = np.array(smpl["posedirs"])
    posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T
    # txyz_list = []  # for test
    # txyz_bs_list = []
    for i in tqdm(frame_ids, desc="blend shapes", leave=False, dynamic_ncols=True):
        # 1. shape contribution
        params = np.load(os.path.join(PARAM_DIR, f"{i}.npy"), allow_pickle=True).item()
        betas = params["shapes"]
        bs = np.sum(shapedirs * betas[None], axis=2)

        # 2. pose blend shapes
        pose = params["poses"].reshape(-1, 3)
        rot_mats = batch_rodrigues(pose).reshape(-1, 3, 3)
        pose_feature = (rot_mats[1:, :, :] - np.eye(3)).reshape(-1)
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = np.matmul(pose_feature, posedirs).reshape(-1, 3)
        bs += pose_offsets

        # get v_shaped_posed (to establish grid)
        A, R, Th, joints = get_transform_params(smpl, params)
        vertices = np.load(os.path.join(VERTICES_DIR, f"{i}.npy"))
        pxyz = np.dot(vertices - Th, R)
        bweights = smpl["weights"]
        A = np.dot(bweights, A.reshape(joints.shape[0], -1)).reshape(-1, 4, 4)
        txyz = pxyz - A[:, :3, 3]
        R_inv = np.linalg.inv(A[:, :3, :3])
        txyz = np.sum(R_inv * txyz[:, None], axis=2)
        # txyz_list.append(txyz)
        # txyz_bs_list.append(txyz - bs)

        smpl_mesh = Mesh(txyz, get_smpl_faces(SMPL_PATH))
        pts = get_grid_points(txyz)
        closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts.reshape(-1, 3))
        vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
            closest_points, closest_face.astype("int32")
        )
        bs_grid = barycentric_interpolation(bs[vert_ids], bary_coords)
        bs_grid = bs_grid.reshape(*pts.shape[:3], 3).astype(np.float32)
        np.save(os.path.join(bs_dir, f"{i}.npy"), bs_grid)
    # print(np.var(txyz_list, axis=0).mean())
    # print(np.var(txyz_bs_list, axis=0).mean())  # should all be close to mano.v_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--capture", default=0, type=int)
    parser.add_argument("--seq", default="0051_dinosaur", type=str)
    parser.add_argument("--mano_dir", default=".", type=str)
    parser.add_argument("--hand_type", default="right", type=str, choices=["left", "right", "both"])
    parser.add_argument("--clear_existed", action="store_true", default=False)
    args = parser.parse_args()

    data_root = os.path.join(args.basedir, "images", args.split, f"Capture{args.capture}", args.seq)
    assert os.path.exists(data_root), f"Wrong path: {data_root}"
    data_root = os.path.abspath(data_root)
    print(f"Generating blend weights for {args.split}/Capture{args.capture}/{args.seq} ...")

    if args.clear_existed:
        os.system(f"rm -rf {os.path.join(data_root, 'lbs')}")

    # ims = glob.glob(os.path.join(data_root, "cam400002", "*.jpg"))
    # ims = list(map(lambda x: x.split(args.seq + "/")[1], ims))
    # ims = np.array(sorted(ims))
    # frame_ids = list(map(lambda x: int(x.split("/")[-1][5:-4]), ims))
    with open(os.path.join(data_root, "frames.txt"), "r") as f:
        frame_ids = f.readlines()
    frame_ids = list(map(int, frame_ids))
    assert frame_ids, "no valid frame"

    hand_list = ("left", "right") if args.hand_type == "both" else (args.hand_type,)
    for HAND_TYPE in hand_list:
        LBS_ROOT = os.path.join(data_root, "lbs", HAND_TYPE)
        os.makedirs(LBS_ROOT, exist_ok=True)
        PARAM_DIR = os.path.join(data_root, "params", HAND_TYPE)
        VERTICES_DIR = os.path.join(data_root, "vertices", HAND_TYPE)
        SMPL_PATH = os.path.join(args.mano_dir, f"MANO_{HAND_TYPE.upper()}.pkl")
        get_tpose_blend_weights()
        prepare_blend_weights(frame_ids)
        prepare_tpose_blendshape(frame_ids)
        print(f"Results stored in '{LBS_ROOT}'")
    print("Done!")
