import numpy as np
import torch
import torch.nn.functional as F


def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_point, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def world_dirs_to_pose_dirs(wdirs, Rh):
    """
    wdirs: n_batch, n_point, 3
    Rh: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, Rh)
    return pts


def pose_points_to_world_points(ppts, Rh, Th):
    """
    ppts: n_batch, n_point, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(ppts, Rh.transpose(1, 2)) + Th
    return pts


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def pose_points_to_tpose_points(ppts, bw, A):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_point, 3
    bw: n_batch, n_joint, n_point
    A: n_batch, n_joint, 4, 4
    """
    sh = ppts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], bw.shape[2], -1))
    A = A.view(sh[0], -1, 4, 4)
    pts = ppts - A[..., :3, 3]
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)
    return pts


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def pose_dirs_to_tpose_dirs(ddirs, bw, A):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_point, 3
    bw: n_batch, n_joint, n_point
    A: n_batch, n_joint, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], bw.shape[2], -1))
    A = A.view(sh[0], -1, 4, 4)
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * ddirs[:, :, None], dim=3)
    return pts


def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_point, 3
    bw: n_batch, n_joint, n_point
    A: n_batch, n_joint, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], bw.shape[2], -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw, A):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_point, 3
    bw: n_batch, n_joint, n_point
    A: n_batch, n_joint, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], bw.shape[2], -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * ddirs[:, :, None], dim=3)
    return pts


def grid_sample_blend_weights(grid_coords, bw):
    # the blend weight is indexed by xyz
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw, grid_coords, padding_mode="border", align_corners=True)
    bw = bw[:, :, 0, 0]
    return bw


def pts_sample_blend_weights(pts: torch.Tensor, bw: torch.Tensor, bounds: torch.Tensor):
    """sample blend weights for points
    pts: n_batch, n_point, 3
    bw: n_batch, d, h, w, n_joint + 1
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw, grid_coords, padding_mode="border", align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x N_samples x n_joint x 3
    bw: batch_size x n_joint x 64 x 64 x 64
    """
    bws = []
    for i in range(bw.shape[1]):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = F.grid_sample(bw[:, i : i + 1], nf_grid_coords_, padding_mode="border", align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = torch.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, N_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = torch.rand([sh[0], N_samples])
    y_vals = torch.rand([sh[0], N_samples])
    z_vals = torch.rand([sh[0], N_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts


def axis_angle_to_norm(axis_angle: np.ndarray) -> np.ndarray:
    """Seperate axis-angle into the unit axis vector and the angle value
    axis_angle: [..., N*3]
    Return: [..., N*4]
    """
    from sklearn.preprocessing import normalize

    shape = list(axis_angle.shape)
    axis_angle = axis_angle.reshape(-1, 3)
    norm = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis_angle_unit = normalize(axis_angle, axis=1)
    axis_angle_norm = np.concatenate((axis_angle_unit, norm), axis=-1)
    shape[-1] = int(shape[-1] / 3 * 4)
    axis_angle_norm = axis_angle_norm.reshape(*shape)
    return axis_angle_norm


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """
    Args:
        axis_angle: [..., N*3]
    Return:
        quaternion: [..., N*4]
    """
    from scipy.spatial.transform import Rotation as R

    shape = list(axis_angle.shape)
    axis_angle = axis_angle.reshape(-1, 3)
    quaternion = R.from_rotvec(axis_angle).as_quat()
    shape[-1] = int(shape[-1] / 3 * 4)
    quaternion = quaternion.reshape(*shape)
    quaternion = quaternion.astype(axis_angle.dtype)
    return quaternion


def get_vertices_from_pose(
    mano_model, pose: np.ndarray, global_orient: np.ndarray = None, shape: np.ndarray = None, transl: np.ndarray = None
) -> np.ndarray:
    mano_pose = torch.FloatTensor(pose).view(-1, 3)
    global_orient = (
        torch.zeros((1, 3), dtype=torch.float32)
        if global_orient is None
        else torch.FloatTensor(global_orient).view(1, 3)
    )
    transl = torch.zeros((1, 3), dtype=torch.float32) if transl is None else torch.FloatTensor(transl).view(1, 3)
    hand_pose = mano_pose[1:, :].view(1, -1)
    shape = torch.zeros((1, 10), dtype=torch.float32) if shape is None else torch.FloatTensor(shape).view(1, 10)
    with torch.no_grad():
        mano_output = mano_model(global_orient=global_orient, hand_pose=hand_pose, betas=shape, transl=transl)
    return mano_output.vertices[0].numpy()


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


def barycentric_interpolation(val: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def get_bw_from_vertices(smpl: dict, vertices: np.ndarray, is_lhand=False):
    from psbody.mesh import Mesh

    faces = smpl["f"]
    v_template = np.array(smpl["v_template"])
    shapedirs = np.array(smpl["shapedirs"])
    if is_lhand:
        # Fix shapedirs bug of MANO
        shapedirs[:, 0, :] *= -1
    betas = np.zeros((1, 10))
    v_shaped = v_template + np.sum(shapedirs * betas[None], axis=2)
    joints = smpl["J_regressor"].dot(v_shaped)
    smpl_mesh = Mesh(vertices, faces)

    # create grid points in the pose space
    pts = get_grid_points(vertices)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    vert_ids, norm = smpl_mesh.closest_vertices(pts, use_cgal=True)
    bweights = smpl["weights"][vert_ids]
    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], joints.shape[0] + 1).astype(np.float32)

    return bweights


def batch_rodrigues(poses: np.ndarray) -> np.ndarray:
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


def get_bs_from_shape_and_pose(smpl: dict, params: dict, vertices: np.ndarray, is_lhand=False):
    from psbody.mesh import Mesh

    def get_rigid_transformation(rot_mats: np.ndarray, joints: np.ndarray, parents: np.ndarray):
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

    shapedirs = np.array(smpl["shapedirs"])
    if is_lhand:
        # Fix shapedirs bug of MANO
        shapedirs[:, 0, :] *= -1
    posedirs = np.array(smpl["posedirs"])
    posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T

    # 1. shape contribution
    shape = params["shapes"].reshape(1, 10)
    bs = np.sum(shapedirs * shape[None], axis=2)

    # 2. pose blend shapes
    pose = params["poses"].reshape(-1, 3)
    rot_mats = batch_rodrigues(pose).reshape(-1, 3, 3)
    pose_feature = (rot_mats[1:, :, :] - np.eye(3)).reshape(-1)
    # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = np.matmul(pose_feature, posedirs).reshape(-1, 3)
    bs += pose_offsets

    # get v_shaped_posed (to establish grid)
    v_template = np.array(smpl["v_template"])
    v_shaped = v_template + np.sum(shapedirs * shape[None], axis=2)
    rot_mats = batch_rodrigues(pose)
    joints = smpl["J_regressor"].dot(v_shaped)
    parents = smpl["kintree_table"][0]
    A = get_rigid_transformation(rot_mats, joints, parents)

    bweights = smpl["weights"]
    A = np.dot(bweights, A.reshape(joints.shape[0], -1)).reshape(-1, 4, 4)
    txyz = vertices - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    txyz = np.sum(R_inv * txyz[:, None], axis=2)

    smpl_mesh = Mesh(txyz, smpl["f"])
    # smpl_mesh.write_obj("test.obj")
    pts = get_grid_points(txyz)
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts.reshape(-1, 3))
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype("int32"))
    bs_grid = barycentric_interpolation(bs[vert_ids], bary_coords)
    bs_grid = bs_grid.reshape(*pts.shape[:3], 3).astype(np.float32)
    return bs_grid
