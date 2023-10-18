import numpy as np
import torch


def repair_mesh(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repair holes to ensure a watertight mesh"""
    import pymeshfix

    vclean, fclean = pymeshfix.clean_from_arrays(v, f)
    return vclean, fclean


@torch.no_grad()
def get_sdf(vertices: torch.Tensor, faces: torch.Tensor, xyz: torch.Tensor, backend="kaolin"):
    """Negative values outside the mesh.
    `pysdf` produces a more reasonable result for non-watertight mesh, but it's unstable and slower.
    Args:
        vertices: [n_vertices, 3]
        faces: [n_faces, 3]
        xyz: [..., 3]
    Returns:
        sdf: [...]
    """
    if backend == "pysdf":  # https://github.com/sxyu/sdf
        from pysdf import SDF

        sdf_fn = SDF(vertices.cpu().numpy(), faces.cpu().numpy())
        sdf = torch.from_numpy(sdf_fn(xyz.reshape(-1, 3).cpu().numpy())).to(xyz)
        sdf = sdf.reshape(*xyz.shape[:-1])
    elif backend == "kaolin":
        from kaolin.metrics.trianglemesh import point_to_mesh_distance
        from kaolin.ops.mesh import check_sign, index_vertices_by_faces

        vertices = vertices.unsqueeze(0).float()
        faces = faces.long()
        face_vertices = index_vertices_by_faces(vertices, faces)
        distance, _, _ = point_to_mesh_distance(xyz.reshape(1, -1, 3), face_vertices)
        sign = check_sign(vertices, faces, xyz.reshape(1, -1, 3))
        sign = (sign.int() - 0.5) * 2
        sdf = torch.sqrt(distance) * sign
        sdf = sdf.reshape(*xyz.shape[:-1])
    else:
        raise ValueError(f"Invalid backend: {backend}")

    return sdf


def sdf_to_density(sdf: torch.Tensor, beta=1.0):
    """According to the CDF of the Laplace distribution (VolSDF)"""
    x = -sdf

    # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
    ind0 = x <= 0
    val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

    # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
    ind1 = x > 0
    val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

    val = torch.zeros_like(sdf)
    val[ind0] = val0
    val[ind1] = val1

    return val


def sdf_to_alpha_hard(sdf: torch.Tensor):
    x = -sdf
    inside = x <= 0
    outside = x > 0
    x[inside] = 1.0
    x[outside] = 0.0
    return x
