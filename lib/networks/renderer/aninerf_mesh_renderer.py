import os

import mcubes
import numpy as np
import torch
import trimesh

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

from lib.config import cfg


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


class Renderer:
    def __init__(self, net: torch.nn.Module):
        self.net = net
        self.meshes = {}

    def batchify_rays(self, wpts, density_decoder, chunk):
        """Render rays in smaller minibatches to avoid OOM."""
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = density_decoder(wpts[i : i + chunk])
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        K = batch["K"].squeeze(0).cpu().numpy()
        R = batch["R_w2c"].squeeze(0).cpu().numpy()
        T = batch["T_w2c"].squeeze(0).cpu().numpy()
        renderer = pyrender.OffscreenRenderer(
            viewport_width=int(cfg.W * cfg.ratio), viewport_height=int(cfg.H * cfg.ratio)
        )
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5))
        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        scene.add(camera, pose=camera_pose)
        # Use 3 directional lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        trans = [0, 0, 0]
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2]) + trans
        scene.add(light, pose=light_pose)

        if cfg.is_interhand and cfg.test_dataset.hand_type == "both":
            batch_list = [batch["left"], batch["right"]]
            net_list = [self.net.Left, self.net.Right]
        else:
            batch_list = [batch]
            net_list = [self.net]

        frame_index = batch["frame_index"].item()
        if frame_index not in self.meshes:
            self.meshes[frame_index] = []
            for batch_, net_ in zip(batch_list, net_list):
                pts = batch_["mesh_pts"]
                sh = pts.shape

                inside = batch_["mesh_inside"][0].bool()
                pts = pts[0][inside]

                density_decoder = lambda x: net_.calculate_density(x, batch_)
                density = self.batchify_rays(pts, density_decoder, 2048 * 64)

                cube = np.zeros(sh[1:-1])
                inside = inside.detach().cpu().numpy()
                cube[inside == 1] = density

                cube = np.pad(cube, 10, mode="constant")
                vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
                vertices = (vertices - 10) * cfg.voxel_size[0]
                vertices = vertices + batch_["wbounds"][0, 0].detach().cpu().numpy()

                self.meshes[frame_index].append({"vertices": vertices, "triangles": triangles})

        mesh_list = []
        for mesh_vt in self.meshes[frame_index]:
            mesh = trimesh.Trimesh(mesh_vt["vertices"] @ R.T + T.T, mesh_vt["triangles"])
            # labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
            # triangles = triangles[labels == 0]
            # import open3d as o3d
            # mesh_o3d = o3d.geometry.TriangleMesh()
            # mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            # mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
            # mesh_o3d.remove_unreferenced_vertices()
            # vertices = np.array(mesh_o3d.vertices)
            # triangles = np.array(mesh_o3d.triangles)

            # Need to flip x-axis
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)

            if cfg.mesh_smooth_iter > 0:
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=cfg.mesh_smooth_iter)

            mesh_list.append(mesh)

        mesh: trimesh.Trimesh = trimesh.util.concatenate(mesh_list)
        mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh), "mesh")
        rgb_mesh, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        # normals = compute_normal(mesh.vertices, mesh.faces)
        normals = mesh.vertex_normals
        colors = ((0.5 * normals + 0.5) * 255).astype(np.uint8)
        mesh.visual.vertex_colors[:, :3] = colors
        scene.remove_node(mesh_node)
        scene.add(pyrender.Mesh.from_trimesh(mesh), "mesh")
        rgb_normal, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        renderer.delete()

        ret = {
            "vertex": np.array(mesh.vertices),
            "posed_vertex": np.array(mesh.vertices),
            "triangle": np.array(mesh.faces),
            "rgb_mesh": rgb_mesh,
            "rgb_normal": rgb_normal,
        }

        return ret
