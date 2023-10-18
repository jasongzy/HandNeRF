import os

import cv2
import numpy as np
import trimesh
from termcolor import colored

from lib.config import cfg


class Visualizer:
    def __init__(self):
        self.result_dir = os.path.join(cfg.mesh_dir, cfg.exp_name)
        if cfg.local_rank == 0:
            print(colored(f"the results are saved at {self.result_dir}", "yellow"))

    def visualize(self, output, batch):
        if cfg.vis_tpose_mesh:
            mesh = trimesh.Trimesh(output["vertex"], output["triangle"], process=False)
        else:
            mesh = trimesh.Trimesh(output["posed_vertex"], output["triangle"], process=False)
        # mesh.show()

        os.makedirs(self.result_dir, exist_ok=True)
        result_path = os.path.join(self.result_dir, "tpose_mesh.npy")
        mesh_path = os.path.join(self.result_dir, "tpose_mesh.ply")
        rgb_mesh_path = os.path.join(self.result_dir, "tpose_mesh.png")
        rgb_normal_path = os.path.join(self.result_dir, "tpose_normal.png")

        if cfg.vis_posed_mesh:
            result_dir = os.path.join(self.result_dir, "posed_mesh")
            os.makedirs(result_dir, exist_ok=True)
            frame_index = batch["frame_index"][0].item()
            view_index = batch["cam_ind"][0].item()
            result_path = os.path.join(result_dir, f"frame{frame_index}.npy")
            mesh_path = os.path.join(result_dir, f"frame{frame_index}.ply")
            rgb_mesh_path = os.path.join(result_dir, f"frame{frame_index}_view{view_index:04d}_mesh.png")
            rgb_normal_path = os.path.join(result_dir, f"frame{frame_index}_view{view_index:04d}_normal.png")

        if not os.path.isfile(result_path):
            np.save(result_path, output)
        if not os.path.isfile(mesh_path):
            mesh.export(mesh_path)
        cv2.imwrite(rgb_mesh_path, output["rgb_mesh"])
        cv2.imwrite(rgb_normal_path, output["rgb_normal"])
