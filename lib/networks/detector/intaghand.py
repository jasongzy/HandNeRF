import os
import sys
import warnings
from functools import lru_cache

import cv2
import torch

FILE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
REPO_DIR = os.path.join(PROJECT_DIR, "tools/IntagHand")
if __name__ == "__main__":
    sys.path.insert(0, PROJECT_DIR)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from tools.IntagHand.apps.eval_interhand import Jr
from tools.IntagHand.models.manolayer import ManoLayer
from tools.IntagHand.models.model import load_model


@lru_cache(maxsize=1)
def get_J_regressor(device="cpu"):
    mano_dir = "misc/mano"
    mano_layer = {
        "left": ManoLayer(os.path.join(REPO_DIR, mano_dir, "MANO_RIGHT.pkl"), center_idx=None),
        "right": ManoLayer(os.path.join(REPO_DIR, mano_dir, "MANO_LEFT.pkl"), center_idx=None),
    }
    if torch.sum(torch.abs(mano_layer["left"].shapedirs[:, 0, :] - mano_layer["right"].shapedirs[:, 0, :])) < 1:
        # print("Fix shapedirs bug of MANO")
        mano_layer["left"].shapedirs[:, 0, :] *= -1
    J_regressor = {
        "left": Jr(mano_layer["left"].J_regressor, device=device),
        "right": Jr(mano_layer["right"].J_regressor, device=device),
    }
    return J_regressor


def get_model(
    cfg_path=os.path.join(REPO_DIR, "misc/model/config.yaml"),
    model_path=os.path.join(REPO_DIR, "misc/model/wild_demo.pth"),
):
    model = load_model(cfg_path)
    state = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except:
        state2 = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state2)
    model.eval()
    # model.cuda()
    return model


def calculate_hand_error(verts_pred: torch.Tensor, verts_gt: torch.Tensor, hand_type="left", device="cpu"):
    """Return joint & vertice mean error in mm."""
    assert hand_type in ("left", "right")

    joints_gt = get_J_regressor(device)[hand_type](verts_gt)

    root_gt = joints_gt[:, 9:10]
    length_gt = torch.linalg.norm(joints_gt[:, 9] - joints_gt[:, 0], dim=-1)
    joints_gt = joints_gt - root_gt
    verts_gt = verts_gt - root_gt

    joints_pred = get_J_regressor(device)[hand_type](verts_pred)

    root_pred = joints_pred[:, 9:10]
    length_pred = torch.linalg.norm(joints_pred[:, 9] - joints_pred[:, 0], dim=-1)
    scale = (length_gt / length_pred).unsqueeze(-1).unsqueeze(-1)

    joints_pred = (joints_pred - root_pred) * scale
    verts_pred = (verts_pred - root_pred) * scale

    joint_loss = torch.linalg.norm((joints_pred - joints_gt), ord=2, dim=-1)
    joint_loss = joint_loss.detach().cpu().numpy()

    vert_loss = torch.linalg.norm((verts_pred - verts_gt), ord=2, dim=-1)
    vert_loss = vert_loss.detach().cpu().numpy()

    return float(joint_loss.mean() * 1000), float(vert_loss.mean() * 1000)


if __name__ == "__main__":
    import argparse
    import glob

    from tqdm import tqdm

    from tools.IntagHand.core.test_utils import InterRender

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=f"{REPO_DIR}/misc/model/config.yaml")
    parser.add_argument("--model", type=str, default=f"{REPO_DIR}/misc/model/wild_demo.pth")
    parser.add_argument("--img_path", type=str, default=f"{REPO_DIR}/demo/")
    parser.add_argument("--save_path", type=str, default=f"{REPO_DIR}/demo/")
    parser.add_argument("--render_size", type=int, default=256)
    args = parser.parse_args()

    model = InterRender(cfg_path=args.cfg, model_path=args.model, render_size=args.render_size)

    img_path_list = glob.glob(os.path.join(args.img_path, "*.jpg")) + glob.glob(os.path.join(args.img_path, "*.png"))
    for img_path in tqdm(img_path_list):
        img_name = os.path.basename(img_path)
        if img_name.find("output.jpg") != -1:
            continue
        img_name = img_name[: img_name.find(".")]
        img = cv2.imread(img_path)
        params = model.run_model(img)
        img_overlap = model.render(params, bg_img=img)
        cv2.imwrite(os.path.join(args.save_path, f"{img_name}_output.jpg"), img_overlap)
