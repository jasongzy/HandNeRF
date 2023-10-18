from lib.config import cfg  # isort: split

import os

import imageio.v2 as imageio
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lib.datasets.interhands_dataset import Dataset as IHDataset
from lib.datasets.novel_views_dataset import Dataset as NVDataset
from lib.networks import make_network
from lib.networks.renderer import make_renderer
from lib.utils.base_utils import init_dist, synchronize, to_cuda
from lib.utils.img_utils import fill_img
from lib.utils.net_utils import load_network_adaptive
from lib.utils.vis_utils import generate_gif, save_img


def run_evaluate():
    cfg.perturb = 0
    # cfg.test_dataset.ratio = cfg.ratio

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    network = make_network(cfg).to(device)
    load_network_adaptive(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch,
        strict=False,
        verbose=cfg.local_rank == 0,
        adaptive=cfg.aninerf_animation or cfg.test_novel_pose,
        latent_dim=cfg.latent_dim,
        both_to_single=cfg.is_interhand and cfg.train_dataset.hand_type == "both",
        device=device,
    )
    network.eval()

    if cfg.test_dataset.hand_type == "both":
        dataset = IHDataset(
            NVDataset, data_root=cfg.test_dataset.data_root, split=cfg.test_dataset.split, ratio=cfg.test_dataset.ratio
        )
    else:
        dataset = NVDataset(
            data_root=cfg.test_dataset.data_root,
            ratio=cfg.test_dataset.ratio,
            hand_type=cfg.test_dataset.hand_type,
        )
    if cfg.distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, sampler=sampler
        )
    else:
        data_loader = DataLoader(dataset=dataset, num_workers=cfg.train.num_workers)
    renderer = make_renderer(cfg, network)
    # evaluator = make_evaluator(cfg)
    if cfg.is_interhand and cfg.test_novel_pose and not cfg.aninerf_animation:
        result_dir = os.path.join(cfg.result_dir, "-".join(cfg.test_dataset.data_root.split("/")[-2:]))
    else:
        result_dir = cfg.result_dir
    if cfg.use_sr != "none":
        result_dir += f"-sr_{cfg.use_sr}"
    elif not cfg.eval_full_img and cfg.test_dataset.ratio != 1:
        cfg.result_dir += f"-ratio_{cfg.test_dataset.ratio}"
    result_dir = os.path.join(result_dir, "novel", f"{cfg.render_type}_{cfg.render_view}_{len(dataset)}")
    os.makedirs(result_dir, exist_ok=True)

    for batch in tqdm(data_loader, dynamic_ncols=True, disable=cfg.local_rank != 0):
        batch = to_cuda(batch, device)
        with torch.no_grad():
            output = renderer.render(batch)
        # evaluator.evaluate(output, batch)
        rgb_pred = output["rgb_map"][0].detach().cpu().numpy()
        H, W = batch["H"].item(), batch["W"].item()
        if (cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr")) or cfg.use_sr != "none":
            img_pred = rgb_pred.reshape((batch["H_sr"].item(), batch["W_sr"].item(), 3))
        else:
            mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
            img_pred = fill_img(rgb_pred, mask_at_box.reshape(H, W))
        save_img(img_pred, result_dir, cfg.render_type, int(batch["frame_index"]), int(batch["view_index"]))

    synchronize()
    if cfg.local_rank == 0:
        print(f"images saved in '{result_dir}'")
        generate_gif(result_dir, cfg.render_type, cfg.render_view, int(batch["frame_index"]))


if __name__ == "__main__":
    if cfg.distributed:
        init_dist()
    run_evaluate()
