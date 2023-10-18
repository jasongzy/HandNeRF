from lib.config import args, cfg  # isort: split

cfg.torch_compile = False

import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from lib.utils.base_utils import init_dist, to_cuda


def run_dataset():
    from lib.datasets import make_data_loader

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm(data_loader):
        pass


def run_network():
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    network = make_network(cfg).to(device)
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm(data_loader):
        batch = to_cuda(batch, device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate(save_depth=False):
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.networks.renderer import make_renderer
    from lib.utils.net_utils import load_network_adaptive

    cfg.perturb = 0
    # cfg.test_dataset.ratio = cfg.ratio

    if cfg.is_interhand and cfg.test_novel_pose and not cfg.aninerf_animation:
        cfg.result_dir = os.path.join(cfg.result_dir, "-".join(cfg.test_dataset.data_root.split("/")[-2:]))
    if cfg.use_sr != "none":
        cfg.result_dir += f"-sr_{cfg.use_sr}"
    elif not cfg.eval_full_img and cfg.test_dataset.ratio != 1:
        cfg.result_dir += f"-ratio_{cfg.test_dataset.ratio}"

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

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=cfg.distributed)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm(data_loader, dynamic_ncols=True, disable=cfg.local_rank != 0):
        batch = to_cuda(batch, device)
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch, save_imgs=True, save_depth=save_depth)
    if cfg.distributed:
        results = evaluator.get_results()
        # evaluator_gather = [None for _ in range(dist.get_world_size())]
        # dist.all_gather_object(evaluator_gather, evaluator)
        results_gather = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(results_gather, results)
        if cfg.local_rank == 0:
            # evaluator.merge_results(evaluator_gather[1:])
            evaluator.merge_results(results_gather[1:])
            evaluator.summarize()
        evaluator.clear_all()
    else:
        evaluator.summarize()


def run_novel_views():
    from novel_views import run_evaluate

    run_evaluate()


def run_visualize():
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.networks.renderer import make_renderer
    from lib.utils.net_utils import load_network
    from lib.visualizers import make_visualizer

    cfg.perturb = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    network = make_network(cfg).to(device)
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch,
        strict=False,
        verbose=cfg.local_rank == 0,
    )
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=cfg.distributed)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm(data_loader, dynamic_ncols=True, disable=cfg.local_rank != 0):
        batch = to_cuda(batch, device)
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch)


def run_evaluate_nv():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm(data_loader):
        batch = to_cuda(batch, device)
        evaluator.evaluate(batch)
    evaluator.summarize()


if __name__ == "__main__":
    if cfg.distributed:
        init_dist()
    if args.type == "depth":
        run_evaluate(save_depth=True)
    else:
        globals()[f"run_{args.type}"]()
