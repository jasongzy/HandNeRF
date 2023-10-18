from lib.config import args, cfg  # isort: split

import importlib
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
from termcolor import colored
from tqdm import tqdm

from lib.datasets import make_data_loader
from lib.evaluators import make_evaluator
from lib.networks import make_network
from lib.networks.renderer import make_renderer
from lib.train import make_lr_scheduler, make_optimizer, make_recorder, make_trainer, set_lr_scheduler
from lib.utils.base_utils import fix_random, init_dist, synchronize
from lib.utils.net_utils import load_model, load_network, save_model

if cfg.fix_random:
    fix_random(seed=cfg.seed)


def get_time_hms(seconds: int):
    hours = seconds // 3600
    seconds -= hours * 3600
    minutes = seconds // 60
    seconds -= minutes * 60
    return int(hours), int(minutes), int(seconds)


def train():
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(
        network, optimizer, scheduler, recorder, cfg.trained_model_dir, resume=cfg.resume, verbose=cfg.local_rank == 0
    )
    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True, is_distributed=cfg.distributed, max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False, is_distributed=cfg.distributed)

    synchronize()

    if cfg.local_rank == 0:
        # if not os.path.exists(cfg.record_dir):
        #     os.makedirs(cfg.record_dir)
        with open(os.path.join(cfg.record_dir, "cfg.yaml"), "w") as f:
            from contextlib import redirect_stdout  # fmt: skip
            with redirect_stdout(f):
                print(cfg.dump())

        if not cfg.resume and os.path.exists(cfg.trained_model_dir):
            print(colored(f"remove contents of directory {cfg.trained_model_dir}", "red"))
            os.system(f"rm -rf {cfg.trained_model_dir}")

        # test_tensor = torch.zeros((cfg.N_rand * cfg.N_samples, 3), device=trainer.device)
        # test_inputs = (
        #     (test_tensor, test_tensor) if cfg.encoding == "mip" else test_tensor,
        #     test_tensor,
        #     test_tensor[..., 0],
        #     trainer.to_cuda(next(iter(train_loader))),
        # )

        # if cfg.train_dataset.hand_type != "both":
        # import warnings  # fmt: skip
        # warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # recorder.writer.add_graph(network, test_inputs, use_strict_trace=False)

        # from torchinfo import summary  # fmt: skip
        # summary(network, input_data=test_inputs)

        # from fvcore.nn import FlopCountAnalysis, parameter_count_table  # fmt: skip
        # print(parameter_count_table(network))
        # flops = FlopCountAnalysis(network, test_inputs)
        # print(f"FLOPs: {flops.total() / 1e6:.2f}M")

        total_params = sum([param.nelement() for param in network.parameters()])
        print("Number of params: " + colored(f"{total_params / 1e6:.4f}M", "blue"))

        print("Begin experiment: " + colored(f"{cfg.exp_name}", "blue"))
        print("Training dataset: " + colored(f"{cfg.train_dataset.data_root}", "blue"))

        time_train_begin = time.time()

    for epoch in tqdm(
        range(begin_epoch, cfg.train.epoch),
        desc="train",
        unit="epoch",
        leave=True,
        dynamic_ncols=True,
        disable=cfg.local_rank != 0,
    ):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()
        synchronize()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch, last=True)

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            # trainer.val(epoch, val_loader, evaluator, recorder)
            trainer.network.eval()
            torch.cuda.empty_cache()
            # for batch in tqdm(val_loader):
            batch = next(iter(val_loader))
            batch = trainer.to_cuda(batch)
            renderer = make_renderer(cfg, network)
            with torch.no_grad():
                output = renderer.render(batch)
            results = evaluator.evaluate_(output, batch, plot_depth=True, joint_metrics=False)
            recorder.writer.add_scalar("val/PSNR", results["psnr"], epoch + 1)
            recorder.writer.add_scalar("val/SSIM", results["ssim"], epoch + 1)
            recorder.writer.add_scalar("val/LPIPS", results["lpips"], epoch + 1)
            recorder.writer.add_scalar("val/depth_l1", results["depth_l1"], epoch + 1)
            # save images
            img_pred = (results["img_pred"] * 255).astype(np.uint8)  # H, W, C
            img_pred = img_pred.transpose(2, 0, 1)
            recorder.writer.add_image("val", img_pred, epoch + 1)
            recorder.writer.add_image("val/depth", results["depth_plot_pred"].transpose(2, 0, 1), epoch + 1)

    if cfg.local_rank == 0:
        time_train_end = time.time()
        print("Training cost: " + "{}h {}m {}s".format(*get_time_hms(time_train_end - time_train_begin)))

    if cfg.torch_compile:  # disable torch.compile for nerf_net_utils.raw2outputs
        from lib.networks.renderer import nerf_net_utils  # fmt: skip
        cfg.torch_compile = False
        importlib.reload(nerf_net_utils)
    torch.cuda.empty_cache()
    trainer.network.eval()
    renderer = make_renderer(cfg, network)
    evaluator.clear_all()
    for batch in tqdm(val_loader, desc="eval", leave=True, dynamic_ncols=True, disable=cfg.local_rank != 0):
        # with next(iter(val_loader)) as batch:
        batch = trainer.to_cuda(batch)
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch, save_imgs=True, save_depth=False)
    if cfg.distributed:
        results = evaluator.get_results()
        # evaluator_gather = [None for _ in range(dist.get_world_size())]
        # dist.all_gather_object(evaluator_gather, evaluator)
        results_gather = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(results_gather, results)
        if cfg.local_rank == 0:
            # evaluator.merge_results(evaluator_gather[1:])
            evaluator.merge_results(results_gather[1:])
            evaluator.summarize(writer=recorder.writer)
        evaluator.clear_all()
    else:
        evaluator.summarize(writer=recorder.writer)

    if cfg.local_rank == 0:
        time_eval_end = time.time()
        print("Evaluating cost: " + "{}h {}m {}s".format(*get_time_hms(time_eval_end - time_train_end)))

    return network


def test():
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


if __name__ == "__main__":
    if cfg.distributed:
        init_dist()
    if args.test:
        test()
    else:
        train()
