import datetime
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.config import cfg
from lib.evaluators.if_nerf import Evaluator
from lib.train.recorder import Recorder


class Trainer(object):
    def __init__(self, network: torch.nn.Module):
        if torch.cuda.is_available():
            device_id = cfg.local_rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
            print("Running on CPU ...")
            if cfg.use_amp:
                cfg.use_amp = False
                print("Cannot use AMP on CPU. Disabling.")
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[device_id],
                output_device=device_id,
                # find_unused_parameters=True
            )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device
        if cfg.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        if isinstance(batch, (tuple, list, set)):
            batch = [self.to_cuda(b) for b in batch]
            return batch

        for k in batch:
            if k == "meta":
                continue
            if isinstance(batch[k], (tuple, list, set)):
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = self.to_cuda(batch[k])
            else:
                batch[k] = batch[k].to(self.device)

        return batch

    def add_iter_step(self, batch, iter_step, max_iter_step):
        if isinstance(batch, (tuple, list, set)):
            for batch_ in batch:
                self.add_iter_step(batch_, iter_step, max_iter_step)
        elif isinstance(batch, dict):
            batch["iter_step"] = iter_step
            batch["max_iter_step"] = max_iter_step
            for hand_type in ("left", "right"):
                if cfg.is_interhand and hand_type in batch:
                    self.add_iter_step(batch[hand_type], iter_step, max_iter_step)

    def train(self, epoch: int, data_loader: DataLoader, optimizer: torch.optim.Optimizer, recorder: Recorder):
        max_iter = len(data_loader)
        self.network.train()
        torch.cuda.empty_cache()
        end = time.time()
        with tqdm(
            total=len(data_loader),
            unit="step",
            leave=False,
            dynamic_ncols=True,
            disable=cfg.local_rank != 0 or not cfg.detailed_info,
        ) as _tqdm:
            _tqdm.set_description(f"epoch: {epoch + 1}/{cfg.train.epoch}")
            for iteration, batch in enumerate(data_loader):
                data_time = time.time() - end
                iteration = iteration + 1

                batch = self.to_cuda(batch)
                self.add_iter_step(batch, epoch * max_iter + iteration, cfg.train.epoch * max_iter)
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    output, loss, loss_stats, image_stats = self.network(batch)

                # training stage: loss; optimizer; scheduler
                if cfg.training_mode == "default":
                    optimizer.zero_grad()
                    loss = loss.mean()
                    if cfg.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                        optimizer.step()
                    # # print unused params
                    # for name, param in self.network.named_parameters():
                    #     if param.requires_grad and param.grad is None:
                    #         print(name)
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                if cfg.local_rank > 0:
                    continue

                # data recording stage: loss_stats, time, image_stats
                recorder.step += 1

                loss_stats = self.reduce_loss_stats(loss_stats)
                recorder.update_loss_stats(loss_stats)

                batch_time = time.time() - end
                end = time.time()
                recorder.batch_time.update(batch_time)
                recorder.data_time.update(data_time)

                if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                    # print training state
                    eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    lr = optimizer.param_groups[0]["lr"]
                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                    exp_name = f"exp: {cfg.exp_name}"
                    training_state = "  ".join([exp_name, "eta: {}", "{}", "lr: {:.6f}", "max_mem: {:.0f}"])
                    training_state = training_state.format(eta_string, str(recorder), lr, memory)
                    # print(training_state, end='\r')

                if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                    # record loss_stats and image_dict
                    recorder.update_image_stats(image_stats)
                    recorder.record("train")

                loss_tqdm = {k: float(v.detach()) for k, v in loss_stats.items()}
                _tqdm.set_postfix(**loss_tqdm)
                _tqdm.update()

    def val(self, epoch: int, data_loader: DataLoader, evaluator: Evaluator = None, recorder: Recorder = None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append(f"{k}: {val_loss_stats[k]:.4f}")
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record("val", epoch, val_loss_stats, image_stats)
