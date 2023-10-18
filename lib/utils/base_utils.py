import os
import pickle

import numpy as np
import torch
import torch.distributed as dist


def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def write_K_pose_inf(K, poses, img_root):
    K = K.copy()
    K[:2] = K[:2] * 8
    K_inf = os.path.join(img_root, "Intrinsic.inf")
    os.makedirs(os.path.dirname(K_inf), exist_ok=True)
    with open(K_inf, "w") as f:
        for i in range(len(poses)):
            f.write(f"{i}\n")
            f.write("{} {} {}\n {} {} {}\n {} {} {}\n".format(*(K.reshape(9).tolist())))
            f.write("\n")

    pose_inf = os.path.join(img_root, "CamPose.inf")
    with open(pose_inf, "w") as f:
        for pose in poses:
            pose = np.linalg.inv(pose)
            A = pose[0:3, :]
            tmp = np.concatenate([A[0:3, 2].T, A[0:3, 0].T, A[0:3, 1].T, A[0:3, 3].T])
            f.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(tmp.tolist())))


def normalize_batch(
    x: torch.Tensor,
    target_min=0.0,
    target_max=1.0,
    x_min: torch.Tensor = None,
    x_max: torch.Tensor = None,
    dim=-1,
    return_ratio=False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Rescale `x` to [`target_min`, `target_max`].
    If not given, the min/max values are determined by `x` and `dim`.
    """
    if x_min is None:
        x_min = torch.min(x, dim=dim, keepdim=True)[0]
    if x_max is None:
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
    # assert target_min < target_max
    # assert (x_min < x_max).all()

    ratio = (target_max - target_min) / (x_max - x_min)
    x_norm = (x - x_min) * ratio + target_min
    return (x_norm, ratio) if return_ratio else x_norm


def fix_random(seed=0):
    import random

    import torch.backends.cudnn
    import torch.cuda

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["PYTHONHASHSEED"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def init_dist():
    from datetime import timedelta

    from lib.config import cfg

    assert torch.cuda.is_available(), "cuda is not available"
    assert len(cfg.gpus) == torch.cuda.device_count(), f"Specified {cfg.gpus=} but {torch.cuda.device_count()=}"

    assert (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ
    ), "To use distributed mode, use `python -m torch.distributed.launch` or `torchrun` to launch the program"

    # cfg.local_rank = int(os.environ["RANK"]) % torch.cuda.device_count()
    if int(os.environ["LOCAL_WORLD_SIZE"]) <= torch.cuda.device_count():
        backend = "nccl"
        torch.cuda.set_device(cfg.local_rank)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    else:
        if cfg.local_rank == 0:
            print("Using gloo as backend, because NCCL does not support using the same device for multiple ranks")
        backend = "gloo"
        torch.cuda.set_device(cfg.local_rank % torch.cuda.device_count())

    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(hours=10))
    synchronize()


def to_cuda(batch: dict, device="cuda", exclude: list[str] = ("meta")):
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    elif isinstance(batch, (tuple, list, set)):
        batch = [to_cuda(b, device) for b in batch]
        return batch
    elif isinstance(batch, dict):
        for k in batch.keys():
            if k not in exclude:
                batch[k] = to_cuda(batch[k], device)
    return batch
