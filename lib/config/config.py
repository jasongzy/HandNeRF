import argparse
import os

from . import yacs
from .yacs import CfgNode as CN

cfg = CN()

cfg.parent_cfg = "configs/default.yaml"

# experiment name
cfg.exp_name = "hello"

# network
cfg.distributed = False

# data
cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
cfg.test_view = []
cfg.begin_ith_frame = 0  # the first smpl
cfg.num_train_frame = 1  # number of smpls
cfg.num_eval_frame = -1  # number of frames to render
cfg.frame_interval = 1
cfg.smpl = "smpl"
cfg.vertices = "vertices"
cfg.params = "params"
cfg.distill = "dino"
cfg.mask_bkgd = True

cfg.box_padding = 0.05

# mesh
cfg.mesh_th = 50  # threshold of density
cfg.mesh_smooth_iter = 10

# task
cfg.task = "nerf4d"

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = "CocoTrain"
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = ""
cfg.train.batch_sampler = "default"
cfg.train.sampler_meta = CN({"min_hw": [256, 256], "max_hw": [480, 640], "strategy": "range"})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = "adam"
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.0

cfg.train.scheduler = CN({"type": "multi_step", "milestones": [80, 120, 200, 240], "gamma": 0.5})

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = "CocoVal"
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = "default"
cfg.test.batch_sampler = "default"
cfg.test.sampler_meta = CN({"min_hw": [480, 640], "max_hw": [480, 640], "strategy": "origin"})
cfg.test.frame_sampler_interval = 30

# trained model
cfg.trained_model_dir = "data/trained_model"

# recorder
cfg.record_dir = "data/record"
cfg.log_interval = 20
cfg.record_interval = 20

# result
cfg.result_dir = "data/result"
cfg.mesh_dir = "data/mesh"

# training
cfg.training_mode = "default"
cfg.aninerf_animation = False
cfg.init_aninerf = "no_pretrain"
cfg.erode_edge = True

cfg.left_mask_value = 100
cfg.right_mask_value = 200

# evaluation
cfg.skip_eval = False
cfg.test_novel_pose = False
cfg.test_rest_frames = False
cfg.vis_pose_sequence = False
cfg.vis_novel_view = False
cfg.vis_tpose_mesh = False
cfg.vis_posed_mesh = False

cfg.fix_random = False
cfg.seed = 0

cfg.chunk = 2048
cfg.use_amp = False
cfg.torch_compile = False

# data
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.0

cfg.pe_scaling = False

cfg.latent_dim = 128

cfg.bw_hidden_dim = 256
cfg.bw_layer_num = 8
cfg.bw_skips = [5]
cfg.bw_activation = "relu"

cfg.xyz_hidden_dim = 256
cfg.xyz_layer_num = 8
cfg.xyz_skips = [5]
cfg.xyz_activation = "relu"

cfg.density_activation = "relu"
cfg.density_bias = 0.0

cfg.view_activation = "sigmoid"

cfg.rgb_prior = [0.0, 0.0, 0.0]

cfg.poses_hidden_dim = 256
cfg.poses_layer_num = 6
cfg.poses_skips = []
cfg.poses_activation = "relu"

cfg.displ_hidden_dim = 256
cfg.displ_layer_num = 9
cfg.displ_skips = [5]
cfg.displ_activation = ["relu", "tanh"]

cfg.detailed_info = True

cfg.is_interhand = False
cfg.joints_num = 24

cfg.use_perceptual_loss = False
cfg.weight_perceptual = 0.1
cfg.use_depth = False
cfg.depth_loss = "smooth_l1"
cfg.use_mask_loss = False
cfg.use_sparsity_loss = False
cfg.use_distortion_loss = False
cfg.use_color_range_loss = False
cfg.use_hard_surface_loss = False
cfg.use_hard_surface_loss_canonical = False
cfg.use_color_var_loss = False
cfg.use_alpha_sdf = "none"
cfg.use_alpha_loss = False
cfg.weight_bw = 1.0
cfg.weight_depth = 1.0
cfg.weight_displ = 0.1
cfg.weight_color_range = 1.0
cfg.weight_mask = 1.0
cfg.weight_sparsity = 0.1
cfg.weight_distortion = 0.1
cfg.weight_hard_surface = 0.1
cfg.weight_color_var = 10.0
cfg.weight_distill = 1.0
cfg.weight_alpha = 0.1
cfg.prior_knowledge_loss_decay = False

cfg.learn_bw = True
cfg.use_poses = False
cfg.poses_format = "axis_angle"
cfg.use_bs = False
cfg.learn_displacement = False
cfg.use_viewdir = True
cfg.use_pose_viewdir = False
cfg.use_tpose_viewdir = False
cfg.shading_mode = "latent"
cfg.rgb_dim = 256
cfg.exact_depth = False
cfg.encoding = "nerf"
cfg.triplane_res = 256
cfg.triplane_dim = 32
cfg.triplane_reduce = "sum"
cfg.use_neural_renderer = False
cfg.neural_renderer_type = "cnn"
cfg.sr_ratio = 2.0
cfg.use_both_renderer = False
cfg.use_distill = False
cfg.use_sr = "none"
cfg.sr_ckpt = "net_g_latest.pth"

cfg.bw_ray_sampling = False
cfg.bw_use_P2T = True
cfg.bw_use_T2P = True
cfg.poses_layer_policy = "finetune"
cfg.bw_policy = "scratch"

cfg.eval_full_img = False
cfg.render_frame = -1
cfg.render_type = "animation"
cfg.render_view = "forward"
cfg.render_cams = 120
cfg.render_forward_range = [0.33, 0.83]

cfg.hands_share_params = True
cfg.use_hands_align_loss = False
cfg.weight_hands_align = 0.01


def parse_cfg(cfg, args):
    if not cfg.task:
        raise ValueError("task must be specified")

    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)

    # cfg.local_rank = args.local_rank
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))

    # assign the gpus
    if -1 in cfg.gpus:
        assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES must be set for adaptive-GPU mode"
        cfg.gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].strip().replace(" ", "").split(",")))
        if cfg.local_rank == 0:
            print(f"Overwritten cfg.gpus with CUDA_VISIBLE_DEVICES: {cfg.gpus}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])

    cfg.distributed = cfg.distributed or args.launcher not in ("none")
    if not cfg.distributed and len(cfg.gpus) > 1:
        if cfg.local_rank == 0:
            print(f"Setting cfg.distributed to True since {cfg.gpus=}")
        cfg.distributed = True


def make_cfg(args) -> CN:
    with open(args.cfg_file, "r") as f:
        current_cfg = yacs.load_cfg(f)

    cfg.merge_strain(current_cfg)

    cfg.merge_from_list(args.opts)

    if cfg.aninerf_animation:
        cfg.merge_from_other_cfg(cfg.aninerf_animation_cfg)

    if cfg.vis_pose_sequence:
        cfg.merge_from_other_cfg(cfg.pose_sequence_cfg)

    if cfg.vis_novel_view:
        cfg.merge_from_other_cfg(cfg.novel_view_cfg)

    if cfg.train_dataset.hand_type == "both" or cfg.test_dataset.hand_type == "both":
        cfg.merge_from_other_cfg(cfg.interhands_cfg)

    if cfg.vis_tpose_mesh or cfg.vis_posed_mesh:
        cfg.merge_from_other_cfg(cfg.mesh_cfg)

    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)

    if args.epoch is not None:
        cfg.train.epoch = args.epoch
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if cfg.is_interhand:
        cfg.joints_num = 16
        cfg.train.dataset = "InterHand2.6M"
        cfg.test.dataset = "InterHand2.6M"
        if cfg.capture != -1 or cfg.seq:
            # overwrite capture & sequence
            path_split_capture, seq = os.path.split(cfg.train_dataset.data_root)
            path_split, capture = os.path.split(path_split_capture)
            path, split = os.path.split(path_split)
            if cfg.split:
                assert split in ("train", "test", "val")
                split = cfg.split
            else:
                cfg.split = split
            if cfg.capture != -1:
                capture = f"Capture{cfg.capture}"
            else:
                cfg.capture = int(capture.strip("Capture"))
            assert 0 <= cfg.capture <= 26
            if cfg.seq:
                seq = cfg.seq
            else:
                cfg.seq = seq
            data_root = os.path.join(path, split, capture, seq)
            assert os.path.isdir(data_root), f"Invalid data path: {data_root}"
            cfg.train_dataset.data_root = data_root
            cfg.test_dataset.data_root = data_root
        if cfg.num_train_frame == -1:
            with open(os.path.join(cfg.train_dataset.data_root, "frames.txt"), "r") as f:
                train_frame_list = f.readlines()
                train_frame_list = [i for i in train_frame_list if i.strip()]
            cfg.num_train_frame = len(train_frame_list)
        if cfg.num_eval_frame == -1:
            if cfg.train_dataset.data_root == cfg.test_dataset.data_root:
                if cfg.test_rest_frames:
                    with open(os.path.join(cfg.test_dataset.data_root, "frames.txt"), "r") as f:
                        test_frame_list = f.readlines()
                        test_frame_list = [i for i in test_frame_list if i.strip()]
                    cfg.num_eval_frame = len(test_frame_list) - cfg.num_train_frame
                else:
                    cfg.num_eval_frame = cfg.num_train_frame
            else:
                with open(os.path.join(cfg.test_dataset.data_root, "frames.txt"), "r") as f:
                    test_frame_list = f.readlines()
                    test_frame_list = [i for i in test_frame_list if i.strip()]
                cfg.num_eval_frame = len(test_frame_list)
        assert cfg.num_train_frame > 0
        assert cfg.num_eval_frame > 0

    if (
        cfg.eval_full_img
        and cfg.use_sr == "none"
        and not (cfg.use_neural_renderer and cfg.neural_renderer_type in ("cnn_sr", "eg3d_sr"))
    ):
        cfg.test_dataset.ratio = 1.0
    else:
        cfg.test_dataset.ratio = cfg.ratio

    # import pprint
    # pprint.pprint(cfg)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument("--test", action="store_true", dest="test", default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
parser.add_argument("--launcher", type=str, default="none", choices=["none", "pytorch"])
parser.add_argument("--epoch", type=int, default=None)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.type:
    cfg.task = "run"
cfg = make_cfg(args)
