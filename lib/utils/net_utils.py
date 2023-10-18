import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from torch import nn


class ResMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | list[int],
        num_layers: int,
        skip_layers: list[int] = None,
        activation: str | tuple[str, str] = "relu",
    ):
        """Linear layers with skip connections.

        Args:
            hidden_dim: the input dims of layers [1, `num_layers` - 2]. \
                        Should be an integer of the same dim for all layers, \
                        or a list of dims for each hidden layer.
            skip_layers: list of layer indices (within open interval (0, `num_layers`)). \
                         Skip connection is added to the input of each given layer, \
                         which means the input dim of that layer is `hidden_dim` + `in_dim`.
            activation: activation function applied to the output of each layer. \
                        Should be a string of the same function name for all layers, \
                        or a tuple of two names for (other_layers, the_last_layer).
        """
        super().__init__()

        assert num_layers > 1, "ResMLP should have more than one layers"
        if skip_layers is None:
            skip_layers = []
        self.skips = skip_layers
        assert all(
            map(lambda x: 0 < x < num_layers, self.skips)
        ), f"Invalid skip_layers: {skip_layers} out of valid range (0, {num_layers})"

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * (num_layers - 1)
        elif isinstance(hidden_dim, list):
            assert (
                len(hidden_dim) == num_layers - 1
            ), f"Invalid hidden_dim list: got {len(hidden_dim)=} but should be {num_layers - 1}"
        else:
            raise TypeError("Invalid hidden_dim")

        if isinstance(activation, str):
            activation = [activation] * 2
        elif isinstance(activation, (tuple, list, set)):
            assert len(activation) == 2, f"Invalid activation list: got {len(activation)=} but should be 2"
        else:
            raise TypeError("Invalid activation")

        self.linears = nn.ModuleList(
            [nn.Conv1d(in_dim, hidden_dim[0], 1)]
            + [
                nn.Conv1d(hidden_dim[i - 1], out_dim if (i == num_layers - 1) else hidden_dim[i], 1)
                if i not in self.skips
                else nn.Conv1d(hidden_dim[i - 1] + in_dim, out_dim if (i == num_layers - 1) else hidden_dim[i], 1)
                for i in range(1, num_layers)
            ]
        )

        self.actvn = nn.ModuleList()
        for actvn_name in activation:
            if actvn_name == "relu":
                self.actvn.append(nn.ReLU())
            elif actvn_name == "softplus":
                self.actvn.append(nn.Softplus())
            elif actvn_name == "sigmoid":
                self.actvn.append(nn.Sigmoid())
            elif actvn_name == "tanh":
                self.actvn.append(nn.Tanh())
            else:
                raise ValueError(f"Invalid density_activation: {actvn_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input size: [batch_size, `in_dim`, length], Output size: [batch_size, `out_dim`, length]"""
        net = x
        for i, l in enumerate(self.linears):
            actvn = self.actvn[1] if (i == len(self.linears) - 1) else self.actvn[0]
            net = actvn(l(net))
            if i + 1 in self.skips:
                net = torch.cat((x, net), dim=1)
        return net


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss -= neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, normalize=True, reduce=True):
    """
    :param vertex_pred:     [b, vn*2, h, w]
    :param vertex_targets:  [b, vn*2, h, w]
    :param vertex_weights:  [b, 1, h, w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma**2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1.0 / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (abs_diff - (0.5 / sigma_2)) * (
        1.0 - smoothL1_sign
    )

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, preds, targets, weights, sigma=1.0, normalize=True, reduce=True):
        return self.smooth_l1_loss(preds, targets, weights, sigma, normalize, reduce)


class AELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ae, ind, ind_mask):
        """
        ae: [b, 1, h, w]
        ind: [b, max_objs, max_parts]
        ind_mask: [b, max_objs, max_parts]
        obj_mask: [b, max_objs]
        """
        # first index
        b, _, h, w = ae.shape
        b, max_objs, max_parts = ind.shape
        obj_mask = torch.sum(ind_mask, dim=2) != 0

        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)
        tag = ae.gather(1, seed_ind).view(b, max_objs, max_parts)

        # compute the mean
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(2) / (ind_mask.sum(2) + 1e-4)

        # pull ae of the same object to their mean
        pull_dist = (tag - tag_mean.unsqueeze(2)).pow(2) * ind_mask
        obj_num = obj_mask.sum(dim=1).float()
        pull = (pull_dist.sum(dim=(1, 2)) / (obj_num + 1e-4)).sum()
        pull /= b

        # push away the mean of different objects
        push_dist = torch.abs(tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2))
        push_dist = 1 - push_dist
        push_dist = F.relu(push_dist, inplace=True)
        obj_mask = (obj_mask.unsqueeze(1) + obj_mask.unsqueeze(2)) == 2
        push_dist = push_dist * obj_mask.float()
        push = ((push_dist.sum(dim=(1, 2)) - obj_num) / (obj_num * (obj_num - 1) + 1e-4)).sum()
        push /= b
        return pull, push


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum):
        super().__init__()

        self.pnum = pnum
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        device = torch.device("cuda")
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

        self.feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2).detach()

    def forward(self, pred, gt, loss_type="L2"):
        pnum = self.pnum
        batch_size = pred.size()[0]
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)
        device = torch.device("cuda")

        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)

        pred_expand = pred.unsqueeze(1)

        dis = pred_expand - gt_expand

        if loss_type == "L2":
            dis = (dis**2).sum(3).sqrt().sum(2)
        elif loss_type == "L1":
            dis = torch.abs(dis).sum(3).sum(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
        # print(min_id)

        # min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)
        # min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
        #                         expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
        # gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

        return torch.mean(min_dis)


class AttentionLoss(nn.Module):
    def __init__(self, beta=4, gamma=0.5):
        super().__init__()

        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))

        loss = 0
        loss -= alpha * edge_beta * torch.log(pred) * gt
        loss -= (1 - alpha) * bg_beta * torch.log(1 - pred) * (1 - gt)
        return torch.mean(loss)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class Ind2dRegL1Loss(nn.Module):
    def __init__(self, type="l1"):
        super().__init__()
        if type == "l1":
            self.loss = F.l1_loss
        elif type == "smooth_l1":
            self.loss = F.smooth_l1_loss

    def forward(self, output, target, ind, ind_mask):
        """ind: [b, max_objs, max_parts]"""
        b, max_objs, max_parts = ind.shape
        ind = ind.view(b, max_objs * max_parts)
        pred = _tranpose_and_gather_feat(output, ind).view(b, max_objs, max_parts, output.size(1))
        mask = ind_mask.unsqueeze(3).expand_as(pred)
        loss = self.loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IndL1Loss1d(nn.Module):
    def __init__(self, type="l1"):
        super().__init__()
        if type == "l1":
            self.loss = F.l1_loss
        elif type == "smooth_l1":
            self.loss = F.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction="sum")
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss


class GeoCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, poly):
        output = F.softmax(output, dim=1)
        output = torch.log(torch.clamp(output, min=1e-4))
        poly = poly.view(poly.size(0), 4, poly.size(1) // 4, 2)
        target = target[..., None, None].expand(poly.size(0), poly.size(1), 1, poly.size(3))
        target_poly = torch.gather(poly, 2, target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(2).sum(2, keepdim=True)
        kernel = torch.exp(-(poly - target_poly).pow(2).sum(3) / (sigma / 3))
        loss = -(output * kernel.transpose(2, 1)).sum(1).mean()
        return loss


def get_model_path(model_dir: str, epoch=-1, verbose=True):
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        if verbose:
            print(colored(f"pretrained model does not exist in '{model_dir}'", "red"))
        return None
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if pth != "latest.pth" and pth.endswith(".pth")]
    if not pths and "latest.pth" not in os.listdir(model_dir):
        return None
    if epoch == -1:
        pth = "latest" if "latest.pth" in os.listdir(model_dir) else max(pths)
    else:
        pth = epoch
    model_path = os.path.join(model_dir, f"{pth}.pth")
    return model_path


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1, verbose=True) -> int:
    if not resume:
        return 0

    model_path = get_model_path(model_dir, epoch, verbose)
    if model_path is None:
        return 0

    if verbose:
        print(colored(f"load model: {model_path}", "yellow"))
    pretrained_model = torch.load(model_path, "cpu")
    net.load_state_dict(pretrained_model["net"])
    optim.load_state_dict(pretrained_model["optim"])
    scheduler.load_state_dict(pretrained_model["scheduler"])
    recorder.load_state_dict(pretrained_model["recorder"])
    return pretrained_model["epoch"] + 1


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.makedirs(model_dir, exist_ok=True)
    model = {
        "net": net.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "recorder": recorder.state_dict(),
        "epoch": epoch,
    }
    if last:
        torch.save(model, os.path.join(model_dir, "latest.pth"))
    else:
        torch.save(model, os.path.join(model_dir, f"{epoch}.pth"))

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if pth != "latest.pth"]
    if len(pths) <= 20:
        return
    os.system(f"rm {os.path.join(model_dir, f'{min(pths)}.pth')}")


def startswith_any(k: str, l: list[str]):
    return any(k.startswith(s) for s in l)


def load_network(
    net: nn.Module, model_dir: str, resume=True, epoch=-1, strict=True, only: list = None, preprocess=None, verbose=True
) -> int:
    if not resume:
        return 0

    model_path = get_model_path(model_dir, epoch, verbose)
    if model_path is None:
        return 0

    if verbose:
        print(colored(f"load network: {model_path}", "yellow"))
    pretrained_model = torch.load(model_path, map_location=None if torch.cuda.is_available() else torch.device("cpu"))
    pretrained_net = pretrained_model["net"]

    if preprocess:
        pretrained_net = preprocess(pretrained_net)

    if only:
        strict = False
        keys = list(pretrained_net.keys())
        for k in keys:
            if not startswith_any(k, only):
                del pretrained_net[k]

    net.load_state_dict(pretrained_net, strict=strict)

    return pretrained_model["epoch"] + 1


def remove_net_prefix(net: OrderedDict, prefix: str):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix) :]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net: OrderedDict, prefix: str):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net: OrderedDict, orig_prefix: str, prefix: str):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix) :]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net: OrderedDict, layers: list[str]):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


def requires_grad(m: nn.Module, req: bool):
    for param in m.parameters():
        param.requires_grad = req


def tensor_sanity_check(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Invalid type: {type(x)}, expected torch.Tensor")
    if (torch.isnan(x).any()) or (torch.isinf(x).any()):
        raise RuntimeError(f"Invalid tensor found: {x}")


def loss_sanity_check(scalar_stats: dict):
    sanity_check = [k if torch.isnan(v) or torch.isinf(v) else None for k, v in scalar_stats.items()]
    if any(sanity_check):
        raise RuntimeError(f"Invalid loss found in {scalar_stats}")


def adaptive_num_emb(network, ckpt_dir: str, latent_dim=128, net_both_hands=False, device=None, verbose=True):
    ckpt_path = get_model_path(ckpt_dir, verbose=verbose)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=None if torch.cuda.is_available() else torch.device("cpu"))
        tnetworks = (network.Left, network.Right) if net_both_hands else (network,)
        ckpt_both_hands = any("Right." in module for module in ckpt["net"])
        prefix = "Right." if ckpt_both_hands else ""
        if f"{prefix}tpose_human.rgb_latent.weight" not in ckpt["net"]:
            return
        num_embeddings_ckpt = ckpt["net"][f"{prefix}tpose_human.rgb_latent.weight"].shape[0]
        num_embeddings_net = tnetworks[0].tpose_human.rgb_latent.num_embeddings
        if num_embeddings_ckpt != num_embeddings_net:
            for tnet in tnetworks:
                tnet.tpose_human.rgb_latent = torch.nn.Embedding(num_embeddings_ckpt, latent_dim, device=device)
                if verbose:
                    print(
                        colored(
                            f"change num_embeddings of rgb_latent from {num_embeddings_net} to {num_embeddings_ckpt}",
                            "yellow",
                        )
                    )
                if hasattr(tnet.deform, "bw_latent"):
                    tnet.deform.bw_latent = torch.nn.Embedding(num_embeddings_ckpt + 1, latent_dim, device=device)
                    if verbose:
                        print(
                            colored(
                                f"change num_embeddings of bw_latent from {num_embeddings_net + 1} to {num_embeddings_ckpt + 1}",
                                "yellow",
                            )
                        )


def load_network_adaptive(
    net: nn.Module,
    model_dir: str,
    resume=True,
    epoch=-1,
    strict=True,
    only: list = None,
    verbose=True,
    adaptive=True,
    latent_dim=128,
    both_to_single=False,
    device="cuda",
) -> int:
    if adaptive:
        adaptive_num_emb(
            net,
            model_dir,
            latent_dim,
            net_both_hands=both_to_single,
            device=device,
            verbose=verbose,
        )
    preprocess = (
        (lambda net: net if any("Right." in module for module in net) else add_net_prefix(net, "Right."))
        if both_to_single
        else None
    )
    return load_network(net, model_dir, resume, epoch, strict, only, preprocess, verbose)
