import math

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch import functional as Fsp
from tqdm import tqdm

from lib.config import cfg


class ResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, kernel_size: int = 3):
        super().__init__()

        self.activation = F.relu
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=kernel_size, padding=padding)
        self.downsample = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        residual = self.downsample(x)
        out = out + residual
        out = self.activation(out)
        return out


class CNNRenderer(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, 256, 1), nn.ReLU(inplace=True))
        self.res1 = ResBlock(256, 128, 128, kernel_size=3)
        self.res2 = ResBlock(128, 64, 64, kernel_size=3)
        self.res3 = ResBlock(64, 32, 32, kernel_size=1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        for layer in (self.conv1, self.res1, self.res2, self.res3, self.conv2):
            x = layer(x)
        return x


class SPResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, kernel_size: int = 3):
        super().__init__()

        self.activation = spconv.SparseSequential(nn.ReLU())
        padding = (kernel_size - 1) // 2
        self.conv1 = spconv.SubMConv2d(in_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = spconv.SubMConv2d(hidden_size, out_size, kernel_size=kernel_size, padding=padding)
        self.downsample = spconv.SubMConv2d(in_size, out_size, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        residual = self.downsample(x)
        # out = out + residual
        out = Fsp.sparse_add(out, residual)
        out = self.activation(out)
        return out


class SPCNNRenderer(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()

        self.conv1 = spconv.SparseSequential(spconv.SubMConv2d(input_dim, 256, kernel_size=1), nn.ReLU())
        self.res1 = SPResBlock(256, 128, 128, kernel_size=3)
        self.res2 = SPResBlock(128, 64, 64, kernel_size=3)
        self.res3 = SPResBlock(64, 32, 32, kernel_size=1)
        self.conv2 = spconv.SparseSequential(spconv.SubMConv2d(32, 3, kernel_size=1), nn.Sigmoid())

    def forward(self, x: spconv.SparseConvTensor, indices, shape, batch_size=1):
        """
        Args:
            x: [batch_size * n_pixel, input_dim]
            indices: [batch_size * n_pixel, 2 or 1+2]
            shape: [H, W]
        Returns:
            [batch_size * n_pixel, 3]
        """
        assert x.shape[0] == indices.shape[0]
        if batch_size == 1 and indices.shape[-1] == 2:
            indices = torch.concat((torch.zeros_like(indices)[:, :1], indices), dim=-1)
        assert indices.shape[-1] == 3
        indices = indices.int()

        x = spconv.SparseConvTensor(x, indices, shape, batch_size)
        for layer in (self.conv1, self.res1, self.res2, self.res3, self.conv2):
            x = layer(x)
        x = x.dense()
        return x


class TransRenderer(nn.Module):
    def __init__(self, dim: int, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_pos=2, scalar=True, ffn=True):
        super().__init__()

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(num_pos, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim),
        )

        if not scalar:
            self.attn_mlp = nn.Sequential(
                nn.Linear(dim, dim * attn_mlp_hidden_mult),
                nn.ReLU(),
                nn.Linear(dim * attn_mlp_hidden_mult, dim),
            )
        self.scalar = scalar

        self.norm = nn.InstanceNorm1d(dim)
        if ffn:
            self.mlp = nn.Sequential(nn.Conv1d(dim, 256, 1), nn.ReLU(inplace=True), nn.Conv1d(256, 3, 1))
            self.activation = torch.sigmoid
        self.ffn = ffn

    @staticmethod
    def max_value(t: torch.Tensor):
        return torch.finfo(t.dtype).max

    @staticmethod
    def batched_index_select(values: torch.Tensor, indices: torch.Tensor, dim=1):
        value_dims = values.shape[(dim + 1) :]
        values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
        indices = indices[(..., *((None,) * len(value_dims)))]
        indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
        value_expand_len = len(indices_shape) - (dim + 1)
        values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

        value_expand_shape = [-1] * len(values.shape)
        expand_slice = slice(dim, (dim + value_expand_len))
        value_expand_shape[expand_slice] = indices.shape[expand_slice]
        values = values.expand(*value_expand_shape)

        dim += value_expand_len
        return values.gather(dim, indices)

    def forward(self, x, pos, num_neighbors: int = None, window_size: int = None, mask=None):
        B, N, D = x.shape

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # prepare mask
        if mask is not None:
            mask = mask[:, :, None] * mask[:, None, :]

        # calculate relative positional embeddings
        if self.scalar:
            pos_emb = self.pos_mlp(pos)
        else:
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
            pos_emb = self.pos_mlp(rel_pos)

        # expand values
        if not self.scalar:
            # v = einops.repeat(v, "b j d -> b i j d", i=n)
            v = v.repeat(N, 1, 1, 1).transpose(1, 0)

        CHUNK = cfg.chunk // 2
        output_all = torch.zeros((B, N, 3 if self.ffn else D), device=x.device)
        for i in tqdm(
            range(0, N, CHUNK),
            desc="transformer",
            leave=False,
            dynamic_ncols=True,
            disable=N <= CHUNK * 2 or cfg.local_rank != 0,
        ):
            q_ = q[:, i : i + CHUNK]

            # queries to keys
            if self.scalar:
                qk_rel = torch.bmm(q_, k.transpose(1, 2))
            else:
                qk_rel = q_[:, :, None, :] - k[:, None, :, :]

            # determine k nearest neighbors for each point, if specified
            if num_neighbors is not None:
                assert 0 < num_neighbors < N
                with torch.no_grad():
                    # rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
                    # rel_dist = rel_pos.norm(dim=-1)
                    rel_dist = torch.cdist(pos[:, i : i + CHUNK], pos, p=torch.inf)
                    if mask is not None:
                        rel_dist.masked_fill_(~mask, self.max_value(rel_dist))
                    dist, indices = rel_dist.topk(num_neighbors, largest=False)
                dim = 1 if self.scalar else 2
                v_ = self.batched_index_select(v, indices, dim=dim)
                pos_emb_ = self.batched_index_select(pos_emb, indices, dim=dim)
                if mask is not None:
                    mask = self.batched_index_select(mask, indices, dim=dim)
                if self.scalar:
                    qk_rel = torch.gather(qk_rel, 2, indices)
                else:
                    qk_rel = self.batched_index_select(qk_rel, indices, dim=dim)
            else:
                v_ = v
                pos_emb_ = pos_emb

            # add relative positional embeddings to value
            v_ = v_ + pos_emb_

            if self.scalar:
                sim = qk_rel
                # from math import sqrt
                # sim = sim / sqrt(D)
            else:
                # use attention mlp, making sure to add relative positional embedding first
                sim = self.attn_mlp(qk_rel + pos_emb_)

            # masking
            if mask is not None:
                sim.masked_fill_(~mask[..., None], -self.max_value(sim))
            # mask sim values outside the window
            if window_size is not None:
                if num_neighbors is None:
                    dist = torch.cdist(pos[:, i : i + CHUNK], pos, p=torch.inf)
                mask_dist = dist > window_size
                sim.masked_fill_(mask_dist, -self.max_value(sim))

            # attention
            attn = sim.softmax(dim=-2)

            # aggregate
            if self.scalar:
                if num_neighbors is None:
                    agg = torch.bmm(attn, v_)
                else:
                    agg = torch.einsum("b i j, b i j d -> b i d", attn, v_)
            else:
                agg = torch.einsum("b i j d, b i j d -> b i d", attn, v_)

            agg = agg + x[:, i : i + CHUNK]
            agg = self.norm(agg)
            if self.ffn:
                output = self.mlp(agg.transpose(1, 2)).transpose(1, 2)
                output = self.activation(output)
            else:
                output = agg
            output_all[:, i : i + CHUNK] = output
            torch.cuda.empty_cache()

        return output_all


class TransCNNRenderer(nn.Module):
    def __init__(self, dim: int, **args):
        super().__init__()
        self.transformer = TransRenderer(dim, ffn=False, **args)
        self.cnn = CNNRenderer(dim)


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    norm = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm[..., None, None])


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.
        behaviour: defines the convolution mode -- correlation (default), using pytorch conv2d,
        or true convolution (kernel is flipped). 2 modes available ``'corr'`` or ``'conv'``.
    Return:
        Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.
    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    # prepare kernel
    b, c, h, w = x.shape
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=x.device, dtype=x.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=x.device, dtype=x.dtype)
        #  str(behaviour).lower() == 'conv':

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        x = F.pad(x, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    x = x.view(-1, tmp_kernel.size(0), x.size(-2), x.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(x, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.in_feature = in_feature
        # self.out_feature = out_feature
        self._make_layer()

    def _make_layer(self):
        self.layer_1 = nn.Conv2d(self.in_feature, self.in_feature * 2, 1, 1, padding=0)
        self.layer_2 = nn.Conv2d(self.in_feature * 2, self.in_feature * 4, 1, 1, padding=0)
        self.blur_layer = Blur()
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor):
        y = x.repeat(1, 4, 1, 1)
        out = self.actvn(self.layer_1(x))
        out = self.actvn(self.layer_2(out))

        out = out + y
        out = F.pixel_shuffle(out, 2)
        out = self.blur_layer(out)

        return out


class CNNRenderer_SR(nn.Module):
    """https://github.com/CrisHY1995/headnerf/blob/main/NetWorks/neural_renderer.py"""

    def __init__(
        self,
        feat_nc=256,
        out_dim=3,
        final_actvn=True,
        min_feat=32,
        sr_ratio=2,
    ):
        super().__init__()
        # assert n_feat == input_dim
        # self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        # self.input_dim = input_dim
        self.n_feat = feat_nc
        self.out_dim = out_dim
        # self.n_blocks = int(math.log2(img_size) - math.log2(featmap_size))
        self.n_blocks = int(math.log2(sr_ratio))
        self.min_feat = min_feat
        self._make_layer()

    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )

        self.rgb_upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
            [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)]
            + [
                nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat), self.out_dim, 1, 1, padding=0)
                for i in range(self.n_blocks)
            ]
        )

        self.feat_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    max(self.n_feat // (2 ** (i)), self.min_feat),
                    max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                    1,
                    1,
                    padding=0,
                )
                for i in range(self.n_blocks)
            ]
        )

        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # res = []
        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x))
        # res.append(rgb)
        net = x
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)

            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
                # res.append(rgb)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        # res.append(rgb)

        return rgb


class EG3D_SR(nn.Module):
    """Superresolution network architectures from the paper Efficient Geometry-aware 3D Generative Adversarial Networks"""

    def __init__(self, feat_nc=256, sr_ratio=2.0, input_res=(256, 167)):
        super().__init__()
        from livehand.models.superresolution import SuperresolutionHybrid

        self.model = SuperresolutionHybrid(
            channels=feat_nc,
            input_resolution=torch.Tensor(input_res),
            sr_factor=sr_ratio,
            sr_num_fp16_res=4,
            sr_antialias=True,
            channel_base=32768,
            channel_max=512,
            fused_modconv_default="inference_only",
        )

    def forward(
        self, rgb_lr: torch.Tensor, feature_map: torch.Tensor, conditioning_signal: torch.Tensor
    ) -> torch.Tensor:
        rgb_sr = self.model(rgb_lr.clone(), feature_map.clone(), ws=conditioning_signal, noise_mode="none")
        return rgb_sr


if __name__ == "__main__":
    rgb_dim = 256
    tt = CNNRenderer_SR(rgb_dim)
    a = torch.rand(2, rgb_dim, 256, 167)
    b = tt(a)
    print(b.size())
