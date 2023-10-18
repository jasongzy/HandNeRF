import torch
import torch.nn as nn

from lib.config import cfg


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class NeRFEncoding(nn.Module):
    """Multi-scale sinusoidal encodings"""

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input=False,
        periodic_fns: list = (torch.sin, torch.cos),
    ):
        super().__init__()

        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Calculates NeRF encoding.
        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
        Returns:
            Output values will be between -1 and 1
        """
        # in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = self.freqs.to(in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        # [sin, sin, ..., cos, cos, ...] --> [sin, cos, sin, cos, ...]
        encoded_inputs_sin, encoded_inputs_cos = torch.chunk(encoded_inputs, 2, dim=-1)
        encoded_inputs = (
            torch.concatenate(
                (encoded_inputs_sin.view(*in_tensor.shape, -1), encoded_inputs_cos.view(*in_tensor.shape, -1)), dim=-2
            )
            .transpose(-1, -2)
            .reshape(*in_tensor.shape[:-1], -1)
        )

        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
        return encoded_inputs


# xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res)
# view_embedder, view_dim = get_embedder(cfg.view_res)

xyz_embedder = NeRFEncoding(
    in_dim=3, num_frequencies=cfg.xyz_res, min_freq_exp=0, max_freq_exp=cfg.xyz_res - 1, include_input=True
)
xyz_dim = xyz_embedder.get_out_dim()
view_embedder = NeRFEncoding(
    in_dim=3, num_frequencies=cfg.view_res, min_freq_exp=0, max_freq_exp=cfg.view_res - 1, include_input=True
)
view_dim = view_embedder.get_out_dim()
