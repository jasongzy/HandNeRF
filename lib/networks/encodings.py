import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TriplaneEncoding(nn.Module):
    """Learned triplane encoding
    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].
    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.
    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.
    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)
    This will return a tensor of shape (bs:..., num_components)
    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce="sum",
    ) -> None:
        super().__init__()

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce.lower()
        assert self.reduce in ("sum", "product", "concat", "none"), f"Invalid reduce type: {self.reduce}"

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components if self.reduce != "concat" else self.num_components * 3

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Sample features from this encoder. Expects in_tensor to be in range [-1, 1]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        elif self.reduce == "sum":
            plane_features = plane_features.sum(0).squeeze(-1).T
        elif self.reduce == "concat":
            plane_features = torch.cat(tuple(plane_features), axis=0).squeeze(-1).T  # [flattened_bs, num_components*3]
            return plane_features.reshape(*original_shape[:-1], self.num_components * 3)
        else:
            plane_features = plane_features.squeeze(-1).permute(2, 1, 0)  # [flattened_bs, num_components, 3]
            return plane_features.reshape(*original_shape[:-1], self.num_components, 3)

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid
        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution


class HashEncoding(nn.Module):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Literal["tcnn", "torch"]. Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]]. Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: str = "torch",
        interpolation: str = None,
    ) -> None:
        super().__init__()

        try:
            import tinycudann as tcnn

            TCNN_EXISTS = True
        except ModuleNotFoundError:
            TCNN_EXISTS = False

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.tcnn_encoding = None
        if implementation == "tcnn":
            if TCNN_EXISTS:
                encoding_config = {
                    "otype": "HashGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": min_res,
                    "per_level_scale": growth_factor,
                }
                if interpolation is not None:
                    encoding_config["interpolation"] = interpolation

                self.tcnn_encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config=encoding_config,
                )

            else:
                print("WARNING: Using a slow implementation for the HashEncoding module. ")
        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"
            self.hash_offset = levels * self.hash_table_size
            self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
            self.hash_table *= hash_init_scale
            self.hash_table = nn.Parameter(self.hash_table)

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: [bs, num_levels, 3] Tensor to be hashed
        Returns:
            [bs, num_levels]
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass using pytorch. Significantly slower than TCNN implementation.
        Args:
            in_tensor: [bs, input_dim]
        Returns:
            [bs, output_dim]
        """

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_tensor: [bs, input_dim]
        Returns:
            [bs, output_dim]
        """
        if self.tcnn_encoding is not None:
            bs = in_tensor.shape[:-1]
            in_tensor = in_tensor.reshape(-1, 3)
            return self.tcnn_encoding(in_tensor).reshape(*bs, -1).float()
        return self.pytorch_fwd(in_tensor)


class SphericalHarmonicsEncoding(nn.Module):
    def __init__(self, degree=4):
        super().__init__()
        import tinycudann as tcnn

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
            },
        )

    def get_out_dim(self) -> int:
        return self.encoding.n_output_dims

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        bs = in_tensor.shape[:-1]
        in_tensor = in_tensor.reshape(-1, 3)
        return self.encoding(in_tensor).reshape(*bs, -1).float()


if __name__ == "__main__":
    import time

    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn(1, 8192 * 64, 3).to(device)

    # encoding = TriplaneEncoding(resolution=256, num_components=32, reduce="sum").to(device)
    encoding = HashEncoding(implementation="tcnn").to(device)
    print(encoding.get_out_dim())

    tic = time.time()
    x_enc = encoding(x)
    print(f"{time.time() - tic:.04}s")
    print(x_enc.shape)
