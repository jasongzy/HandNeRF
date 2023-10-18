import torch
import torch.nn as nn


class NegPearson(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert preds.shape == labels.shape, f"Inputs should have the same shape: {preds.shape} vs {labels.shape}."
        preds = preds.view(preds.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2)))
            )

            # # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
            # if (pearson >= 0).data.cpu().numpy():  # torch.cuda.ByteTensor -->  numpy
            #     loss += 1 - pearson
            # else:
            #     loss += 1 - torch.abs(pearson)

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class DistortionLoss(nn.Module):
    """https://github.com/nerfstudio-project/nerfstudio/blob/7c4e139f2e28e4f90e294170c0bd5c30f6e99906/nerfstudio/model_components/losses.py#L146"""

    def __init__(self, is_mip=True):
        super().__init__()
        self.is_mip = is_mip

    def forward(
        self, weights: torch.Tensor, z_vals: torch.Tensor, near: torch.Tensor, far: torch.Tensor
    ) -> torch.Tensor:
        """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.
        .. math::
            \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
            \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}
        where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
        is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.
        Args:
            densities: Predicted sample densities
            weights: Predicted weights from densities and sample locations
        """

        if not self.is_mip:  # mid points --> bounds
            z_vals_bounds = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
            z_vals = torch.cat(
                (
                    2 * z_vals[..., 0:1] - z_vals_bounds[..., 0:1],
                    z_vals_bounds,
                    2 * z_vals[..., -1:] - z_vals_bounds[..., -1:],
                ),
                dim=-1,
            )
        bins = (z_vals - near) / (far - near)
        starts = bins[..., :-1, None]
        ends = bins[..., 1:, None]

        midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

        loss = (
            weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
        )  # (..., num_samples, num_samples)
        loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
        loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

        return loss
