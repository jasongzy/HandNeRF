import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img


def bgr_to_rgb(img: np.ndarray):
    return img[..., [2, 1, 0]]


def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0 : (w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0 : (w0 + w1)] = inp1
    return inp


def vertical_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0 : (h0 + h1), :w1, :] = inp1
    else:
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0 : (h0 + h1), :w1] = inp1
    return inp


def transparent_cmap(cmap):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = 0.3
    return mycmap


cmap = transparent_cmap(plt.get_cmap("jet"))


def set_grid(ax, h, w, interval=8):
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def enlarge_box(bounding_box: list[int], ratio=1.5, img_shape: list[int] = None):
    if ratio <= 1:
        return bounding_box
    im_h, im_w = (1e5, 1e5) if img_shape is None else img_shape
    x0, y0, w, h = bounding_box
    new_x0 = max(x0 - w * (ratio - 1) // 2, 0)
    new_y0 = max(y0 - h * (ratio - 1) // 2, 0)
    new_w = min(w * ratio, im_w - new_x0)
    new_h = min(h * ratio, im_h - new_y0)
    return [int(new_x0), int(new_y0), int(new_w), int(new_h)]


def fill_img(pixel_values: np.ndarray, mask: np.ndarray):
    """Convert the pixels into an image
    Args:
        pixel_values: [n_point, dim]
        mask: [H, W]
    Returns:
        [H, W, dim]
    """
    H, W = mask.shape[:2]
    C = pixel_values.shape[-1] if len(pixel_values.shape) != 1 else 1
    img = np.zeros((H, W, C) if C > 1 else (H, W), dtype=pixel_values.dtype)
    img[mask] = pixel_values
    return img


def fill_img_torch(pixel_values: torch.Tensor, coord: torch.Tensor, H: int, W: int):
    """Convert the pixels into an image
    Args:
        pixel_values: [n_batch, n_point, dim]
        coord: [n_point, 2]
    Returns:
        [n_batch, dim, H, W]
    """
    img = torch.zeros((H, W, pixel_values.shape[-1]), device=coord.device)
    img[coord[:, 0], coord[:, 1]] = pixel_values.to(coord.device)
    img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
    return img


def pad_img_to_square(img: np.ndarray):
    H, W = img.shape[:2]
    if H != W:
        new_size = max(H, W)
        img = np.pad(img, ((0, new_size - H), (0, new_size - W), (0, 0)), mode="constant")
        assert img.shape[0] == img.shape[1] == new_size
    return img


def pad_img_to_square_torch(img: torch.Tensor):
    H, W = img.shape[-2:]
    if H != W:
        new_size = max(H, W)
        img = F.pad(img, (0, new_size - W, 0, new_size - H), mode="constant")
        assert img.shape[2] == img.shape[-1] == new_size
    return img


def np2torch_img(img: np.ndarray, bgr2rgb=False, device=None):
    """
    Args:
        img: [<B,> H, W, 3]
    Returns:
        Tensor: [B, 3, H, W]
    """
    if len(img.shape) == 3:
        img = img[np.newaxis, :]
    assert len(img.shape) == 4, f"Invalid shape of img: {img.shape}"
    if bgr2rgb:
        img = bgr_to_rgb(img)
    img_torch = torch.from_numpy(np.transpose(img, (0, 3, 1, 2))).float()
    if device is not None:
        img_torch = img_torch.to(device)
    return img_torch
