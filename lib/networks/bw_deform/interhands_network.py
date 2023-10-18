import torch.nn as nn

from .tpose_nerf_network import Network as TNetwork
from lib.config import cfg


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.Right = TNetwork(hand_type="right")
        if cfg.hands_share_params:
            self.Left = self.Right
        else:
            self.Left = TNetwork(hand_type="left")
        if cfg.use_neural_renderer:
            self.Left.neural_renderer = self.Right.neural_renderer
