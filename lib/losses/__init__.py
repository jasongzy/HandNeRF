from .perceptual_loss import VGGPerceptualLoss
from .ray_losses import *
from .ssim import SSIMLoss

# import imp
# import os

# from lib.config import cfg


# def get_loss(name):
#     path = os.path.join("lib/losses", cfg.task, f"{name}.py")
#     module = f".{cfg.task}.{name}"
#     loss = imp.load_source(module, path).Loss()
#     return loss
