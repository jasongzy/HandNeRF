import torch
from torchvision.models import VGG16_Weights, vgg16

from lib.utils.img_utils import pad_img_to_square_torch


class VGGPerceptualLoss(torch.nn.Module):
    """https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49"""

    def __init__(self, resize=True):
        super().__init__()
        blocks = []
        blocks.append(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, feature_layers=(0, 1, 2, 3), style_layers=()
    ) -> torch.Tensor:
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = pad_img_to_square_torch(input)
            target = pad_img_to_square_torch(target)
            input = self.transform(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


if __name__ == "__main__":
    perceptual_loss = VGGPerceptualLoss(resize=True)
    device = "cuda"
    loss_fn = perceptual_loss.to(device)
    pred_img = torch.randn(1, 3, 256, 167).to(device)
    gt_img = torch.randn(1, 3, 256, 167).to(device)
    loss = loss_fn(pred_img, gt_img)
    print(loss)
