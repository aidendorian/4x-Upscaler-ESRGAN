import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(36)])
        for p in self.slice.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        x = normalize_vgg(x)
        return self.slice(x)


def normalize_vgg(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def perception_loss(vgg_sr_features, vgg_hr_features):
    loss = nn.functional.l1_loss(vgg_sr_features, vgg_hr_features)
    return loss


pixel_loss = nn.L1Loss()

bce_loss = nn.BCEWithLogitsLoss()


def gan_loss_D(real_pred, fake_pred):
    real_loss = bce_loss(
        real_pred - fake_pred.mean(dim=0, keepdim=True), 
        torch.ones_like(real_pred)
    )
    fake_loss = bce_loss(
        fake_pred - real_pred.mean(dim=0, keepdim=True), 
        torch.zeros_like(fake_pred)
    )
    
    return (real_loss + fake_loss) / 2


def gan_loss_G(real_pred, fake_pred):
    return bce_loss(
        fake_pred - real_pred.mean(dim=0, keepdim=True),
        torch.ones_like(fake_pred)
    )

def color_consistency_loss(sr_imgs, hr_imgs):
    sr_mean = sr_imgs.mean(dim=[2, 3])
    hr_mean = hr_imgs.mean(dim=[2, 3])
    return torch.mean(torch.abs(sr_mean - hr_mean))