import torch

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()