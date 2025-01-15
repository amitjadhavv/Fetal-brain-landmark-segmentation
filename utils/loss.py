import torch

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Ensure predictions are probabilities
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return 1 - dice.mean()

import torch

def binary_weighted_dice_loss(pred, target, weight=1.0, smooth=1e-6):
    """
    Computes the binary weighted Dice loss for 3D segmentation.

    Args:
        pred (torch.Tensor): Predicted probabilities (after sigmoid). Shape: (batch_size, x, y, z)
        target (torch.Tensor): Ground truth binary mask. Shape: (batch_size, x, y, z)
        weight (float): Weight for the foreground class. Default is 1.0.
        smooth (float): Smoothing term to avoid division by zero.

    Returns:
        torch.Tensor: Weighted Dice loss.
    """
    # Flatten the predictions and target masks
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()

    # Compute Dice coefficient
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

    # Apply weighting: the weight is applied to the foreground (positive class)
    weighted_dice = dice * weight + (1 - dice) * (1 - weight)

    # Dice loss is 1 - Dice coefficient
    return 1 - weighted_dice

# Example usage
# class_weights = torch.tensor([class_weights[cls] for cls in sorted(class_counts.keys())], dtype=torch.float32)
# loss = weighted_dice_loss(predictions, targets, class_weights)
def multi_class_dice_loss(pred, target, num_classes, smooth=1e-6):
    dice_loss = 0
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_loss += (1 - dice)

    return dice_loss / num_classes