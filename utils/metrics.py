import torch

def peak_distance_mm(pred, target, spacing=(1,1,1), ignore_bg=True):
    if ignore_bg and pred.shape[1] > 1:
        pred   = pred[:, 1:]     # drop channel-0
        target = target[:, 1:]
    B, C, D, H, W = pred.shape
    # flatten spatial dims
    pred_flat   = pred.view(B, C, -1)            # (B,C, D*H*W)
    target_flat = target.view(B, C, -1)

    # index of max voxel per sample & class
    pred_idx   = pred_flat.argmax(dim=-1)        # (B,C)  int64
    target_idx = target_flat.argmax(dim=-1)      # (B,C)

    # convert linear idx → (z,y,x) coordinates
    z_pred   = pred_idx   // (H*W)
    y_pred   = (pred_idx  % (H*W)) // W
    x_pred   = pred_idx   %  W

    z_tgt   = target_idx // (H*W)
    y_tgt   = (target_idx % (H*W)) // W
    x_tgt   = target_idx %  W

    # voxel-space difference → world-space (mm)
    dz = (z_pred - z_tgt).float() * spacing[0]
    dy = (y_pred - y_tgt).float() * spacing[1]
    dx = (x_pred - x_tgt).float() * spacing[2]

    dist = torch.sqrt(dx**2 + dy**2 + dz**2)     # (B,C)
    return dist
