import torch
def peak_distance_mm(pred, target, spacing=(1,1,1), ignore_bg=True, topk=2):
    if ignore_bg and pred.shape[1] > 1:
        pred   = pred[:, 1:]
        target = target[:, 1:]

    B, C, D, H, W = pred.shape
    result = []

    for b in range(B):
        dist_per_channel = []
        for c in range(C):
            tk = topk if c in [0, 2] else 1  # adjust for channels with multiple peaks

            pred_flat = pred[b, c].view(-1)
            tgt_flat = target[b, c].view(-1)

            pred_topk = torch.topk(pred_flat, tk).indices
            tgt_topk = torch.topk(tgt_flat, tk).indices

            def idx_to_coords(idx):
                z = idx // (H * W)
                y = (idx % (H * W)) // W
                x = idx % W
                return torch.stack([z, y, x], dim=1).float()

            pred_coords = idx_to_coords(pred_topk)
            tgt_coords = idx_to_coords(tgt_topk)

            pred_coords *= torch.tensor(spacing).float()
            tgt_coords *= torch.tensor(spacing).float()

            dists = torch.cdist(pred_coords, tgt_coords)
            min_dists = dists.min(dim=1)[0]
            dist_per_channel.append(min_dists.mean().item())

        result.append(dist_per_channel)

    return torch.tensor(result)
