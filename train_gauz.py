import functorch.dim
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.AttentionVNet import AttentionVNet
import torchio as tio
from configs.config import Config
from utils.metrics import peak_distance_mm
from monai.losses import DiceLoss
import  json
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from torchmetrics.functional import jaccard_index
from utils.loss import total_variation_loss_3d
# Define augmentations using torchio
import warnings
warnings.filterwarnings("ignore")

transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),          # Randomly flip along axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=30),  # Apply random scaling and rotation
    tio.RandomNoise(mean=0.0, std=0.1)
    # Add random noise
])
# Load dataset
image_paths = Config.get_image_paths()
mask_paths = Config.get_heatmap_paths()
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform, augmentation_factor=1)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print(len(train_dataloader))
# class_weight calculation
norm_class_weights = [0.00101918, 0.23416625, 0.04894709, 0.14261888, 0.5732486 ]
# Nrmalize so all weights sum to 1
norm_class_weights = torch.tensor(norm_class_weights, dtype=torch.float32).to(Config.DEVICE)
print("Normalized Class Weights:", norm_class_weights)

# Initialize model
model = AttentionVNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
dice_loss = DiceLoss(include_background=True, softmax=True, reduction="mean", weight=norm_class_weights)
# ce_loss = nn.CrossEntropyLoss(weight=norm_class_weights)
# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
# Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

def combined_loss(logits, heatmaps):
    # Cross-Entropy
    log_outputs = torch.log_softmax(logits, dim=1)
    dl = dice_loss(logits, heatmaps)
    # Weighted KL: scale each voxel's KL by class_weights
    kl_loss = torch.nn.functional.kl_div(
        log_outputs,  # (B, 5, D, H, W)
        heatmaps,  # (B, 5, D, H, W)
        reduction='mean'
    )
    return 0.5 * kl_loss + 0.5 * dl + 0.1 * total_variation_loss_3d(torch.softmax(logits, dim=1))

train_loss_history = []
max_train_metric = 0
# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()
    train_loss = 0
    train_metric = 0
    for images, heatmaps in train_dataloader:
        images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
        # Forward pass
        outputs = model(images)
        loss = combined_loss(outputs, heatmaps)
        # print(loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_probs = torch.softmax(outputs, dim=1)
        dist_mm = peak_distance_mm(pred_probs, heatmaps, spacing=(1.0, 1.0, 1.0))
        # dist_mm: (B,C) – you can take mean over batch & classes
        mean_dist = dist_mm.mean().item()
        # iou = jaccard_index(pred_labels, heatmaps.squeeze(1), task="multiclass", num_classes=Config.NUM_CLASSES)
        train_metric += mean_dist
    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, ED mm: {train_metric:.4f}, {epoch_time:.2f} seconds, Epoch {epoch + 1} , Current LR: {current_lr}")
    scheduler.step()
    if train_loss < 0.2:
        if train_metric > max_train_metric:
            max_train_metric = train_metric
            torch.save(model.state_dict(), "AVnet_model_cropped_best.pth")
            print(f"Model state dictionary saved to AVnet_model_cropped_best.pth at Epoch: {epoch + 1} with ED: {train_metric:.4f}")
model_save_path = "AV_net_model_cropped.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dictionary saved to {model_save_path}")
loss_history = {
    "train_loss": train_loss_history
}
with open("loss_history.json", "w") as f:
    json.dump(loss_history, f)
