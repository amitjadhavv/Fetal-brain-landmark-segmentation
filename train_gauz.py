import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
from configs.config import Config
from monai.losses import DiceLoss
from utils.metrics import ed_mm_mixed_batch
import  json
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from utils.loss import total_variation_loss_3d
# Define augmentations using torchio
import warnings
warnings.filterwarnings("ignore")

transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),
    tio.RandomAffine(scales=(0.85, 1.15), degrees=45),
    tio.RandomElasticDeformation(),
    tio.RandomGamma(log_gamma=(0.7, 1.3)),
    tio.RandomNoise(mean=0.0, std=0.15)
])
# Load dataset
image_paths = Config.get_image_paths()
mask_paths = Config.get_heatmap_paths()
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform, augmentation_factor=8)
val_dataset = MRIDataset(image_paths, mask_paths, split="val", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_dataset,batch_size=Config.BATCH_SIZE,num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print(len(train_dataloader))
# class_weight calculation
norm_class_weights = torch.tensor([0.23416625, 0.04894709, 0.14261888, 0.5732486],dtype=torch.float32,device=Config.DEVICE)
norm_class_weights = norm_class_weights.view(1, 4, 1, 1, 1)  #0.00101918,
# Nrmalize so all weights sum to 1
norm_class_weights = torch.tensor(norm_class_weights, dtype=torch.float32).to(Config.DEVICE)

# Initialize model
model = VNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
dice_loss = DiceLoss(include_background=True, softmax=True, reduction="mean", weight=norm_class_weights)
bce_loss = nn.BCEWithLogitsLoss(
    reduction='mean',          # default; or 'sum', or 'none'
    pos_weight= norm_class_weights           # optional tensor to rebalance 0/1
)
# Learning Rate Scheduler (Cosine Annealing for smooth decay)
# scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

def combined_loss(logits, heatmaps):
    # KL loss
    bce = bce_loss(logits, heatmaps)
    B, C, D, H, W = logits.shape
    N = D * H * W
    log_outputs = torch.log_softmax(logits.view(B * C, N), dim=1)
    q = heatmaps.view(B * C, N)
    q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
    kl = torch.nn.functional.kl_div(
        log_outputs,  # (B, 5, D, H, W)
        q,  # (B, 5, D, H, W)
        reduction='batchmean'
    )
    tv = total_variation_loss_3d(torch.softmax(logits, dim=1))
    return 0.7 * bce + 0.2 * kl + 0.1 * tv

train_loss_history = []
val_history =[]
best_val_metric = 100
# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()
    train_loss = 0.0
    train_metric = 100.0
    for images, heatmaps, spacings in train_dataloader:
        images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
        # Forward pass
        outputs = model(images)
        loss = combined_loss(outputs, heatmaps)
        if isinstance(spacings, torch.Tensor):  # spac.shape == (B,3)
            spacings = [tuple(s.cpu().tolist()) for s in spacings]
            print(spacings)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            dist_mm = ed_mm_mixed_batch(pred_probs, heatmaps, spacings)
        # dist_mm: (B,C) – you can take mean over batch & classes
            mean_dist = dist_mm
        train_metric += mean_dist
    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)
    model.eval()
    val_loss = 0.0
    val_metric = 0.0
    with torch.no_grad():
        for images, heatmaps,spacings in val_loader:
            images = images.to(Config.DEVICE)
            heatmaps = heatmaps.to(Config.DEVICE)
            if isinstance(spacings, torch.Tensor):  # spac.shape == (B,3)
                spacing_list = [tuple(s.cpu().tolist()) for s in spacings]
            logits = model(images)
            loss = combined_loss(logits, heatmaps)
            val_loss += loss.item()
            probs = torch.sigmoid(logits)
            dist_mm = ed_mm_mixed_batch(probs, heatmaps, spacings)
            val_metric += dist_mm

    val_loss /= len(val_loader)
    val_metric /= len(val_loader)
    val_history.append({"loss": val_loss, "ed_mm": val_metric})
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, train ED mm: {train_metric:.4f}, val loss:{val_loss:4f}, val ED mm:{val_metric:4f} {epoch_time:.2f} seconds, Epoch {epoch + 1} , Current LR: {current_lr}")
    scheduler.step()
    if val_loss < 0.05:
        if train_metric < best_val_metric:
            best_train_metric = train_metric
            torch.save(model.state_dict(), "Vnet_model_cropped_best.pth")
            print(f"Model state dictionary saved to Vnet_model_cropped_best.pth at Epoch: {epoch + 1} with ED: {train_metric:.4f}")
model_save_path = "V_net_model_cropped.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dictionary saved to {model_save_path}")
# train_loss_history = {
#     "train_loss": train_loss_history
# }
# val_loss_history = {
#     "val_loss": val_history
# }
# with open("train_loss_history.json", "w") as f:
#     json.dump(train_loss_history, f)
# with open("val_loss_history.json", "w") as f:
#     json.dump(val_loss_history, f)