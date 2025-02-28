import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
import numpy as np
from configs.config import Config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import  json
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
# Define augmentations using torchio
transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),          # Randomly flip along axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=30),  # Apply random scaling and rotation
    tio.RandomNoise(mean=0.0, std=0.1)
    # Add random noise
])
# Load dataset
image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform, augmentation_factor=1)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print(len(train_dataloader))
# class_weight calculation
class_weights = [3.8217e-05, 2.6055e-01, 5.4443e-02, 1.4835e-01, 5.3661e-01]
# Nrmalize so all weights sum to 1
norm_class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("Normalized Class Weights:", norm_class_weights)

# Initialize model
model = VNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
dice_loss_fn = DiceLoss(include_background=True, softmax=False, squared_pred=True,weight=norm_class_weights, reduction="mean")
dice_metric = DiceMetric(include_background=False)
kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
# Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

train_loss_history = []
# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()  # Start time tracking
    train_loss = 0
    train_metric = 0
    for images, heatmaps in train_dataloader:
        images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
        outputs = model(images)
        # Ensure heatmap and outputs have the same shape
        background = 1 - torch.sum(heatmaps, dim=1, keepdim=True)  # Compute background class
        heatmaps = torch.cat([background, heatmaps], dim=1)

        print(heatmaps.shape, heatmaps.dtype, outputs.shape, outputs.dtype)
        # Forward pass
        dice_loss = dice_loss_fn(outputs, heatmaps)
        kl_loss = kl_loss_fn(torch.log(outputs + 1e-6), heatmaps)
        loss = dice_loss + kl_loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss, type(loss))
        train_loss += loss.item()
        dice = dice_metric(y_pred=outputs, y=heatmaps)
        print(dice)
        if dice.ndim > 0:
            dice = dice.mean()
        train_metric += dice.item() * images.size(0)

    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}, Time: {epoch_time:.2f} seconds")
    scheduler.step()
model_save_path = "V_net_model_cropped.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dictionary saved to {model_save_path}")
loss_history = {
    "train_loss": train_loss_history
}

with open("loss_history.json", "w") as f:
    json.dump(loss_history, f)
