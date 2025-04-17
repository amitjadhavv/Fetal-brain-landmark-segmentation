import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.AttentionVNet import AttentionVNet
import torchio as tio
from configs.config import Config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import  json
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from torchmetrics.functional import jaccard_index
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
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform, augmentation_factor=4)
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
dice_loss = DiceLoss(include_background=True,to_onehot_y=True, softmax=True, reduction="mean")
ce_loss = nn.CrossEntropyLoss(weight=norm_class_weights)
# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
# Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

def combined_loss(logits, masks):
    # Cross-Entropy
    ce = ce_loss(logits, masks.squeeze(1).long())
    # Dice (monai automatically does one-hot + softmax)
    d = dice_loss(logits, masks)
    return 0.5 * ce + 0.5 * d

train_loss_history = []
max_train_metric = 0
# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()
    train_loss = 0
    train_metric = 0
    for images, masks in train_dataloader:
        images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
        # Forward pass
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        iou = jaccard_index(pred_labels, masks.squeeze(1), task="multiclass", num_classes=Config.NUM_CLASSES)
        train_metric += iou.item()
    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, IoU Score: {train_metric:.4f}, {epoch_time:.2f} seconds, Epoch {epoch + 1} , Current LR: {current_lr}")
    if epoch > 200:
        scheduler.step()
    if train_metric > 0.80:
        if train_metric > max_train_metric:
            max_train_metric = train_metric
            torch.save(model.state_dict(), "AV_net_model_cropped_best.pth")
            print(f"Model state dictionary saved to V_net_model_roi_best.pth at Epoch: {epoch + 1} with IoU Score: {train_metric:.4f}")
model_save_path = "AV_net_model_cropped.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dictionary saved to {model_save_path}")
loss_history = {
    "train_loss": train_loss_history
}

with open("loss_history.json", "w") as f:
    json.dump(loss_history, f)
