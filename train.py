import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
import numpy as np
from utils.support import EarlyStopping
from utils.loss import multi_class_dice_loss
from utils.metrics import multiclass_dice_coefficient
from configs.config import Config
from tqdm import tqdm

# Define augmentations using torchio
transform = tio.Compose([
    #tio.RandomFlip(axes=(0, 1, 2)),          # Randomly flip along axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=30),  # Apply random scaling and rotation
    tio.RandomNoise(mean=0.0, std=0.1)     # Add random noise
])

# Load dataset
image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_dataset = MRIDataset(image_paths,mask_paths, split="val")
val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

#class_weight calculation
# class_weights=np.zeros(Config.NUM_CLASSES+1)
# for i in dataloader:
#     masks = torch.flatten(i[1]).to(torch.long)
#     print(torch.unique(masks))
#     a=list(torch.bincount(masks).numpy())
#     for j in range(Config.NUM_CLASSES):
#         class_weights[j]+=a[j]/sum(a)
# class_weights = class_weights/len(dataloader)
# print(float(class_weights[0]), float(class_weights[1]))

# Initialize model
model = VNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
early_stopping = EarlyStopping(patience=5, delta=1e-4)
print(len(train_dataloader))
print(len(val_dataloader))
print(Config.DEVICE)

# # Training loop
for epoch in tqdm(range(Config.NUM_EPOCHS)):
    model.train()
    train_loss = 0
    train_metric = 0
    for images, masks in train_dataloader:
        images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
        # Forward pass
        outputs = model(images)
        loss = multi_class_dice_loss(outputs, masks, Config.NUM_CLASSES)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_metric += multiclass_dice_coefficient(outputs, masks, num_classes=Config.NUM_CLASSES)
    train_loss /= len(train_dataloader)
    train_metric /= len(train_dataloader)

    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}")
    model.eval()
    val_loss = 0.0
    val_metric = 0.0

    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            outputs = model(images)
            loss = multi_class_dice_loss(outputs, masks, Config.NUM_CLASSES)

            val_loss += loss.item()
            val_metric += multiclass_dice_coefficient(outputs, masks, num_classes=Config.NUM_CLASSES)
    val_loss /= len(val_dataloader)
    val_metric /= len(val_dataloader)

    print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Val Loss: {val_loss:.4f}, Val Dice: {val_metric:.4f}")
    early_stopping(val_metric)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        model_save_path = "V_net_model_cropped.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model state dictionary saved to {model_save_path}")
        break