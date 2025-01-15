from distutils.command.config import config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
import numpy as np
from utils.loss import binary_weighted_dice_loss
from utils.metrics import dice_coefficient
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
dataset = MRIDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

#class_weight calculation
class_weights=np.zeros(Config.NUM_CLASSES+1)
for i in dataloader:
    masks = torch.flatten(i[1]).to(torch.long)
    a=list(torch.bincount(masks).numpy())
    for j in range(Config.NUM_CLASSES+1):
        class_weights[j]+=a[j]/sum(a)
class_weights = class_weights/len(dataloader)
# Initialize model
model = VNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
print(len(dataloader))
print(Config.DEVICE)
print(float(class_weights[0]), float(class_weights[1]))
# Training loop
for epoch in tqdm(range(Config.NUM_EPOCHS)):
    model.train()
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

        # Forward pass
        outputs = model(images)
        loss = binary_weighted_dice_loss(outputs, masks, class_weights[1])
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")
