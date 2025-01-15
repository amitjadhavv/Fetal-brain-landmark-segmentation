import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio  # For optional data augmentation

from torch.utils.data import DataLoader

class MRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            image_paths (list): List of paths to MRI images.
            mask_paths (list): List of paths to segmentation masks.
            transform (callable, optional): Optional transform to apply on images and masks.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the MRI image and mask
        img = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Normalize the image
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Add channel dimension to both image and mask
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensors
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)  # Use long for segmentation labels

        # Resize to 64x64x64
        target_size = (64, 64, 64)
        img = F.interpolate(img.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=target_size, mode='nearest').squeeze(0)

        # Ensure correct shape for torchio
        img = img.squeeze(0)  # Remove unnecessary batch dimension
        mask = mask.squeeze(0)

        # Apply transformations (if any)
        if self.transform:
            img = self.transform(img.unsqueeze(0))  # Add channel dimension back
            mask = self.transform(mask.unsqueeze(0))  # Add channel dimension back

        return img, mask