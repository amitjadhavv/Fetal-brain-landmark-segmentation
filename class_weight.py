"""
Compute class weights for 3D MRI segmentation masks in the dataset.

This script assumes that each mask is a **3‑D NIfTI** file whose voxel values are
integer class labels in the range **0 … NUM_CLASSES‑1**.

It counts the voxels per class across *all* masks and derives weights using the
**median‑frequency balancing** scheme (Enet paper), which usually works well
for highly imbalanced medical datasets.
"""
import numpy as np
import nibabel as nib
from tqdm import tqdm
from configs.config import Config

# --------------------------------------------------
# Settings
# --------------------------------------------------
NUM_CLASSES = 5  # ← total number of segmentation classes (incl. background)
MASK_PATHS = Config.get_mask_paths()

# --------------------------------------------------
# Pass 1: voxel counting
# --------------------------------------------------
voxel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
print(f"Scanning {len(MASK_PATHS)} 3‑D mask volumes …")
for mp in tqdm(MASK_PATHS):
    mask = nib.load(mp).get_fdata()
    if mask.ndim != 3:
        raise ValueError(f"Expected 3‑D mask volume but got shape {mask.shape} for {mp}")
    mask = mask.astype(np.int64)
    if mask.max() >= NUM_CLASSES:
        raise ValueError(
            f"Mask {mp} contains label {mask.max()} ≥ NUM_CLASSES ({NUM_CLASSES})."
            " Check NUM_CLASSES or the data.")
    voxel_counts += np.bincount(mask.ravel(), minlength=NUM_CLASSES)

# --------------------------------------------------
# Derive class weights
# --------------------------------------------------
print("\nVoxel counts per class:", voxel_counts)
total_voxels = voxel_counts.sum()
frequencies = voxel_counts / total_voxels
median_freq = np.median(frequencies[frequencies > 0])
class_weights = median_freq / frequencies
norm_class_weights = class_weights / class_weights.sum()
print("Class frequencies:", frequencies)
print("Median frequency:", median_freq)
print("\n→ Class weights (median‑frequency balanced):", norm_class_weights)

# (Optional) persist the weights
np.save("class_weights.npy", norm_class_weights)
print("Weights saved to class_weights.npy")