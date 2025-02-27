import torch
import random
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio  # For optional data augmentation
from scipy.ndimage import label


class MRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, split="train", train_ratio=0.9, val_ratio=0.0, test_ratio=0.10,
                 seed=123, transform=None, augmentation_factor=1, landmark_types=[1, 3, 4, 6], alpha=1.0, sigma_scaling=0.65):
        """
        Args:
            image_paths (list): List of paths to MRI images.
            mask_paths (list): List of paths to segmentation masks.
            transform (callable, optional): Optional transform to apply on images and masks.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.split = split
        self.transform = transform
        self.augmentation_factor = augmentation_factor
        self.landmark_types = landmark_types
        self.alpha = alpha
        self.sigma_scaling = sigma_scaling

        # Shuffle data with seed
        data = list(zip(image_paths, mask_paths))
        random.seed(seed)
        random.shuffle(data)
        self.image_paths, self.mask_paths = zip(*data)

        # Compute split indices
        total_files = len(self.image_paths)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)

        self.train_indices = range(0, train_count)
        self.val_indices = range(train_count, train_count + val_count)
        self.test_indices = range(train_count + val_count, total_files)

        # Select the appropriate split
        if split == "train":
            self.indices = self.train_indices
        elif split == "val":
            self.indices = self.val_indices
        elif split == "test":
            self.indices = self.test_indices
        else:
            raise ValueError("Invalid split! Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.indices) * self.augmentation_factor

    def load_nifti(self, file_path):
        """Loads a NIfTI image and returns its data as a NumPy array."""
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata()

    def get_landmarks_by_type(self, mask):
        """Extracts landmarks and separates multiple clusters for IDs 1 and 4."""
        landmarks_dict = {}
        for landmark_type in self.landmark_types:
            coords = np.argwhere(mask == landmark_type)
            if len(coords) > 0:
                if landmark_type in [1, 4]:  # Handle separate clusters for IDs 1 & 4
                    labeled_array, num_clusters = label(mask == landmark_type)
                    clusters = [np.argwhere(labeled_array == i + 1) for i in range(num_clusters)]
                    landmarks_dict[landmark_type] = clusters
                else:
                    landmarks_dict[landmark_type] = [coords]
        return landmarks_dict

    def compute_adaptive_sigma(self, landmarks, half_sigma=False):
        """Computes an adaptive sigma value based on the spatial spread of landmarks."""
        if len(landmarks) == 1:
            sigma = 3.0
        else:
            landmarks = np.array(landmarks)
            centroid = np.mean(landmarks, axis=0)
            variance = np.mean(np.sum((landmarks - centroid) ** 2, axis=1))
            sigma = self.alpha * np.sqrt(variance)
        sigma *= self.sigma_scaling
        return sigma / 2 if half_sigma else sigma

    def create_gaussian_heatmap(self, image_shape, landmarks, sigma):
        """Generates a 3D Gaussian heatmap mask for given landmarks."""
        mask = torch.zeros(image_shape, dtype=torch.float32)
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(image_shape[0]),
            torch.arange(image_shape[1]),
            torch.arange(image_shape[2]),
            indexing='ij'
        )
        for landmark in landmarks:
            x, y, z = landmark
            distance = (grid_x - x) ** 2 + (grid_y - y) ** 2 + (grid_z - z) ** 2
            mask += torch.exp(-distance / (2 * sigma ** 2))
        return mask / mask.max()

    def __getitem__(self, idx):
        actual_idx = self.indices[idx // self.augmentation_factor]
        img = self.load_nifti(self.image_paths[actual_idx])
        mask = self.load_nifti(self.mask_paths[actual_idx])
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.expand_dims(img, axis=0)
        img = F.interpolate(torch.tensor(img).unsqueeze(0), size=(64, 64, 64), mode='trilinear', align_corners=False).squeeze(0)
        landmarks_dict = self.get_landmarks_by_type(mask)
        heatmap_4d = torch.zeros((len(self.landmark_types), 64, 64, 64), dtype=torch.float32)
        for i, landmark_type in enumerate(self.landmark_types):
            heatmap_mask = torch.zeros((64, 64, 64), dtype=torch.float32)
            if landmark_type in landmarks_dict:
                for cluster in landmarks_dict[landmark_type]:
                    half_sigma = landmark_type in [1, 4]
                    adaptive_sigma = self.compute_adaptive_sigma(cluster, half_sigma, )
                    cluster_heatmap = self.create_gaussian_heatmap((64, 64, 64), cluster, adaptive_sigma)
                    heatmap_mask += cluster_heatmap
            heatmap_4d[i] = heatmap_mask
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img),
                heatmap=tio.LabelMap(tensor=heatmap_4d)
            )
            subject = self.transform(subject)
            img = subject['image'].data
            heatmap_4d = subject['heatmap'].data # Ensure channels remain consistent
        return img, heatmap_4d
