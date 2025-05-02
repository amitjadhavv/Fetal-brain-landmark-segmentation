import torch
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label

INPUT_IMAGE_DIR = "MRI_data/cropped_images"  # Update this path
INPUT_MASK_DIR = "MRI_data/cropped_labels"    # Update this path
OUTPUT_DIR = "MRI_data/cropped_heatmaps"
LANDMARK_TYPES = [1, 2, 3, 4]
ALPHA = 1.0
SIGMA_SCALING_FACTOR = 0.60


def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def save_nifti(data, filename, affine):
    nib.save(nib.Nifti1Image(data, affine), filename)

def get_landmarks_by_type(mask, landmark_types):
    landmarks_dict = {}
    for landmark_type in landmark_types:
        coords = np.argwhere(mask == landmark_type)
        if len(coords) > 0:
            if landmark_type in [1, 3]:
                labeled_array, num_clusters = label(mask == landmark_type)
                clusters = [np.argwhere(labeled_array == i + 1) for i in range(num_clusters)]
                landmarks_dict[landmark_type] = clusters
            else:
                landmarks_dict[landmark_type] = [coords]
    return landmarks_dict

def compute_adaptive_sigma(landmarks, alpha=1.0, half_sigma=False, scaling_factor=0.75):
    if len(landmarks) == 1:
        sigma = 3.0
    else:
        landmarks = np.array(landmarks)
        centroid = np.mean(landmarks, axis=0)
        variance = np.mean(np.sum((landmarks - centroid) ** 2, axis=1))
        sigma = alpha * np.sqrt(variance)
    sigma *= scaling_factor
    return sigma / 2 if half_sigma else sigma

def create_gaussian_heatmap(image_shape, landmarks, sigma):
    mask = torch.zeros(image_shape, dtype=torch.float32)
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(image_shape[0]),
        torch.arange(image_shape[1]),
        torch.arange(image_shape[2]),
        indexing='ij')
    for landmark in landmarks:
        x, y, z = landmark
        distance = (grid_x - x) ** 2 + (grid_y - y) ** 2 + (grid_z - z) ** 2
        mask += torch.exp(-distance / (2 * sigma ** 2))
    return mask / mask.max()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = sorted(os.listdir(INPUT_IMAGE_DIR))
    mask_files = sorted(os.listdir(INPUT_MASK_DIR))

    for img_file, msk_file in zip(image_files, mask_files):
        img_path = os.path.join(INPUT_IMAGE_DIR, img_file)
        msk_path = os.path.join(INPUT_MASK_DIR, msk_file)
        print(f"Processing: {img_file} and {msk_file}")

        image_data, affine = load_nifti(img_path)
        mask_data, _ = load_nifti(msk_path)

        heatmap_4d = np.zeros((len(LANDMARK_TYPES), *image_data.shape), dtype=np.float32)
        landmarks_dict = get_landmarks_by_type(mask_data, LANDMARK_TYPES)

        for i, landmark_type in enumerate(LANDMARK_TYPES):
            heatmap_mask = torch.zeros(image_data.shape, dtype=torch.float32)
            if landmark_type in landmarks_dict:
                for cluster in landmarks_dict[landmark_type]:
                    half_sigma = landmark_type in [1, 3]
                    adaptive_sigma = compute_adaptive_sigma(cluster, ALPHA, half_sigma, SIGMA_SCALING_FACTOR)
                    cluster_heatmap = create_gaussian_heatmap(image_data.shape, cluster, adaptive_sigma)
                    heatmap_mask += cluster_heatmap
            heatmap_4d[i] = heatmap_mask.numpy()

        output_filename = os.path.join(OUTPUT_DIR, f"heatmap_{img_file}")
        save_nifti(heatmap_4d, output_filename, affine)
        print(f"Saved: {output_filename}")

if __name__ == "__main__":
    main()
