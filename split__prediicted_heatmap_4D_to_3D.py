import nibabel as nib
import os
import numpy as np

# Path to your 4D heatmap .nii.gz file
input_path = "/home/amit/PycharmProjects/fetalMRI/MRI_data/output/predicted_cropped_image_355.nii"
output_dir = "/home/amit/PycharmProjects/fetalMRI/MRI_data/output/"
os.makedirs(output_dir, exist_ok=True)

# Load the heatmap
heatmap_img = nib.load(input_path)
heatmap_data = heatmap_img.get_fdata()  # Expected shape: (C, D, H, W)
affine = heatmap_img.affine

# Save each channel as a separate 3D NIfTI file
for i in range(heatmap_data.shape[0]):
    volume = heatmap_data[i]  # Shape: (D, H, W)
    volume_nifti = nib.Nifti1Image(volume, affine)
    nib.save(volume_nifti, os.path.join(output_dir, f"heatmap_output{i}.nii.gz"))
