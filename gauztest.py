import nibabel as nib
import os
import numpy as np

def save_nifti(data, filename, affine):
    nib.save(nib.Nifti1Image(data, affine), filename)

def split_and_save_4d_heatmap(input_4d_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = nib.load(input_4d_path)
    data_4d = img.get_fdata()
    affine = img.affine

    num_channels = data_4d.shape[0]  # Assuming shape is [C, D, H, W]

    for i in range(num_channels):
        heatmap_3d = data_4d[i]
        output_filename = os.path.join(output_dir, f"heatmap_channel_{i + 1}.nii.gz")
        save_nifti(heatmap_3d, output_filename, affine)
        print(f"Saved: {output_filename}")

# Example usage:
split_and_save_4d_heatmap("/home/amit/PycharmProjects/fetalMRI/MRI_data/cropped_heatmaps/heatmap_cropped_image_373.nii", "/home/amit/PycharmProjects/fetalMRI/MRI_data/output/")
