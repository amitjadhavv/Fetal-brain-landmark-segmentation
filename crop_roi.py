import numpy as np
import nibabel as nib

def remap_other_mask_classes(mask_data):
    """
    Apply class mapping to the mask:
    0 -> 0, 1 -> 1, 3 -> 2, 4 -> 3, 6 -> 4.
    """
    class_mapping = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
    remapped_mask = np.zeros_like(mask_data, dtype=np.uint8)
    for original_class, new_class in class_mapping.items():
        remapped_mask[mask_data == original_class] = new_class
    return remapped_mask

def crop_roi_to_cube(image_path, mask_path, output_image_path,
                     other_mask_path=None, output_other_mask_path=None, buffer=2):
    # Load image and both masks
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    image_data = image.get_fdata()
    mask_data = mask.get_fdata()

    # Load the separate mask used to determine ROI for cropping
    if other_mask_path:
        roi_mask = nib.load(other_mask_path)
        roi_data = roi_mask.get_fdata()
    else:
        roi_data = mask_data  # fallback to main mask if not provided

    # Compute center of mass of the ellipse (ROI)
    roi_indices = np.argwhere(roi_data > 0)
    center = np.round(roi_indices.mean(axis=0)).astype(int)

    # Calculate bounding box size of the ROI and derive cube size from the largest dimension
    min_coords = roi_indices.min(axis=0)
    max_coords = roi_indices.max(axis=0)
    bbox_size = max_coords - min_coords
    max_dim = int(np.max(bbox_size)) + 2 * buffer
    half_size = max_dim // 2

    start = center - half_size
    end = center + half_size

    # Adjust if we go beyond image boundaries
    for i in range(3):
        if start[i] < 0:
            start[i] = 0
            end[i] = max_dim
        if end[i] > image_data.shape[i]:
            end[i] = image_data.shape[i]
            start[i] = end[i] - max_dim
        start[i] = max(start[i], 0)
        end[i] = min(end[i], image_data.shape[i])

    # Crop image and original mask (unaltered image, discretized mask)
    cropped_image = image_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # Save cropped image and mask
    cropped_image_nifti = nib.Nifti1Image(cropped_image, affine=image.affine)
    nib.save(cropped_image_nifti, output_image_path)

    # Crop and save the ROI mask for locating (also discretized)
    if other_mask_path and output_other_mask_path:
        original_roi_data = nib.load(other_mask_path).get_fdata()
        cropped_other_mask = original_roi_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        cropped_other_mask = np.round(cropped_other_mask).astype(np.uint8)
        cropped_other_mask = remap_other_mask_classes(cropped_other_mask)
        cropped_other_mask_nifti = nib.Nifti1Image(cropped_other_mask, affine=image.affine)
        nib.save(cropped_other_mask_nifti, output_other_mask_path)


import os


def process_folder(input_image_folder, input_mask_folder, input_label_folder, output_folder, buffer=2):
    output_image_folder = os.path.join(output_folder, "cropped_images")
    output_label_folder = os.path.join(output_folder, "cropped_labels")
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    image_files = sorted(os.listdir(input_image_folder))
    mask_files = sorted(os.listdir(input_mask_folder))
    label_files = sorted(os.listdir(input_label_folder))

    for img_file, mask_file, label_file in zip(image_files, mask_files, label_files):
        input_image_path = os.path.join(input_image_folder, img_file)
        input_mask_path = os.path.join(input_mask_folder, mask_file)
        input_label_path = os.path.join(input_label_folder, label_file)

        output_image_path = os.path.join(output_image_folder, f"cropped_{img_file}")
        output_label_path = os.path.join(output_label_folder, f"cropped_{label_file}")

        crop_roi_to_cube(
            input_image_path,
            input_mask_path,
            output_image_path,
            input_label_path,
            output_label_path,
            buffer=buffer
        )


process_folder(
    "/home/amit/PycharmProjects/fetalMRI/MRI_data/new_images",
    "/home/amit/PycharmProjects/fetalMRI/MRI_data/new_global_masks",
    "/home/amit/PycharmProjects/fetalMRI/MRI_data/new_labels",
    "/home/amit/PycharmProjects/fetalMRI/MRI_data/",
    buffer=4
)
