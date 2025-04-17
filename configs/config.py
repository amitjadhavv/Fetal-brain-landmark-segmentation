import os
import torch
class Config:
    #path to the data directory
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MRI_data'))

    @classmethod
    def get_image_paths(cls):
        images = sorted(os.listdir(os.path.join(cls.dir_path, "cropped_images")))
        return [os.path.join(cls.dir_path, "cropped_images", i) for i in images]

    @classmethod
    def get_mask_paths(cls):
        masks = sorted(os.listdir(os.path.join(cls.dir_path, "cropped_labels")))
        return [os.path.join(cls.dir_path, "cropped_labels", i) for i in masks]

# Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5
    NUM_EPOCHS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

