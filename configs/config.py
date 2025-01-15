import os
import torch
class Config:
    #path to the data directory
    dir_path = "/home/amit/Downloads/OneDrive_1_10-15-2024/Dataset080_Landmarks/"

    @classmethod
    def get_image_paths(cls):
        images = os.listdir(os.path.join(cls.dir_path, "imagesTr"))
        return [os.path.join(cls.dir_path, "imagesTr", i) for i in images]

    @classmethod
    def get_mask_paths(cls):
        masks = os.listdir(os.path.join(cls.dir_path, "labelsTr"))
        return [os.path.join(cls.dir_path, "labelsTr", i) for i in masks]

# Hyperparameters
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    NUM_EPOCHS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"