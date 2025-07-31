import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from configs.config import Config
from models.AttentionVNet import AttentionVNet
from utils.metrics import ed_mm_mixed_batch

image_paths = Config.get_image_paths()
mask_paths = Config.get_heatmap_paths()
# Load the test dataset
test_dataset = MRIDataset(image_paths, mask_paths, split="test")
test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

dice_metric = DiceMetric(include_background=True, get_not_nans=False)

model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
model_path = "Vnet_model_cropped_best1.pth"

state_dict = torch.load(model_path)
# Remove 'module.' prefix if present
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
test_metric = 0.0

with torch.no_grad():
    for images, heatmaps, spacings in test_dataloader:
        images = images.to(Config.DEVICE)
        heatmaps = heatmaps.to(Config.DEVICE)
        if isinstance(spacings, torch.Tensor):  # spac.shape == (B,3)
            spacing_list = [tuple(s.cpu().tolist()) for s in spacings]
        logits = model(images)
        probs = torch.sigmoid(logits)
        dist_mm = ed_mm_mixed_batch(probs, heatmaps, spacings)
        test_metric += dist_mm
    # Aggregate Dice scores
    test_metric /= len(test_dataloader)
    print(f"Mean Dice Coefficient: {test_metric:.4f}")
