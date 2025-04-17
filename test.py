import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from configs.config import Config
from models.AttentionVNet import AttentionVNet

image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
# Load the test dataset
test_dataset = MRIDataset(image_paths, mask_paths, split="test")
test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

dice_metric = DiceMetric(include_background=True, get_not_nans=False)

model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
model_path = "AV_net_model_cropped_best.pth"

state_dict = torch.load(model_path)
# Remove 'module.' prefix if present
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
val_loss = 0.0
val_metric = 0.0

with torch.no_grad():
    for images, masks in test_dataloader:
        images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

        # Forward pass
        probs = torch.softmax(model(images), dim=1)  # (B, 5, D, H, W)
        pred_labels = torch.argmax(probs, dim=1)  # (B, D, H, W)

        # ── convert to one‑hot so DiceMetric sees 5 channels ─────────────────
        pred_1hot = torch.nn.functional.one_hot(pred_labels, num_classes=5)  # (B, D, H, W, 5)
        pred_1hot = pred_1hot.permute(0, 4, 1, 2, 3).float()  # (B, 5, D, H, W)

        masks_1hot = torch.nn.functional.one_hot(masks, num_classes=5)
        masks_1hot = masks_1hot.permute(0, 4, 1, 2, 3).float()

        dice = dice_metric(y_pred=pred_labels, y=masks_1hot)
        dice = dice.mean()
        val_metric += dice.item()
    # Aggregate Dice scores
    val_metric /= len(test_dataloader)
    print(f"Mean Dice Coefficient: {val_metric:.4f}")
