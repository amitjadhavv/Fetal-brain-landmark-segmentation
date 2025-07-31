import torch
import numpy as np
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from configs.config import Config
from models.AttentionVNet import AttentionVNet
from utils.metrics import ed_mm_per_landmark_batch
import warnings
warnings.filterwarnings("ignore")
from utils.metrics import ed_mm_per_landmark_batch
import numpy as np

def evaluate_split(split="test", model_path="Vnet_model_cropped_best1.pth"):
    """
    Evaluate model on a dataset split ('test' or 'val') and return mean ± std ED per landmark.
    """
    image_paths = Config.get_image_paths()
    mask_paths = Config.get_heatmap_paths()
    dataset = MRIDataset(image_paths, mask_paths, split=split)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    all_dists = []  # will be list of lists [eyes, nose, temporal, cerebellum]

    with torch.no_grad():
        for images, heatmaps, spacings in dataloader:
            images = images.to(Config.DEVICE)
            heatmaps = heatmaps.to(Config.DEVICE)

            if isinstance(spacings, torch.Tensor):
                spacings = [tuple(s.cpu().tolist()) for s in spacings]

            logits = model(images)
            probs = torch.sigmoid(logits)

            per_sample = ed_mm_per_landmark_batch(probs, heatmaps, spacings)
            all_dists.extend(per_sample)

    all_dists = np.array(all_dists)  # shape (N, 4)
    landmark_names = ["Eyes", "Nose", "Temporal Lobes", "Cerebellum"]

    for i, name in enumerate(landmark_names):
        mean_val = np.mean(all_dists[:, i])
        std_val = np.std(all_dists[:, i], ddof=1) if len(all_dists) > 1 else 0.0
        print(f"{split.capitalize()} {name}: {mean_val:.2f} ± {std_val:.2f} mm")

    return all_dists


if __name__ == "__main__":
    # Set batch_size = 1 in config for per-sample stats
    print("Evaluating Validation Set:")
    evaluate_split(split="val", model_path="Vnet_model_cropped_best1.pth")

    print("\nEvaluating Test Set:")
    evaluate_split(split="test", model_path="Vnet_model_cropped_best1.pth")
