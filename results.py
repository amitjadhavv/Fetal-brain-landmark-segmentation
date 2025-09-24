import torch
import numpy as np
from torch.utils.data import DataLoader
from configs.config import Config
from models.AttentionVNet import AttentionVNet
from Datasets.dataset import MRIDataset
from utils.metrics import ed_mm_per_landmark_batch  # use the per-landmark ED function

def evaluate_split(split="test", model_path="AVnet_model_cropped_best.pth", key=None):
    """
    Evaluate Attention V-Net on a dataset split ('val' or 'test')
    and return mean ± std ED per landmark and overall.
    """
    # Load dataset
    image_paths = Config.get_image_paths()
    mask_paths = Config.get_heatmap_paths()
    dataset = MRIDataset(image_paths, mask_paths, split=split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 for per-sample ED

    # Load model
    model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): v for v, k in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    all_dists = []  # will store [eyes, nose, temporal, cerebellum] per sample

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

    results = {}
    for i, name in enumerate(landmark_names):
        mean_val = np.mean(all_dists[:, i])
        std_val = np.std(all_dists[:, i], ddof=1) if len(all_dists) > 1 else 0.0
        results[name] = (mean_val, std_val)
        print(f"{split.capitalize()} {name}: {mean_val:.2f} ± {std_val:.2f} mm")

    # Overall mean (average across all landmarks per subject)
    overall = np.mean(all_dists, axis=1)
    results["Overall"] = (np.mean(overall), np.std(overall, ddof=1))
    print(f"{split.capitalize()} Overall: {results['Overall'][0]:.2f} ± {results['Overall'][1]:.2f} mm")

    return results


if __name__ == "__main__":
    print("Evaluating Validation Set:")
    val_results = evaluate_split(split="val", model_path="AVnet_model_cropped_best.pth")

    print("\nEvaluating Test Set:")
    test_results = evaluate_split(split="test", model_path="AVnet_model_cropped_best.pth")
