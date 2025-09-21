import time
import torch
from torch.utils.data import DataLoader
from configs.config import Config
from models.AttentionVNet import AttentionVNet
from Datasets.dataset import MRIDataset

def measure_inference_time_stage2(model_path="AVnet_model_cropped_best.pth"):
    # Force CPU
    device = torch.device("cpu")

    # Load dataset
    image_paths = Config.get_image_paths()
    mask_paths = Config.get_heatmap_paths()
    dataset = MRIDataset(image_paths, mask_paths, split="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    times = []
    with torch.no_grad():
        for images,mask,_ in dataloader:
            images = images.to(device)

            start = time.time()
            _ = model(images)
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Stage 2 (Landmark Detection) average inference time: {avg_time:.4f} seconds per volume on CPU")

if __name__ == "__main__":
    measure_inference_time_stage2()
