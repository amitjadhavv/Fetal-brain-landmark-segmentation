import argparse
from pathlib import Path
import numpy as np
import torch
import nibabel as nib


def build_5ch_heatmap(fg: np.ndarray, thresh: float = 0.2, eps: float = 1e-8) -> np.ndarray:
    """Convert un‑normalised 4‑channel map → 5‑channel prob‑map with thresholding.

    Workflow per voxel:
    1. Zero‑out foreground values < `thresh` ("crop")
    2. Clip negatives → 0
    3. Normalise the remaining 4 fg channels (if their sum > 0)
    4. bg = 1 − Σ fg
    5. Final renorm so Σ five‑channels = 1
    """
    assert fg.ndim == 4 and fg.shape[0] == 4, "expect (4,D,H,W)"

    # Step 1: crop weak responses
    fg = np.where(fg >= thresh, fg, 0.0)

    # Step 2: ensure non‑negative
    fg = np.clip(fg, 0, None)

    # Step 3: per‑voxel normalisation
    fg_sum = fg.sum(axis=0, keepdims=True)            # (1,D,H,W)
    fg_probs = np.where(fg_sum > 0, fg / (fg_sum + eps), 0.0)

    # Step 4: background channel
    bg = 1.0 - fg_probs.sum(axis=0, keepdims=True)
    bg = np.clip(bg, 0, None)

    # Step 5: concat + final renorm
    heat5 = np.concatenate([bg, fg_probs], axis=0)
    heat5 = heat5 + eps
    heat5 = heat5 / heat5.sum(axis=0, keepdims=True)
    return heat5.astype(np.float32)


def process_file(in_path: Path, out_path: Path):
    """Load 4‑channel NIfTI/NumPy/Torch heat‑map, add bg, normalise, save."""
    if in_path.suffix in {".nii", ".nii.gz"}:
        img = nib.load(str(in_path))
        fg  = img.get_fdata(dtype=np.float32)           # (D,H,W,4) or (4,D,H,W)
        if fg.shape[-1] == 4:                           # if channels last
            fg = np.moveaxis(fg, -1, 0)                 # → (4,D,H,W)
    elif in_path.suffix == ".npy":
        fg = np.load(in_path)
    elif in_path.suffix in {".pt", ".pth"}:
        fg = torch.load(in_path).cpu().numpy()
    else:
        print(f"[skip] {in_path.name}: unsupported extension")
        return

    if fg.shape[0] != 4:
        print(f"[skip] {in_path.name}: expected 4 channels, got {fg.shape[0]}")
        return

    heat5 = build_5ch_heatmap(fg)                       # (5,D,H,W)

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix in {".nii", ".nii.gz"}:
        # channels → last axis for NIfTI
        # heat5_last = np.moveaxis(heat5, 0, -1)          # (D,H,W,5)
        nib.save(nib.Nifti1Image(heat5, affine=img.affine), str(out_path))
    else:
        torch.save(torch.from_numpy(heat5), out_path)
    print(f"[ok]   {in_path.name} → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Add background channel and normalise 4‑ch heat‑maps.")
    parser.add_argument("input_dir", type=Path, help="Folder with 4‑channel .nii/.npy/.pt heat‑maps")
    parser.add_argument("output_dir", type=Path, help="Destination folder for 5‑channel files")
    parser.add_argument("--ext", default=".nii", choices=[".nii.gz", ".nii", ".npy", ".pt"], help="Output file extension")
    args = parser.parse_args()

    files = sorted([p for p in args.input_dir.iterdir() if p.suffix in {".nii", ".nii.gz", ".npy", ".pt", ".pth"}])
    if not files:
        print("No heat‑map files found in", args.input_dir)
        return

    for p in files:
        out_path = args.output_dir / p.with_suffix(args.ext).name
        process_file(p, out_path)


if __name__ == "__main__":
    main()
