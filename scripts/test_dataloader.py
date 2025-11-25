import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# === Paths ===
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "wlasl100"

if str(FILE_DIR) not in sys.path:
    sys.path.append(str(FILE_DIR))

from WLASLDataset import WLASLVideoDataset, NUM_FRAMES  # noqa: E402


def main():

    print("[INFO] Probando DataLoader con varios batches (batch=8)")

    dataset = WLASLVideoDataset(
        split="train",
        base_path=str(DATA_ROOT),
        videos_folder="dataset",
    )

    print(f"[INFO] Muestras en split train: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Windows-friendly
        pin_memory=torch.cuda.is_available(),
    )

    # Probar 5 batches seguidos
    for i, (videos, labels) in enumerate(dataloader):
        print(f"\n[INFO] Batch {i+1}")
        print(f" - Videos shape: {videos.shape}")
        print(f" - Labels shape: {labels.shape}")
        print(f" - dtype: {videos.dtype}")
        print(f" - Frame count: {NUM_FRAMES}")

        if i == 4:
            break  # solo 5 batches

    print("\n[OK] DataLoader funcionando sin errores.")


if __name__ == "__main__":
    main()


