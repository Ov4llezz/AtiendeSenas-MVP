import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "wlasl100"

if str(FILE_DIR) not in sys.path:
    sys.path.append(str(FILE_DIR))

from WLASLDataset import WLASLVideoDataset, NUM_FRAMES  # noqa: E402


def main():
    print("[INFO] Probando WLASLVideoDataset + DataLoader (batch=8)")

    try:
        dataset = WLASLVideoDataset(
            split="train",
            base_path=str(DATA_ROOT),
            videos_folder="dataset",
        )
    except Exception as e:
        print("[ERROR] No se pudo inicializar el dataset:")
        print(repr(e))
        return

    print(f"[INFO] Muestras en split 'train': {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    try:
        videos, labels = next(iter(dataloader))
    except Exception as e:
        print("[ERROR] No se pudo obtener un batch del DataLoader:")
        print(repr(e))
        return

    print("[INFO] Batch cargado correctamente")
    print(f" - Videos shape: {videos.shape}")
    print(f" - Labels shape: {labels.shape}")
    print(f" - NUM_FRAMES: {NUM_FRAMES}")
    print(f" - dtype videos: {videos.dtype}, dtype labels: {labels.dtype}")
    print(f" - min: {videos.min().item():.4f}, max: {videos.max().item():.4f}")


if __name__ == "__main__":
    main()






