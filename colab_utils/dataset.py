import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NUM_FRAMES = 16

# ============================================================
#   Cargar mapas de labels (video_id -> clase)
#   Usa nslt_100.json: { "05237": {"subset": "train", "action": [77, 1, 55]}, ... }
#   action[0] ya es un id de clase entre 0 y 99
# ============================================================
def load_label_maps(meta_json: str, subset_json: str):
    """
    meta_json: no se utiliza por ahora (se deja por compatibilidad).
    subset_json: ruta a nslt_100.json
    """
    with open(subset_json, "r", encoding="utf-8") as f:
        subset = json.load(f)

    vid2label = {}
    label_set = set()

    for vid, info in subset.items():
        label = info["action"][0]      # id de clase (0..99)
        vid2label[vid] = label
        label_set.add(label)

    labels_sorted = sorted(label_set)
    label2id = {lab: lab for lab in labels_sorted}
    id2label = {lab: lab for lab in labels_sorted}

    return vid2label, label2id, id2label


# ============================================================
#   Leer listas de splits
# ============================================================
def load_split_list(split_txt: str):
    with open(split_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ============================================================
#   Extraer frames uniformes del video
# ============================================================
def sample_frames_uniform(video_path: str, num_frames: int = NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video vacío o corrupto: {video_path}")

    indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No se pudieron leer frames de {video_path}")

    # Si faltan frames, duplicamos último
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames[:num_frames]


# ============================================================
#   Clase principal PyTorch Dataset
# ============================================================
class WLASLVideoDataset(Dataset):
    def __init__(
        self,
        split: str,
        base_path: str = "data/wlasl100",
        videos_folder: str = "dataset",  # carpeta con train/val/test
        meta_json: str = "WLASL_v0.3.json",
        subset_json: str = "nslt_100.json",
        dataset_size: int = 100,  # Número de clases: 100 o 300
    ):
        """
        split: 'train', 'val', 'test'
        base_path: ruta base del dataset (ej: data/wlasl100 o data/wlasl300)
        videos_folder: subcarpeta donde están train/val/test
        dataset_size: número de clases en el dataset (100 o 300)

        Nota: Si dataset_size=300, automáticamente se ajustan los nombres de archivos JSON
        """
        assert split in ["train", "val", "test"]
        self.split = split
        self.dataset_size = dataset_size

        # === Auto-detectar dataset_size si no se especificó ===
        if dataset_size == 100 and "wlasl300" in base_path.lower():
            self.dataset_size = 300
        elif dataset_size == 300 and "wlasl100" in base_path.lower():
            self.dataset_size = 100

        # === Ajustar nombres de archivos JSON automáticamente ===
        if self.dataset_size == 300:
            if meta_json == "WLASL_v0.3.json":
                meta_json = "WLASL_v0.3_300.json"
            if subset_json == "nslt_100.json":
                subset_json = "nslt_300.json"

        # === Rutas completas ===
        self.base = base_path
        self.splits_dir = os.path.join(base_path, "splits")

        # Determinar el directorio de videos correcto
        # Para V2: val split apunta a videos en test/, no en val/
        # Leer el split file para detectar esto
        split_txt_path_temp = os.path.join(base_path, "splits", f"{split}_split.txt")
        if os.path.exists(split_txt_path_temp):
            with open(split_txt_path_temp, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            # Detectar el directorio real (ej: "test\\00625.mp4" → directorio es "test")
            if first_line:
                real_dir = first_line.split("\\")[0].split("/")[0]
                self.videos_dir = os.path.join(base_path, videos_folder, real_dir)
            else:
                self.videos_dir = os.path.join(base_path, videos_folder, split)
        else:
            self.videos_dir = os.path.join(base_path, videos_folder, split)

        self.meta_json = os.path.join(base_path, meta_json)
        self.subset_json = os.path.join(base_path, subset_json)

        # === Cargar mapas de labels ===
        self.vid2label, self.label2id, self.id2label = load_label_maps(
            self.meta_json,
            self.subset_json
        )

        # === Cargar lista de videos corruptos para este split (opcional) ===
        corrupt_list_path = os.path.join(self.base, f"corrupt_videos_{split}.txt")
        self.corrupt_set = set()
        if os.path.exists(corrupt_list_path):
            with open(corrupt_list_path, "r", encoding="utf-8") as f:
                self.corrupt_set = {line.strip() for line in f if line.strip()}
            print(f"[INFO] Ignorando {len(self.corrupt_set)} videos corruptos definidos en {corrupt_list_path}")

        # === Cargar lista de videos del split ===
        split_txt_path = os.path.join(self.splits_dir, f"{split}_split.txt")
        file_list = load_split_list(split_txt_path)

        # === Construir lista de muestras ===
        self.samples = []
        for raw_fname in file_list:
            # En el txt viene algo como "train\\00623.mp4" o "train/00623.mp4"
            norm = raw_fname.replace("\\", "/")
            basename = os.path.basename(norm)           #  "00623.mp4"

            if not basename.endswith(".mp4"):
                continue

            # Saltar si está marcado como corrupto
            if basename in self.corrupt_set:
                continue

            vid = os.path.splitext(basename)[0]         # "00623"
            video_path = os.path.join(self.videos_dir, basename)

            if os.path.exists(video_path) and vid in self.vid2label:
                label = self.vid2label[vid]
                self.samples.append((video_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron muestras para Split={split}")

        # === Transforms ===
        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        """
        Retorna lista de labels para calcular class weights.
        """
        return [label for _, label in self.samples]

    def __getitem__(self, idx):
        """
        Devuelve (video_tensor, label).

        Si el video está corrupto o vacío, lo salta y prueba con el siguiente
        índice, hasta un máximo de 5 intentos para evitar bucles infinitos.
        """
        original_idx = idx

        for attempt in range(5):
            video_path, label = self.samples[idx]

            try:
                frames = sample_frames_uniform(video_path, NUM_FRAMES)
            except Exception as e:
                print(f"[WARN] Video corrupto o vacío: {video_path} ({e}). Saltando...")
                idx = (idx + 1) % len(self.samples)
                continue

            frames_t = [self.transform(f) for f in frames]
            # VideoMAE espera (B, T, C, H, W) después del DataLoader
            # Cada frame transformado es (C, H, W), stack en dim=0 da (T, C, H, W)
            # El DataLoader añade batch dimension: (B, T, C, H, W) ✓
            video_tensor = torch.stack(frames_t, dim=0)  # (T, C, H, W) = (16, 3, 224, 224)

            return video_tensor, torch.tensor(label, dtype=torch.long)

        raise RuntimeError(
            f"Demasiados videos corruptos seguidos empezando en idx={original_idx}"
        )


def create_dataloaders(config):
    """
    Crea DataLoaders para train y val basándose en la configuración.

    Args:
        config: Diccionario de configuración con claves:
            - data_root: ruta base del dataset
            - batch_size: tamaño del batch
            - num_workers: workers para DataLoader
            - num_classes: número de clases (100 o 300)
            - device: 'cuda' o 'cpu'

    Returns:
        train_loader, val_loader, train_dataset
    """
    from torch.utils.data import DataLoader

    # Determinar base_path según dataset_name
    dataset_name = config.get('dataset_name', 'wlasl100')
    base_path = config.get('data_root', f'data/{dataset_name}')

    print(f"[INFO] Creando dataloaders para: {dataset_name}")
    print(f"[INFO] Base path: {base_path}")

    # Crear datasets
    try:
        train_dataset = WLASLVideoDataset(
            split='train',
            base_path=base_path,
            dataset_size=config.get('num_classes', 100)
        )
        print(f"[INFO] Train dataset: {len(train_dataset)} muestras")
    except Exception as e:
        raise RuntimeError(f"Error al crear train dataset: {e}")

    # Para V2, val usa el test set
    # Detectar si es V2 por el nombre del dataset
    is_v2 = '_v2' in dataset_name

    try:
        if is_v2:
            # V2: val split apunta a test/
            val_dataset = WLASLVideoDataset(
                split='val',  # Lee val_split.txt que apunta a videos en test/
                base_path=base_path,
                dataset_size=config.get('num_classes', 100)
            )
        else:
            # V1: val normal
            val_dataset = WLASLVideoDataset(
                split='val',
                base_path=base_path,
                dataset_size=config.get('num_classes', 100)
            )
        print(f"[INFO] Val dataset: {len(val_dataset)} muestras")
    except Exception as e:
        raise RuntimeError(f"Error al crear val dataset: {e}")

    # Crear dataloaders
    pin_memory = config.get('device', 'cpu') == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 6),
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 6),
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=pin_memory
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}")

    return train_loader, val_loader, train_dataset


