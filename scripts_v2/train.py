"""
train_v2.py - Entrenamiento EXPERIMENTAL VideoMAE (Train+Val Combinados)

Autor: Rafael Ovalle - Tesis UNAB
Dataset: WLASL100_V2/WLASL300_V2 (train+val combinados)
Modelo: VideoMAE (MCG-NJU/videomae-base-finetuned-kinetics)

Configuración V2 - Experimentos para maximizar datos de entrenamiento:
- Train + Val combinados → 1,001 videos de entrenamiento (WLASL100)
- Test usado como validación Y test
- Batch size: 6 (reducido de 16)
- Learning rate: 1e-5 (reducido de 1e-4)
- Weight decay: 0.0 (eliminado)
- Label smoothing: 0.0 (desactivado)
- Class weighting: False (desactivado)
- Patience: 10 epochs (aumentado de 5)
- Todas las capas descongeladas
- Frame sampling: Uniforme (np.linspace)
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm
import numpy as np

from WLASLDataset import WLASLVideoDataset


# ============================================================
#   CONFIGURACIÓN POR DEFECTO (V2 - EXPERIMENTACIÓN)
# ============================================================
DEFAULT_CONFIG = {
    "model_name": "MCG-NJU/videomae-base-finetuned-kinetics",
    "dataset": "wlasl100_v2",  # "wlasl100_v2" (train+val combinados)
    "num_classes": 100,
    "batch_size": 6,  # Reducido de 16 a 6
    "max_epochs": 30,
    "lr": 1e-5,  # Reducido de 1e-4 a 1e-5
    "weight_decay": 0.0,  # Cambiado de 0.05 a 0.0
    "label_smoothing": 0.0,  # Cambiado de 0.1 a 0.0
    "class_weighted": False,  # Cambiado de True a False
    "warmup_ratio": 0.1,
    "min_lr": 1e-6,
    "early_stopping": True,
    "patience": 10,  # Aumentado de 5 a 10
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_path": "data/wlasl100_v2",  # Apunta a nueva estructura v2
    "checkpoint_dir": "models_v2/checkpoints",  # Checkpoints separados
    "logs_dir": "runs_v2",  # Logs separados
    "save_every": 5,
    "gradient_clip": 1.0,
}


# ============================================================
#   FUNCIONES AUXILIARES
# ============================================================
def setup_directories(checkpoint_dir: str, logs_dir: str):
    """Crea directorios necesarios para checkpoints y logs"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Directorio de checkpoints: {checkpoint_dir}")
    print(f"[INFO] Directorio de logs: {logs_dir}")


def compute_class_weights(labels: list, num_classes: int, device: str) -> torch.Tensor:
    """
    Calcula pesos por clase inversamente proporcional a la frecuencia.

    Args:
        labels: Lista de labels del dataset
        num_classes: Número total de clases
        device: Device donde crear el tensor

    Returns:
        Tensor de pesos normalizados
    """
    class_counts = Counter(labels)

    # Inicializar con pequeño epsilon para evitar división por cero
    weights = torch.zeros(num_classes, dtype=torch.float32)

    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            weights[class_id] = 1.0 / count
        else:
            weights[class_id] = 0.0

    # Normalizar para que la media sea 1
    if weights.sum() > 0:
        weights = weights / weights.mean()

    return weights.to(device)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Guarda checkpoint del modelo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }

    # Checkpoint regular
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"[CHECKPOINT] Guardado en: {checkpoint_path}")

    # Mejor modelo
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"[BEST MODEL] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, scheduler=None):
    """Carga checkpoint del modelo"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"[LOAD] Checkpoint cargado desde: {checkpoint_path}")
    print(f"       Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A')}")

    return checkpoint['epoch']


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Calcula accuracy dado logits y labels"""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy


# ============================================================
#   WARMUP + COSINE SCHEDULER CUSTOM
# ============================================================
class WarmupCosineScheduler:
    """
    Scheduler con warmup lineal + cosine decay hasta min_lr
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = last_epoch + 1

    def step(self):
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step < self.warmup_steps:
                # Warmup lineal
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']


# ============================================================
#   ENTRENAMIENTO
# ============================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    gradient_clip: float = 1.0
):
    """Entrena una época"""
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [TRAIN]")

    for batch_idx, (videos, labels) in enumerate(progress_bar):
        # videos: (B, T, C, H, W)
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(pixel_values=videos)
        logits = outputs.logits

        # Loss con label smoothing y class weights
        loss = criterion(logits, labels)

        # Calcular accuracy del batch
        batch_acc = calculate_accuracy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        # Acumular métricas
        running_loss += loss.item()
        running_acc += batch_acc

        # Actualizar progress bar
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{batch_acc:.1f}%",
            'lr': f"{current_lr:.2e}"
        })

        # TensorBoard logging (por batch)
        global_step = (epoch - 1) * total_batches + batch_idx
        writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
        writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)
        writer.add_scalar('Train/Learning_rate', current_lr, global_step)

    # Métricas promedio de la época
    avg_loss = running_loss / total_batches
    avg_acc = running_acc / total_batches

    return avg_loss, avg_acc


# ============================================================
#   EVALUACIÓN
# ============================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    split: str = "VAL"
):
    """Evalúa el modelo en validation o test set"""
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [{split:^5}]")

    for videos, labels in progress_bar:
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(pixel_values=videos)
        logits = outputs.logits

        # Loss
        loss = criterion(logits, labels)

        # Calcular accuracy
        batch_acc = calculate_accuracy(logits, labels)

        # Acumular métricas
        running_loss += loss.item()
        running_acc += batch_acc

        # Actualizar progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{batch_acc:.1f}%"
        })

    # Métricas promedio
    avg_loss = running_loss / total_batches
    avg_acc = running_acc / total_batches

    return avg_loss, avg_acc


# ============================================================
#   FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================
def main(args):
    """Función principal de entrenamiento"""

    # ===== CONFIGURACIÓN =====
    config = DEFAULT_CONFIG.copy()

    # Mapear num_epochs a max_epochs para retrocompatibilidad
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        args.max_epochs = args.num_epochs

    config.update(vars(args))

    # ===== AUTO-CONFIGURACIÓN BASADA EN DATASET =====
    dataset_name = config["dataset"].lower()
    if dataset_name == "wlasl300":
        config["num_classes"] = 300
        config["base_path"] = "data/wlasl300"
        dataset_label = "WLASL300"
    elif dataset_name == "wlasl300_v2":
        config["num_classes"] = 300
        config["base_path"] = "data/wlasl300_v2"
        dataset_label = "WLASL300-V2 (Train+Val Combined)"
    elif dataset_name == "wlasl100_v2":
        config["num_classes"] = 100
        config["base_path"] = "data/wlasl100_v2"
        dataset_label = "WLASL100-V2 (Train+Val Combined)"
    else:  # wlasl100 por defecto
        config["num_classes"] = 100
        config["base_path"] = "data/wlasl100"
        dataset_label = "WLASL100"

    device = config["device"]
    print(f"\n{'='*70}")
    print(f"{'ENTRENAMIENTO VideoMAE - ' + dataset_label + ' (OPTIMIZED)':^70}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_label} ({config['num_classes']} clases)")
    print(f"Base path: {config['base_path']}")
    print(f"Device: {device}")
    print(f"Modelo: {config['model_name']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max epochs: {config['max_epochs']}")
    print(f"Learning rate: {config['lr']:.2e}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Label smoothing: {config['label_smoothing']}")
    print(f"Class weighted: {config['class_weighted']}")
    print(f"Early stopping: {config['early_stopping']} (patience={config['patience']})")
    print(f"{'='*70}\n")

    # ===== DIRECTORIOS =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(config["checkpoint_dir"], f"run_{timestamp}")
    logs_dir = os.path.join(config["logs_dir"], f"run_{timestamp}")
    setup_directories(checkpoint_dir, logs_dir)

    # Guardar configuración
    config_save = {k: v for k, v in config.items() if not k.startswith('_')}
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_save, f, indent=2)
    print(f"[INFO] Configuración guardada en: {config_path}\n")

    # ===== TENSORBOARD =====
    writer = SummaryWriter(log_dir=logs_dir)

    # ===== DATASETS Y DATALOADERS =====
    print("[INFO] Cargando datasets...")
    train_dataset = WLASLVideoDataset(
        split="train",
        base_path=config["base_path"],
        dataset_size=config["num_classes"]
    )
    val_dataset = WLASLVideoDataset(
        split="val",
        base_path=config["base_path"],
        dataset_size=config["num_classes"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if device == "cuda" else False
    )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples: {len(val_dataset)}")
    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}\n")

    # ===== CLASS WEIGHTS =====
    class_weights = None
    if config["class_weighted"]:
        print("[INFO] Calculando class weights...")
        train_labels = train_dataset.get_labels()
        class_weights = compute_class_weights(train_labels, config["num_classes"], device)
        print(f"[INFO] Class weights calculados (min={class_weights.min():.3f}, max={class_weights.max():.3f})\n")

    # ===== LOSS FUNCTION =====
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["label_smoothing"]
    )
    print(f"[INFO] Loss: CrossEntropyLoss (label_smoothing={config['label_smoothing']}, weighted={config['class_weighted']})\n")

    # ===== MODELO =====
    print(f"[INFO] Cargando modelo: {config['model_name']}...")
    model = VideoMAEForVideoClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_classes"],
        ignore_mismatched_sizes=True
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parámetros totales: {total_params:,}")
    print(f"[INFO] Parámetros entrenables: {trainable_params:,}\n")

    # ===== OPTIMIZER =====
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999)
    )

    # ===== SCHEDULER (10% warmup + cosine decay) =====
    total_steps = len(train_loader) * config["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config["min_lr"]
    )

    print(f"[INFO] Optimizer: AdamW (lr={config['lr']:.2e}, wd={config['weight_decay']}, betas=(0.9, 0.999))")
    print(f"[INFO] Scheduler: Warmup ({config['warmup_ratio']*100:.0f}%) + Cosine Decay")
    print(f"[INFO] Total steps: {total_steps}")
    print(f"[INFO] Warmup steps: {warmup_steps}")
    print(f"[INFO] Min LR: {config['min_lr']:.2e}\n")

    # ===== CARGAR CHECKPOINT (OPCIONAL) =====
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler) + 1
        print(f"[INFO] Resumiendo desde epoch {start_epoch}\n")

    # ===== TRAINING LOOP =====
    print(f"{'='*70}")
    print(f"{'INICIO DEL ENTRENAMIENTO':^70}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improve = 0

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{config['max_epochs']}")
        print(f"{'='*70}")

        # ===== TRAIN =====
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            writer=writer,
            gradient_clip=config["gradient_clip"]
        )

        # ===== VALIDATION =====
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            split="VAL"
        )

        # ===== LOGGING =====
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n{'='*70}")
        print(f"RESULTADOS EPOCH {epoch}")
        print(f"{'='*70}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR actual:  {current_lr:.2e}")
        print(f"{'='*70}\n")

        # TensorBoard logging (por época)
        writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_acc, epoch)
        writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)

        # ===== EARLY STOPPING Y CHECKPOINTS =====
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Guardar checkpoint cada N epochs o si es el mejor o último
        if epoch % config["save_every"] == 0 or is_best or epoch == config["max_epochs"]:
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )

        # Early stopping
        if config["early_stopping"] and epochs_without_improve >= config["patience"]:
            print(f"\n[EARLY STOP] No mejora en Val Loss durante {config['patience']} epochs")
            print(f"[EARLY STOP] Deteniendo entrenamiento en epoch {epoch}")
            break

    # ===== FINALIZACIÓN =====
    print(f"\n{'='*70}")
    print(f"{'ENTRENAMIENTO COMPLETADO':^70}")
    print(f"{'='*70}")
    print(f"Mejor Val Loss: {best_val_loss:.4f}")
    print(f"Mejor Val Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints guardados en: {checkpoint_dir}")
    print(f"Logs de TensorBoard en: {logs_dir}")
    print(f"{'='*70}\n")

    writer.close()


# ============================================================
#   ARGUMENTOS DE LÍNEA DE COMANDOS
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[V2] Entrenamiento VideoMAE - Train+Val Combinados - WLASL100/WLASL300",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG["dataset"],
                        choices=["wlasl100", "wlasl300", "wlasl100_v2", "wlasl300_v2"],
                        help="Dataset a utilizar: wlasl100/300 (original) o wlasl100_v2/300_v2 (train+val combinados)")

    # Modelo
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Nombre del modelo pre-entrenado de Hugging Face")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_CONFIG["num_classes"],
                        help="Número de clases del dataset (se ajusta automáticamente con --dataset)")

    # Entrenamiento
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size para entrenamiento")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_CONFIG["max_epochs"],
                        help="Número máximo de epochs a entrenar")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Alias de max_epochs (retrocompatibilidad)")

    # Hiperparámetros optimizados
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"],
                        help="Learning rate inicial")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"],
                        help="Weight decay para regularización L2")
    parser.add_argument("--label_smoothing", type=float, default=DEFAULT_CONFIG["label_smoothing"],
                        help="Label smoothing para CrossEntropyLoss")
    parser.add_argument("--class_weighted", type=lambda x: str(x).lower() == 'true',
                        default=DEFAULT_CONFIG["class_weighted"],
                        help="Usar class weights inversamente proporcionales a frecuencia")

    # Scheduler
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"],
                        help="Ratio de warmup respecto al total de steps (0.1 = 10%%)")
    parser.add_argument("--min_lr", type=float, default=DEFAULT_CONFIG["min_lr"],
                        help="Learning rate mínimo en cosine decay")

    # Early stopping
    parser.add_argument("--early_stopping", type=lambda x: str(x).lower() == 'true',
                        default=DEFAULT_CONFIG["early_stopping"],
                        help="Activar early stopping basado en Val Loss")
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"],
                        help="Epochs sin mejora antes de early stopping")

    parser.add_argument("--gradient_clip", type=float, default=DEFAULT_CONFIG["gradient_clip"],
                        help="Valor máximo para gradient clipping (0 = desactivado)")

    # Dataset
    parser.add_argument("--base_path", type=str, default=DEFAULT_CONFIG["base_path"],
                        help="Ruta base del dataset WLASL100")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"],
                        help="Número de workers para DataLoader")

    # Directorios
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CONFIG["checkpoint_dir"],
                        help="Directorio para guardar checkpoints")
    parser.add_argument("--logs_dir", type=str, default=DEFAULT_CONFIG["logs_dir"],
                        help="Directorio para logs de TensorBoard")
    parser.add_argument("--save_every", type=int, default=DEFAULT_CONFIG["save_every"],
                        help="Guardar checkpoint cada N epochs")

    # Utilidades
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta al checkpoint para resumir entrenamiento")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"],
                        help="Device para entrenamiento (cuda/cpu)")

    args = parser.parse_args()

    main(args)
