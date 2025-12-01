"""
Training module for VideoMAE WLASL
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import VideoMAEForVideoClassification
from tqdm.auto import tqdm
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd


def compute_class_weights(labels, num_classes, device):
    """Calculate class weights inversely proportional to frequency."""
    class_counts = Counter(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)

    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            weights[class_id] = 1.0 / count
        else:
            weights[class_id] = 0.0

    if weights.sum() > 0:
        weights = weights / weights.mean()

    return weights.to(device)


class WarmupCosineScheduler:
    """Warmup + Cosine Decay LR Scheduler"""

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
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
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


def calculate_accuracy(outputs, labels):
    """Calculate accuracy from logits and labels."""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer, gradient_clip=1.0):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [TRAIN]", leave=False)

    for batch_idx, (videos, labels) in enumerate(progress_bar):
        videos = videos.to(device)
        labels = labels.to(device)

        # === VALIDACIÓN DE DATOS DE ENTRADA ===
        if torch.isnan(videos).any() or torch.isinf(videos).any():
            print(f"\n[WARN] Batch {batch_idx}: Videos con NaN/Inf detectados - SALTANDO")
            print(f"  NaN: {torch.isnan(videos).sum().item()}, Inf: {torch.isinf(videos).sum().item()}")
            continue

        # === FORWARD PASS CON PROTECCIÓN ===
        try:
            outputs = model(pixel_values=videos)
            logits = outputs.logits

            # === VALIDACIÓN Y CLIPPING DE LOGITS ===
            # Detectar valores explosivos ANTES de que causen problemas
            logits_max = logits.abs().max().item()

            if logits_max > 1e10 or torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n[WARN] Batch {batch_idx}: Logits explosivos detectados")
                print(f"  Max logit: {logits_max:.2e}")
                print(f"  Videos range: [{videos.min():.3f}, {videos.max():.3f}]")
                print(f"  SALTANDO batch problemático...")
                continue

            # Clipping de seguridad para logits (prevenir explosión)
            logits = torch.clamp(logits, min=-100, max=100)

        except RuntimeError as e:
            print(f"\n[ERROR] Batch {batch_idx}: Error en forward pass - {e}")
            print(f"  SALTANDO batch...")
            continue

        # === CALCULAR LOSS ===
        loss = criterion(logits, labels)

        # === VALIDACIÓN DE LOSS ===
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e10:
            print(f"\n[WARN] Batch {batch_idx}: Loss inválido detectado")
            print(f"  Loss: {loss.item():.2e}")
            print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            print(f"  SALTANDO batch...")
            continue

        batch_acc = calculate_accuracy(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_acc += batch_acc

        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{batch_acc:.1f}%",
            'lr': f"{current_lr:.2e}"
        })

        global_step = (epoch - 1) * total_batches + batch_idx
        writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
        writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)
        writer.add_scalar('Train/Learning_rate', current_lr, global_step)

    avg_loss = running_loss / total_batches
    avg_acc = running_acc / total_batches

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, epoch, split="VAL"):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [{split:^5}]", leave=False)

    for videos, labels in progress_bar:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=videos)
        logits = outputs.logits

        loss = criterion(logits, labels)

        # Check for NaN/Inf en validation
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\\n[WARN] Loss es NaN/Inf en validation")
            continue

        batch_acc = calculate_accuracy(logits, labels)

        running_loss += loss.item()
        running_acc += batch_acc

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{batch_acc:.1f}%"
        })

    avg_loss = running_loss / total_batches
    avg_acc = running_acc / total_batches

    return avg_loss, avg_acc


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, train_acc, val_loss, val_acc, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
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

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"[CHECKPOINT] Guardado: epoch_{epoch}.pt")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"[BEST MODEL] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


def train_model(config, train_loader, val_loader, train_dataset):
    """
    Complete training pipeline.

    Returns:
        model, training_history, run_checkpoint_dir, log_dir
    """
    device = config['device']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\\n{'='*70}")
    print(f"{'INICIALIZANDO ENTRENAMIENTO':^70}")
    print(f"{'='*70}\\n")

    # Load model
    model = VideoMAEForVideoClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )

    # === REINICIALIZAR CLASSIFIER HEAD CON PESOS CONSERVADORES ===
    # Esto previene logits explosivos desde el inicio
    print("[INFO] Reinicializando classifier head con std=0.01...")
    if hasattr(model, 'classifier'):
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print("✅ Classifier head reinicializado (std=0.01)")
    else:
        print("[WARN] No se encontró 'classifier' - verificando estructura del modelo...")
        # Buscar la última capa linear
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' in name.lower():
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                print(f"✅ Capa '{name}' reinicializada (std=0.01)")

    # === Configurar capas entrenables ===
    freeze_backbone = config.get('freeze_backbone', False)

    if freeze_backbone:
        # Congelar backbone, solo entrenar clasificador
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("⚠️  BACKBONE CONGELADO - Solo clasificador entrenable")
    else:
        # Todas las capas entrenables (por defecto)
        for param in model.parameters():
            param.requires_grad = True
        print("✅ TODAS las 12 capas del modelo VideoMAE son entrenables")

    model.to(device)

    # === VALIDAR QUE EL MODELO NO TIENE NaN ===
    print("[INFO] Verificando inicialización del modelo...")
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_params.append(name)

    if nan_params:
        print(f"[ERROR] Modelo tiene parámetros con NaN/Inf:")
        for name in nan_params:
            print(f"  - {name}")
        raise ValueError("Modelo mal inicializado - contiene NaN/Inf")
    else:
        print("✅ Modelo inicializado correctamente (sin NaN/Inf)")

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Modelo: {config['model_name']}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Parámetros congelados: {frozen_params:,}\\n")

    # Class weights
    class_weights = None
    if config['class_weighted']:
        print("[INFO] Calculando class weights...")
        train_labels = train_dataset.get_labels()
        class_weights = compute_class_weights(train_labels, config['num_classes'], device)
        print(f"Class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f})\\n")

    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config['label_smoothing']
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Scheduler
    total_steps = len(train_loader) * config['max_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config['min_lr']
    )

    print(f"Optimizer: AdamW (lr={config['lr']:.2e}, wd={config['weight_decay']})")
    print(f"Scheduler: Warmup + Cosine Decay")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}\\n")

    # Directories
    run_checkpoint_dir = f"{config['checkpoint_dir']}/run_{timestamp}"
    log_dir = f"{config['logs_dir']}/run_{timestamp}"
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Save config
    import json
    with open(f"{run_checkpoint_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"{'='*70}")
    print(f"{'INICIO DEL ENTRENAMIENTO':^70}")
    print(f"{'='*70}\\n")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improve = 0
    training_history = []

    try:
        for epoch in range(1, config['max_epochs'] + 1):
            print(f"\\n{'='*70}")
            print(f"EPOCH {epoch}/{config['max_epochs']}")
            print(f"{'='*70}")

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch, writer, config['gradient_clip']
            )

            # Validate
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, epoch, "VAL"
            )

            # Log
            current_lr = scheduler.get_last_lr()[0]
            print(f"\\n{'='*70}")
            print(f"RESULTADOS EPOCH {epoch}")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"LR actual:  {current_lr:.2e}")
            print(f"{'='*70}\\n")

            writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
            writer.add_scalar('Train/Accuracy_epoch', train_acc, epoch)
            writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)

            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr
            })

            # Early stopping
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            # Save checkpoint
            if epoch % config['save_every'] == 0 or is_best or epoch == config['max_epochs']:
                save_checkpoint(
                    epoch, model, optimizer, scheduler,
                    train_loss, train_acc, val_loss, val_acc,
                    run_checkpoint_dir, is_best
                )

            # Early stop
            if epochs_without_improve >= config['patience']:
                print(f"\\n[EARLY STOP] No mejora durante {config['patience']} epochs")
                break

    except KeyboardInterrupt:
        print("\\n[INTERRUPTED] Guardando checkpoint...")
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            train_loss, train_acc, val_loss, val_acc,
            run_checkpoint_dir, False
        )

    writer.close()

    # Save history
    history_df = pd.DataFrame(training_history)
    history_path = f"{run_checkpoint_dir}/training_history.csv"
    history_df.to_csv(history_path, index=False)

    print(f"\\n{'='*70}")
    print(f"{'ENTRENAMIENTO COMPLETADO':^70}")
    print(f"{'='*70}")
    print(f"Mejor Val Loss: {best_val_loss:.4f}")
    print(f"Mejor Val Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints: {run_checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print(f"{'='*70}\\n")

    return model, training_history, run_checkpoint_dir, log_dir
