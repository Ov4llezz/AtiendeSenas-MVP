"""
Configuration module for VideoMAE WLASL training
"""

import os
import json
from datetime import datetime
from pathlib import Path


def setup_environment():
    """
    Setup environment and install dependencies.
    """
    print("🔧 Installing dependencies...")

    import subprocess
    import sys

    packages = [
        "transformers==4.36.0",
        "opencv-python-headless",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tensorboard",
        "pandas",
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("✅ Dependencies installed\n")


def create_config(
    dataset_type="wlasl100",
    version="v1",
    data_root="/home/ov4lle/AtiendeSenas-MVP/data",

    # === HIPERPARÁMETROS CONFIGURABLES ===
    # Valores por defecto: aplicables tanto a V1 como V2
    batch_size=6,
    max_epochs=30,
    learning_rate=1e-5,
    patience=10,
    weight_decay=0.0,
    label_smoothing=0.0,
    class_weighted=False,
    freeze_backbone=False,

    # === OTROS PARÁMETROS ===
    warmup_ratio=0.1,
    min_lr=1e-6,
    gradient_clip=1.0,
    num_workers=2,
    save_every=5,
):
    """
    Crea configuración para entrenamiento de VideoMAE en WLASL.

    Args:
        dataset_type: "wlasl100" o "wlasl300"
        version: "v1" (train/val/test separados) o "v2" (train+val combinados)
        data_root: Ruta base donde están los datasets (default: VM path)

        # Hiperparámetros (GENERALES para V1 y V2)
        batch_size: Tamaño del batch (default: 6)
        max_epochs: Número máximo de epochs (default: 30)
        learning_rate: Learning rate inicial (default: 1e-5)
        patience: Epochs sin mejora para early stopping (default: 10)
        weight_decay: Regularización L2 (default: 0.0 = desactivado)
        label_smoothing: Label smoothing (default: 0.0 = desactivado)
        class_weighted: Usar pesos de clases en pérdida (default: False)
        freeze_backbone: Congelar backbone VideoMAE (default: False = entrenar todo)

        # Otros parámetros
        warmup_ratio: Porcentaje de warmup (default: 0.1)
        min_lr: Learning rate mínimo (default: 1e-6)
        gradient_clip: Gradient clipping (default: 1.0)
        num_workers: Workers para DataLoader (default: 2)
        save_every: Guardar checkpoint cada N epochs (default: 5)

    Returns:
        Diccionario de configuración completo
    """
    import torch

    # Determinar número de clases y nombre del dataset
    if dataset_type == "wlasl100":
        num_classes = 100
        dataset_name = "wlasl100_v2" if version == "v2" else "wlasl100"
    elif dataset_type == "wlasl300":
        num_classes = 300
        dataset_name = "wlasl300_v2" if version == "v2" else "wlasl300"
    else:
        raise ValueError("dataset_type debe ser 'wlasl100' o 'wlasl300'")

    # Validar versión
    if version not in ["v1", "v2"]:
        raise ValueError("version debe ser 'v1' o 'v2'")

    # Configuración completa
    config = {
        # Modelo
        "model_name": "MCG-NJU/videomae-base-finetuned-kinetics",
        "num_classes": num_classes,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "version": version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Hiperparámetros (GENERALES, no específicos por versión)
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "class_weighted": class_weighted,
        "patience": patience,
        "freeze_backbone": freeze_backbone,

        # Scheduler y optimización
        "warmup_ratio": warmup_ratio,
        "min_lr": min_lr,
        "gradient_clip": gradient_clip,

        # DataLoader
        "num_workers": num_workers,

        # Checkpointing
        "save_every": save_every,
    }

    # Rutas (basadas en data_root, NO drive_root)
    # La estructura es: /home/ov4lle/AtiendeSenas-MVP/data/{dataset_name}
    config.update({
        "data_root": f"{data_root}/{dataset_name}",
        "checkpoint_dir": f"{data_root}/../models/{version}/{dataset_name}/checkpoints",
        "logs_dir": f"{data_root}/../runs/{version}/{dataset_name}",
        "results_dir": f"{data_root}/../results/{version}/{dataset_name}",
    })

    # Crear directorios si no existen
    for key in ["checkpoint_dir", "logs_dir", "results_dir"]:
        os.makedirs(config[key], exist_ok=True)

    return config


def print_config(config):
    """
    Imprime la configuración en formato legible.
    """
    print(f"\\n{'='*70}")
    print(f"{'CONFIGURACIÓN DEL EXPERIMENTO':^70}")
    print(f"{'='*70}")
    print(f"Dataset: {config['dataset_type'].upper()} ({config['num_classes']} clases)")
    print(f"Versión: {config['version'].upper()}")
    print(f"Dataset Name: {config['dataset_name']}")

    print(f"\\n{'Hiperparámetros Principales':─^70}")
    print(f"  Batch Size:         {config['batch_size']}")
    print(f"  Max Epochs:         {config['max_epochs']}")
    print(f"  Learning Rate:      {config['lr']:.2e}")
    print(f"  Patience (E.Stop):  {config['patience']} epochs")

    print(f"\\n{'Regularización':─^70}")
    print(f"  Weight Decay:       {config['weight_decay']}")
    print(f"  Label Smoothing:    {config['label_smoothing']}")
    print(f"  Class Weighted:     {config['class_weighted']}")

    print(f"\\n{'Modelo':─^70}")
    print(f"  Freeze Backbone:    {config.get('freeze_backbone', False)}")
    print(f"  Gradient Clip:      {config['gradient_clip']}")

    print(f"\\n{'Scheduler':─^70}")
    print(f"  Warmup Ratio:       {config['warmup_ratio']}")
    print(f"  Min LR:             {config['min_lr']:.2e}")

    print(f"\\n{'Rutas':─^70}")
    print(f"  Data:        {config['data_root']}")
    print(f"  Checkpoints: {config['checkpoint_dir']}")
    print(f"  Logs:        {config['logs_dir']}")
    print(f"  Results:     {config['results_dir']}")

    print(f"\\n{'Sistema':─^70}")
    print(f"  Device:             {config['device']}")
    print(f"  Num Workers:        {config['num_workers']}")
    print(f"  Save Every:         {config['save_every']} epochs")
    print(f"{'='*70}\\n")


def save_config(config, output_dir):
    """
    Save configuration to JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = f"{output_dir}/config_{timestamp}.json"

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuración guardada en: {config_path}")
    return config_path
