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


def create_config(dataset_type="wlasl100", version="v1", drive_root="/content/drive/MyDrive/TESIS_WLASL"):
    """
    Create configuration dictionary based on dataset and version.

    Args:
        dataset_type: "wlasl100" or "wlasl300"
        version: "v1" (baseline) or "v2" (experimental)
        drive_root: Root directory in Google Drive

    Returns:
        Configuration dictionary
    """
    import torch

    # Determine num_classes and dataset_name
    if dataset_type == "wlasl100":
        num_classes = 100
        dataset_name = "wlasl100_v2" if version == "v2" else "wlasl100"
    elif dataset_type == "wlasl300":
        num_classes = 300
        dataset_name = "wlasl300_v2" if version == "v2" else "wlasl300"
    else:
        raise ValueError("dataset_type must be 'wlasl100' or 'wlasl300'")

    # Base configuration
    config = {
        "model_name": "MCG-NJU/videomae-base-finetuned-kinetics",
        "num_classes": num_classes,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "version": version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Version-specific hyperparameters
    if version == "v1":
        config.update({
            "batch_size": 16,
            "max_epochs": 30,
            "lr": 1e-4,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "class_weighted": True,
            "patience": 5,
        })
    elif version == "v2":
        config.update({
            "batch_size": 6,
            "max_epochs": 30,
            "lr": 1e-5,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "class_weighted": False,
            "patience": 10,
        })
    else:
        raise ValueError("version must be 'v1' or 'v2'")

    # Common hyperparameters
    config.update({
        "warmup_ratio": 0.1,
        "min_lr": 1e-6,
        "gradient_clip": 1.0,
        "num_workers": 2,
        "save_every": 5,
    })

    # Paths
    config.update({
        "data_root": f"{drive_root}/data/{dataset_name}",
        "checkpoint_dir": f"{drive_root}/models/{version}/{dataset_name}/checkpoints",
        "logs_dir": f"{drive_root}/runs/{version}/{dataset_name}",
        "results_dir": f"{drive_root}/results/{version}/{dataset_name}",
    })

    # Create directories
    for key in ["checkpoint_dir", "logs_dir", "results_dir"]:
        os.makedirs(config[key], exist_ok=True)

    return config


def print_config(config):
    """
    Print configuration in a nice format.
    """
    print(f"\\n{'='*70}")
    print(f"{'CONFIGURACIÓN DEL EXPERIMENTO':^70}")
    print(f"{'='*70}")
    print(f"Dataset: {config['dataset_type'].upper()} ({config['num_classes']} clases)")
    print(f"Versión: {config['version'].upper()}")
    print(f"Dataset Name: {config['dataset_name']}")
    print(f"\\nHiperparámetros:")
    print(f"  - Batch Size: {config['batch_size']}")
    print(f"  - Learning Rate: {config['lr']:.2e}")
    print(f"  - Weight Decay: {config['weight_decay']}")
    print(f"  - Label Smoothing: {config['label_smoothing']}")
    print(f"  - Class Weighted: {config['class_weighted']}")
    print(f"  - Patience: {config['patience']}")
    print(f"  - Max Epochs: {config['max_epochs']}")
    print(f"\\nRutas:")
    print(f"  - Data: {config['data_root']}")
    print(f"  - Checkpoints: {config['checkpoint_dir']}")
    print(f"  - Logs: {config['logs_dir']}")
    print(f"  - Results: {config['results_dir']}")
    print(f"\\nDevice: {config['device']}")
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
