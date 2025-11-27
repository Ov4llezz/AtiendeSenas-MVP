"""
test.py - Evaluación completa del modelo VideoMAE en test set

Autor: Rafael Ovalle - Tesis UNAB
Dataset: WLASL100/WLASL300 (100 o 300 clases de lengua de señas)

Genera:
- Accuracy total y por clase
- Matriz de confusión
- Precision, Recall, F1-score
- Top-K accuracy
- Reporte detallado en JSON y TXT
- Visualizaciones (confusion matrix heatmap)
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

from WLASLDataset import WLASLVideoDataset


# ============================================================
#   CONFIGURACIÓN
# ============================================================
DEFAULT_CONFIG = {
    "checkpoint_path": None,  # Requerido por CLI
    "base_path": "data/wlasl100",
    "batch_size": 16,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "evaluation_results",
    "top_k": [1, 3, 5],  # Top-K accuracy
}


# ============================================================
#   FUNCIONES AUXILIARES
# ============================================================
def list_available_runs(checkpoint_dir: str = "models/checkpoints"):
    """
    Lista todos los runs disponibles con su información.

    Returns:
        Lista de dicts con información de cada run
    """
    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Directorio {checkpoint_dir} no existe")
        return []

    # Buscar todos los directorios run_*
    run_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "run_*")), reverse=True)

    if not run_dirs:
        print(f"[ERROR] No se encontraron runs en {checkpoint_dir}")
        return []

    runs_info = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        best_model_path = os.path.join(run_dir, "best_model.pt")
        config_path = os.path.join(run_dir, "config.json")

        # Verificar que exista best_model.pt
        if not os.path.exists(best_model_path):
            continue

        run_info = {
            "run_name": run_name,
            "run_dir": run_dir,
            "best_model_path": best_model_path,
            "epoch": "N/A",
            "val_acc": "N/A",
            "val_loss": "N/A",
            "config": {}
        }

        # Leer config.json si existe
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    run_info["config"] = json.load(f)
            except:
                pass

        # Leer best_model.pt para obtener métricas
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')

            # Intentar múltiples claves posibles (compatibilidad con versiones anteriores)
            run_info["epoch"] = checkpoint.get('epoch', checkpoint.get('epochs', 'N/A'))

            # Para val_acc: intentar varias claves
            val_acc = checkpoint.get('val_acc')
            if val_acc is None:
                val_acc = checkpoint.get('best_val_acc')
            if val_acc is None:
                val_acc = checkpoint.get('accuracy')
            run_info["val_acc"] = val_acc if val_acc is not None else 'N/A'

            # Para val_loss: intentar varias claves
            val_loss = checkpoint.get('val_loss')
            if val_loss is None:
                val_loss = checkpoint.get('best_val_loss')
            if val_loss is None:
                val_loss = checkpoint.get('loss')
            run_info["val_loss"] = val_loss if val_loss is not None else 'N/A'

        except Exception as e:
            # Si falla, simplemente dejar N/A
            print(f"[WARN] No se pudieron leer métricas de {run_name}: {e}")
            pass

        runs_info.append(run_info)

    return runs_info


def print_available_runs(runs_info: list):
    """Imprime la lista de runs disponibles de forma legible"""
    if not runs_info:
        print("[INFO] No hay runs disponibles")
        return

    print(f"\n{'='*80}")
    print(f"{'RUNS DISPONIBLES':^80}")
    print(f"{'='*80}\n")

    print(f"{'ID':<4} {'Run Name':<25} {'Epoch':<8} {'Val Acc':<12} {'Val Loss':<12}")
    print("-"*80)

    for idx, run in enumerate(runs_info, 1):
        # Formatear val_acc
        if isinstance(run['val_acc'], (int, float)):
            val_acc_str = f"{run['val_acc']:.2f}%"
        else:
            val_acc_str = str(run['val_acc'])

        # Formatear val_loss
        if isinstance(run['val_loss'], (int, float)):
            val_loss_str = f"{run['val_loss']:.4f}"
        else:
            val_loss_str = str(run['val_loss'])

        # Formatear epoch
        epoch_str = str(run['epoch'])

        print(f"{idx:<4} {run['run_name']:<25} {epoch_str:<8} {val_acc_str:<12} {val_loss_str:<12}")

    print(f"\n{'='*80}")
    print("Para evaluar un run específico, usa: --run-id <ID>")
    print("Ejemplo: python scripts/test.py --run-id 1")
    print(f"\nNOTA: Si ves 'N/A', el checkpoint no tiene esos metadatos guardados,")
    print("      pero aún puedes evaluar el modelo normalmente con --run-id <ID>")
    print(f"{'='*80}\n")


def get_checkpoint_from_run_id(run_id: int, checkpoint_dir: str = "models/checkpoints") -> str:
    """
    Obtiene la ruta del checkpoint dado un run_id.

    Args:
        run_id: ID del run (1-indexed)
        checkpoint_dir: Directorio de checkpoints

    Returns:
        Ruta al best_model.pt del run seleccionado
    """
    runs_info = list_available_runs(checkpoint_dir)

    if not runs_info:
        raise ValueError("No hay runs disponibles")

    if run_id < 1 or run_id > len(runs_info):
        raise ValueError(f"run_id debe estar entre 1 y {len(runs_info)}")

    selected_run = runs_info[run_id - 1]

    print(f"\n[INFO] Run seleccionado: {selected_run['run_name']}")
    print(f"[INFO] Epoch: {selected_run['epoch']}")
    print(f"[INFO] Val Accuracy: {selected_run['val_acc']}")
    print(f"[INFO] Val Loss: {selected_run['val_loss']}\n")

    return selected_run["best_model_path"]


def load_model(checkpoint_path: str, device: str):
    """Carga modelo desde checkpoint"""
    print(f"[INFO] Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Obtener configuración del modelo
    config_dir = Path(checkpoint_path).parent
    config_path = config_dir / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            train_config = json.load(f)
        model_name = train_config.get("model_name", "MCG-NJU/videomae-base-finetuned-kinetics")
        num_classes = train_config.get("num_classes", 100)
        print(f"[INFO] Modelo: {model_name}")
        print(f"[INFO] Num classes: {num_classes}")
    else:
        print("[WARN] config.json no encontrado, usando defaults")
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        num_classes = 100

    # Cargar modelo
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Información del checkpoint - intentar múltiples claves para compatibilidad
    epoch = checkpoint.get('epoch')
    if epoch is None:
        epoch = checkpoint.get('epochs', 'N/A')

    val_acc = checkpoint.get('val_acc')
    if val_acc is None:
        val_acc = checkpoint.get('best_val_acc')
    if val_acc is None:
        val_acc = checkpoint.get('accuracy')
    if val_acc is None:
        val_acc = 'N/A'

    val_loss = checkpoint.get('val_loss')
    if val_loss is None:
        val_loss = checkpoint.get('best_val_loss')
    if val_loss is None:
        val_loss = checkpoint.get('loss')
    if val_loss is None:
        val_loss = 'N/A'

    print(f"[INFO] Checkpoint epoch: {epoch}")
    print(f"[INFO] Val Accuracy: {val_acc}")
    print(f"[INFO] Val Loss: {val_loss}\n")

    return model, train_config if config_path.exists() else {}


def calculate_top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Calcula Top-K accuracy"""
    _, top_k_preds = torch.topk(logits, k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    top_k_acc = correct.sum().float() / labels.size(0) * 100.0
    return top_k_acc.item()


# ============================================================
#   EVALUACIÓN
# ============================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    top_k_list: list = [1, 3, 5]
):
    """
    Evalúa el modelo y retorna métricas detalladas.

    Returns:
        dict con:
        - predictions: lista de predicciones
        - labels: lista de labels verdaderos
        - logits: lista de logits
        - top_k_accuracies: dict con top-k accuracies
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_logits = []

    print("[INFO] Evaluando en test set...")
    progress_bar = tqdm(dataloader, desc="Evaluación")

    for videos, labels in progress_bar:
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(pixel_values=videos)
        logits = outputs.logits

        # Predicciones
        predictions = torch.argmax(logits, dim=1)

        # Guardar resultados
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu())

    # Concatenar logits
    all_logits = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.tensor(all_labels)

    # Calcular Top-K accuracies
    top_k_accuracies = {}
    for k in top_k_list:
        top_k_acc = calculate_top_k_accuracy(all_logits, all_labels_tensor, k)
        top_k_accuracies[f"top_{k}"] = top_k_acc

    return {
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "logits": all_logits.numpy(),
        "top_k_accuracies": top_k_accuracies
    }


# ============================================================
#   MÉTRICAS Y REPORTES
# ============================================================
def compute_metrics(predictions: np.ndarray, labels: np.ndarray, num_classes: int):
    """Calcula métricas detalladas"""

    # Accuracy total
    total_acc = accuracy_score(labels, predictions) * 100.0

    # Precision, Recall, F1 (macro y weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # Por clase
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0, labels=range(num_classes)
    )

    # Accuracy por clase
    conf_matrix = confusion_matrix(labels, predictions, labels=range(num_classes))
    class_accuracies = []
    for i in range(num_classes):
        if support_per_class[i] > 0:
            class_acc = conf_matrix[i, i] / support_per_class[i] * 100.0
        else:
            class_acc = 0.0
        class_accuracies.append(class_acc)

    return {
        "total_accuracy": total_acc,
        "precision_macro": precision_macro * 100.0,
        "recall_macro": recall_macro * 100.0,
        "f1_macro": f1_macro * 100.0,
        "precision_weighted": precision_weighted * 100.0,
        "recall_weighted": recall_weighted * 100.0,
        "f1_weighted": f1_weighted * 100.0,
        "per_class": {
            "accuracy": class_accuracies,
            "precision": (precision_per_class * 100.0).tolist(),
            "recall": (recall_per_class * 100.0).tolist(),
            "f1": (f1_per_class * 100.0).tolist(),
            "support": support_per_class.tolist()
        },
        "confusion_matrix": conf_matrix
    }


def plot_confusion_matrix(conf_matrix: np.ndarray, output_dir: str, top_n: int = 20):
    """
    Genera heatmap de la matriz de confusión.
    Si hay muchas clases, muestra solo las top_n con más muestras.
    """
    num_classes = conf_matrix.shape[0]

    # Si hay muchas clases, mostrar solo las más frecuentes
    if num_classes > top_n:
        class_totals = conf_matrix.sum(axis=1)
        top_classes = np.argsort(class_totals)[-top_n:][::-1]
        conf_matrix_subset = conf_matrix[np.ix_(top_classes, top_classes)]
        title = f"Confusion Matrix (Top {top_n} clases con más muestras)"
    else:
        conf_matrix_subset = conf_matrix
        top_classes = range(num_classes)
        title = "Confusion Matrix (Todas las clases)"

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix_subset,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=top_classes,
        yticklabels=top_classes,
        cbar_kws={'label': 'Número de muestras'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.ylabel('Clase Real', fontsize=12)
    plt.tight_layout()

    # Guardar
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVE] Confusion matrix guardada en: {output_path}")


def plot_class_performance(metrics: dict, output_dir: str, top_n: int = 20):
    """Genera gráfico de las mejores y peores clases"""

    class_accs = np.array(metrics["per_class"]["accuracy"])
    class_support = np.array(metrics["per_class"]["support"])

    # Filtrar clases sin muestras
    valid_classes = class_support > 0
    class_accs_valid = class_accs[valid_classes]
    class_ids_valid = np.arange(len(class_accs))[valid_classes]

    # Top y Bottom N
    sorted_indices = np.argsort(class_accs_valid)
    bottom_indices = sorted_indices[:top_n]
    top_indices = sorted_indices[-top_n:][::-1]

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top N mejores clases
    top_classes = class_ids_valid[top_indices]
    top_accs = class_accs_valid[top_indices]
    axes[0].barh(range(len(top_classes)), top_accs, color='green', alpha=0.7)
    axes[0].set_yticks(range(len(top_classes)))
    axes[0].set_yticklabels([f"Clase {c}" for c in top_classes])
    axes[0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0].set_title(f'Top {len(top_classes)} Mejores Clases', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Bottom N peores clases
    bottom_classes = class_ids_valid[bottom_indices]
    bottom_accs = class_accs_valid[bottom_indices]
    axes[1].barh(range(len(bottom_classes)), bottom_accs, color='red', alpha=0.7)
    axes[1].set_yticks(range(len(bottom_classes)))
    axes[1].set_yticklabels([f"Clase {c}" for c in bottom_classes])
    axes[1].set_xlabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Top {len(bottom_classes)} Peores Clases', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Guardar
    output_path = os.path.join(output_dir, "class_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVE] Class performance guardado en: {output_path}")


def save_results(
    metrics: dict,
    top_k_accs: dict,
    output_dir: str,
    checkpoint_info: dict
):
    """Guarda resultados en JSON y TXT"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== JSON =====
    json_results = {
        "timestamp": timestamp,
        "checkpoint_info": checkpoint_info,
        "overall_metrics": {
            "total_accuracy": metrics["total_accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        },
        "top_k_accuracies": top_k_accs,
        "per_class_metrics": metrics["per_class"]
    }

    json_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"[SAVE] JSON results guardado en: {json_path}")

    # ===== TXT =====
    txt_path = os.path.join(output_dir, f"test_results_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("EVALUACIÓN EN TEST SET - VideoMAE WLASL100\n")
        f.write("="*70 + "\n\n")

        f.write(f"Fecha: {timestamp}\n")
        f.write(f"Checkpoint: {checkpoint_info.get('checkpoint_path', 'N/A')}\n")
        f.write(f"Epoch: {checkpoint_info.get('epoch', 'N/A')}\n")
        f.write(f"Val Accuracy: {checkpoint_info.get('val_acc', 'N/A')}\n")
        f.write(f"Val Loss: {checkpoint_info.get('val_loss', 'N/A')}\n\n")

        f.write("="*70 + "\n")
        f.write("MÉTRICAS GLOBALES\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total Accuracy:      {metrics['total_accuracy']:.2f}%\n\n")

        f.write(f"Precision (Macro):   {metrics['precision_macro']:.2f}%\n")
        f.write(f"Recall (Macro):      {metrics['recall_macro']:.2f}%\n")
        f.write(f"F1-Score (Macro):    {metrics['f1_macro']:.2f}%\n\n")

        f.write(f"Precision (Weighted):{metrics['precision_weighted']:.2f}%\n")
        f.write(f"Recall (Weighted):   {metrics['recall_weighted']:.2f}%\n")
        f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.2f}%\n\n")

        f.write("="*70 + "\n")
        f.write("TOP-K ACCURACIES\n")
        f.write("="*70 + "\n\n")

        for k, acc in top_k_accs.items():
            f.write(f"{k.upper()}: {acc:.2f}%\n")

        f.write("\n" + "="*70 + "\n")
        f.write("MÉTRICAS POR CLASE (Top 10 Mejores)\n")
        f.write("="*70 + "\n\n")

        class_accs = np.array(metrics["per_class"]["accuracy"])
        class_support = np.array(metrics["per_class"]["support"])
        valid_classes = class_support > 0

        sorted_indices = np.argsort(class_accs)[::-1]
        sorted_valid = [i for i in sorted_indices if valid_classes[i]][:10]

        f.write(f"{'Clase':<8} {'Acc(%)':<8} {'Prec(%)':<8} {'Rec(%)':<8} {'F1(%)':<8} {'Support':<10}\n")
        f.write("-"*70 + "\n")

        for idx in sorted_valid:
            f.write(f"{idx:<8} "
                   f"{metrics['per_class']['accuracy'][idx]:<8.2f} "
                   f"{metrics['per_class']['precision'][idx]:<8.2f} "
                   f"{metrics['per_class']['recall'][idx]:<8.2f} "
                   f"{metrics['per_class']['f1'][idx]:<8.2f} "
                   f"{int(metrics['per_class']['support'][idx]):<10}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("MÉTRICAS POR CLASE (Top 10 Peores)\n")
        f.write("="*70 + "\n\n")

        sorted_indices_asc = np.argsort(class_accs)
        sorted_valid_asc = [i for i in sorted_indices_asc if valid_classes[i]][:10]

        f.write(f"{'Clase':<8} {'Acc(%)':<8} {'Prec(%)':<8} {'Rec(%)':<8} {'F1(%)':<8} {'Support':<10}\n")
        f.write("-"*70 + "\n")

        for idx in sorted_valid_asc:
            f.write(f"{idx:<8} "
                   f"{metrics['per_class']['accuracy'][idx]:<8.2f} "
                   f"{metrics['per_class']['precision'][idx]:<8.2f} "
                   f"{metrics['per_class']['recall'][idx]:<8.2f} "
                   f"{metrics['per_class']['f1'][idx]:<8.2f} "
                   f"{int(metrics['per_class']['support'][idx]):<10}\n")

    print(f"[SAVE] TXT results guardado en: {txt_path}")


# ============================================================
#   FUNCIÓN PRINCIPAL
# ============================================================
def main(args):
    """Función principal de evaluación"""

    # ===== MANEJO DE --list-runs =====
    if args.list_runs:
        runs_info = list_available_runs(args.checkpoint_dir)
        print_available_runs(runs_info)
        return

    # ===== MANEJO DE --run-id =====
    if args.run_id is not None:
        try:
            checkpoint_path = get_checkpoint_from_run_id(args.run_id, args.checkpoint_dir)
            args.checkpoint_path = checkpoint_path
        except ValueError as e:
            print(f"[ERROR] {e}")
            print("\nUsa --list-runs para ver los runs disponibles")
            return

    # ===== VALIDAR QUE SE ESPECIFICÓ UN CHECKPOINT =====
    if args.checkpoint_path is None:
        print("[ERROR] Debes especificar uno de los siguientes:")
        print("  --checkpoint_path <ruta>   : Ruta específica al checkpoint")
        print("  --run-id <ID>              : ID del run (usa --list-runs para ver opciones)")
        print("  --list-runs                : Listar todos los runs disponibles")
        return

    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))

    device = config["device"]

    # ===== CARGAR MODELO PRIMERO PARA OBTENER CONFIG =====
    print(f"\n{'='*70}")
    print(f"{'CARGANDO MODELO Y CONFIGURACIÓN':^70}")
    print(f"{'='*70}\n")
    model, train_config = load_model(config["checkpoint_path"], device)

    # ===== CONFIGURAR DATASET BASADO EN TRAIN_CONFIG =====
    num_classes = train_config.get("num_classes", 100)
    # Usar base_path del entrenamiento si está disponible, sino usar el default/especificado
    if "base_path" in train_config:
        config["base_path"] = train_config["base_path"]
    # Auto-detectar dataset basado en num_classes si base_path no fue especificado por CLI
    elif config["base_path"] == "data/wlasl100" and num_classes == 300:
        config["base_path"] = "data/wlasl300"

    dataset_label = "WLASL300" if num_classes == 300 else "WLASL100"

    print(f"\n{'='*70}")
    print(f"{'EVALUACIÓN EN TEST SET - VideoMAE ' + dataset_label:^70}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_label} ({num_classes} clases)")
    print(f"Base path: {config['base_path']}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"{'='*70}\n")

    # ===== CREAR DIRECTORIO DE SALIDA =====
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ===== CARGAR TEST DATASET =====
    print("[INFO] Cargando test dataset...")
    test_dataset = WLASLVideoDataset(
        split="test",
        base_path=config["base_path"],
        dataset_size=num_classes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if device == "cuda" else False
    )

    print(f"[INFO] Test samples: {len(test_dataset)}")
    print(f"[INFO] Test batches: {len(test_loader)}\n")

    # ===== EVALUACIÓN =====
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        top_k_list=config["top_k"]
    )

    # ===== CALCULAR MÉTRICAS =====
    print("\n[INFO] Calculando métricas...")
    num_classes = train_config.get("num_classes", 100)
    metrics = compute_metrics(
        predictions=results["predictions"],
        labels=results["labels"],
        num_classes=num_classes
    )

    # ===== IMPRIMIR RESULTADOS =====
    print(f"\n{'='*70}")
    print(f"{'RESULTADOS':^70}")
    print(f"{'='*70}")
    print(f"Total Accuracy:      {metrics['total_accuracy']:.2f}%")
    print(f"Precision (Macro):   {metrics['precision_macro']:.2f}%")
    print(f"Recall (Macro):      {metrics['recall_macro']:.2f}%")
    print(f"F1-Score (Macro):    {metrics['f1_macro']:.2f}%")
    print(f"{'='*70}")

    print(f"\n{'TOP-K ACCURACIES':^70}")
    print(f"{'='*70}")
    for k, acc in results["top_k_accuracies"].items():
        print(f"{k.upper()}: {acc:.2f}%")
    print(f"{'='*70}\n")

    # ===== VISUALIZACIONES =====
    print("[INFO] Generando visualizaciones...")
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir, top_n=20)
    plot_class_performance(metrics, output_dir, top_n=20)

    # ===== GUARDAR RESULTADOS =====
    print("\n[INFO] Guardando resultados...")

    # Leer checkpoint info directamente del checkpoint para asegurar compatibilidad
    checkpoint = torch.load(config["checkpoint_path"], map_location='cpu')

    epoch = checkpoint.get('epoch')
    if epoch is None:
        epoch = checkpoint.get('epochs', 'N/A')

    val_acc = checkpoint.get('val_acc')
    if val_acc is None:
        val_acc = checkpoint.get('best_val_acc')
    if val_acc is None:
        val_acc = checkpoint.get('accuracy')
    if val_acc is None:
        val_acc = 'N/A'

    val_loss = checkpoint.get('val_loss')
    if val_loss is None:
        val_loss = checkpoint.get('best_val_loss')
    if val_loss is None:
        val_loss = checkpoint.get('loss')
    if val_loss is None:
        val_loss = 'N/A'

    checkpoint_info = {
        "checkpoint_path": config["checkpoint_path"],
        "epoch": epoch,
        "val_acc": val_acc,
        "val_loss": val_loss,
    }

    save_results(metrics, results["top_k_accuracies"], output_dir, checkpoint_info)

    # ===== FINALIZACIÓN =====
    print(f"\n{'='*70}")
    print(f"{'EVALUACIÓN COMPLETADA':^70}")
    print(f"{'='*70}")
    print(f"Resultados guardados en: {output_dir}/")
    print(f"{'='*70}\n")


# ============================================================
#   ARGUMENTOS CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación de modelo VideoMAE en test set de WLASL100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Checkpoint (una de las tres opciones es requerida)
    checkpoint_group = parser.add_argument_group('Checkpoint Selection (elige una opción)')
    checkpoint_group.add_argument("--checkpoint_path", type=str, default=None,
                        help="Ruta específica al checkpoint (.pt) a evaluar")
    checkpoint_group.add_argument("--run-id", type=int, default=None,
                        help="ID del run a evaluar (usa --list-runs para ver opciones)")
    checkpoint_group.add_argument("--list-runs", action="store_true",
                        help="Listar todos los runs disponibles y salir")

    # Directorio de checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints",
                        help="Directorio donde están los runs")

    # Dataset
    parser.add_argument("--base_path", type=str, default=DEFAULT_CONFIG["base_path"],
                        help="Ruta base del dataset WLASL100")

    # Evaluación
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size para evaluación")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"],
                        help="Número de workers para DataLoader")

    # Output
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directorio para guardar resultados")

    # Device
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"],
                        help="Device para evaluación (cuda/cpu)")

    args = parser.parse_args()

    main(args)
