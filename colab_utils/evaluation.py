"""
Evaluation module for VideoMAE WLASL
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


@torch.no_grad()
def evaluate_detailed(model, dataloader, device, num_classes):
    """
    Detailed evaluation with all metrics.

    Returns:
        Dictionary with predictions, labels, logits and metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_logits = []

    progress_bar = tqdm(dataloader, desc="Evaluación Detallada")

    for videos, labels in progress_bar:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=videos)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu())

    # Concatenate
    all_logits = torch.cat(all_logits, dim=0)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # ===== OVERALL METRICS =====
    total_acc = accuracy_score(all_labels, all_predictions) * 100.0

    # Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )

    # ===== TOP-K ACCURACY =====
    logits_np = all_logits.numpy()
    top_k_accuracies = {}

    for k in [1, 3, 5]:
        if k <= num_classes:
            top_k_preds = np.argsort(logits_np, axis=1)[:, -k:]
            correct = np.array([label in preds for label, preds in zip(all_labels, top_k_preds)])
            top_k_acc = correct.sum() / len(all_labels) * 100.0
            top_k_accuracies[f'top_{k}'] = top_k_acc

    # ===== PER-CLASS METRICS =====
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0, labels=range(num_classes)
    )

    # Per-class accuracy
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    class_accuracies = []
    for i in range(num_classes):
        if support_per_class[i] > 0:
            class_acc = conf_matrix[i, i] / support_per_class[i] * 100.0
        else:
            class_acc = 0.0
        class_accuracies.append(class_acc)

    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': logits_np,
        'confusion_matrix': conf_matrix,
        'metrics': {
            'total_accuracy': total_acc,
            'precision_macro': precision_macro * 100.0,
            'recall_macro': recall_macro * 100.0,
            'f1_macro': f1_macro * 100.0,
            'precision_weighted': precision_weighted * 100.0,
            'recall_weighted': recall_weighted * 100.0,
            'f1_weighted': f1_weighted * 100.0,
            'top_k': top_k_accuracies,
        },
        'per_class': {
            'accuracy': class_accuracies,
            'precision': (precision_per_class * 100.0).tolist(),
            'recall': (recall_per_class * 100.0).tolist(),
            'f1': (f1_per_class * 100.0).tolist(),
            'support': support_per_class.tolist()
        }
    }


def print_results(results):
    """Print evaluation results in a nice format."""
    metrics = results['metrics']

    print(f"\\n{'='*70}")
    print(f"{'RESULTADOS GENERALES':^70}")
    print(f"{'='*70}")
    print(f"Total Accuracy:       {metrics['total_accuracy']:.2f}%")
    print(f"\\nPrecision (Macro):    {metrics['precision_macro']:.2f}%")
    print(f"Recall (Macro):       {metrics['recall_macro']:.2f}%")
    print(f"F1-Score (Macro):     {metrics['f1_macro']:.2f}%")
    print(f"\\nPrecision (Weighted): {metrics['precision_weighted']:.2f}%")
    print(f"Recall (Weighted):    {metrics['recall_weighted']:.2f}%")
    print(f"F1-Score (Weighted):  {metrics['f1_weighted']:.2f}%")
    print(f"\\n{'='*70}")
    print(f"{'TOP-K ACCURACIES':^70}")
    print(f"{'='*70}")
    for k, acc in metrics['top_k'].items():
        print(f"{k.upper().replace('_', '-')}: {acc:.2f}%")
    print(f"{'='*70}\\n")


def print_top_classes(results, top_n=10):
    """Print top best and worst classes."""
    per_class = results['per_class']
    class_accs = np.array(per_class['accuracy'])
    class_support = np.array(per_class['support'])

    # Filter valid classes
    valid_classes = class_support > 0
    class_accs_valid = class_accs[valid_classes]
    class_ids_valid = np.arange(len(class_accs))[valid_classes]

    # Top and bottom
    sorted_indices = np.argsort(class_accs_valid)
    top_indices = sorted_indices[-top_n:][::-1]
    bottom_indices = sorted_indices[:top_n]

    # Print best
    print(f"{'='*80}")
    print(f"{'TOP 10 MEJORES CLASES':^80}")
    print(f"{'='*80}")
    print(f"{'Clase':<8} {'Acc(%)':<10} {'Prec(%)':<10} {'Rec(%)':<10} {'F1(%)':<10} {'Support':<10}")
    print("-" * 80)

    for idx in top_indices:
        class_id = class_ids_valid[idx]
        print(f"{class_id:<8} "
              f"{per_class['accuracy'][class_id]:<10.2f} "
              f"{per_class['precision'][class_id]:<10.2f} "
              f"{per_class['recall'][class_id]:<10.2f} "
              f"{per_class['f1'][class_id]:<10.2f} "
              f"{int(per_class['support'][class_id]):<10}")

    # Print worst
    print(f"\\n{'='*80}")
    print(f"{'TOP 10 PEORES CLASES':^80}")
    print(f"{'='*80}")
    print(f"{'Clase':<8} {'Acc(%)':<10} {'Prec(%)':<10} {'Rec(%)':<10} {'F1(%)':<10} {'Support':<10}")
    print("-" * 80)

    for idx in bottom_indices:
        class_id = class_ids_valid[idx]
        print(f"{class_id:<8} "
              f"{per_class['accuracy'][class_id]:<10.2f} "
              f"{per_class['precision'][class_id]:<10.2f} "
              f"{per_class['recall'][class_id]:<10.2f} "
              f"{per_class['f1'][class_id]:<10.2f} "
              f"{int(per_class['support'][class_id]):<10}")


def save_results(results, config, checkpoint_info, output_dir):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = results['metrics']

    # Prepare export data
    export_data = {
        'timestamp': timestamp,
        'configuration': config,
        'training': checkpoint_info,
        'test_metrics': {
            'total_accuracy': float(metrics['total_accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_weighted': float(metrics['recall_weighted']),
            'f1_weighted': float(metrics['f1_weighted']),
            'top_k_accuracies': {k: float(v) for k, v in metrics['top_k'].items()},
        },
        'per_class_metrics': results['per_class'],
    }

    # Save JSON
    json_path = f"{output_dir}/complete_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✅ JSON guardado: {json_path}")

    # Save TXT report
    txt_path = f"{output_dir}/report_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\\n")
        f.write(f"{'REPORTE COMPLETO - VIDEOMAE WLASL':^80}\\n")
        f.write("="*80 + "\\n\\n")

        f.write(f"Fecha: {timestamp}\\n")
        f.write(f"Dataset: {config.get('dataset_type', 'N/A').upper()}\\n")
        f.write(f"Versión: {config.get('version', 'N/A').upper()}\\n\\n")

        f.write("="*80 + "\\n")
        f.write("RESULTADOS EN TEST SET\\n")
        f.write("="*80 + "\\n\\n")
        f.write(f"Total Accuracy:       {metrics['total_accuracy']:.2f}%\\n\\n")
        f.write(f"Precision (Macro):    {metrics['precision_macro']:.2f}%\\n")
        f.write(f"Recall (Macro):       {metrics['recall_macro']:.2f}%\\n")
        f.write(f"F1-Score (Macro):     {metrics['f1_macro']:.2f}%\\n\\n")
        f.write(f"Precision (Weighted): {metrics['precision_weighted']:.2f}%\\n")
        f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.2f}%\\n")
        f.write(f"F1-Score (Weighted):  {metrics['f1_weighted']:.2f}%\\n\\n")

        f.write("="*80 + "\\n")
        f.write("TOP-K ACCURACIES\\n")
        f.write("="*80 + "\\n\\n")
        for k, acc in metrics['top_k'].items():
            f.write(f"{k.upper().replace('_', '-')}: {acc:.2f}%\\n")

    print(f"✅ TXT guardado: {txt_path}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'correct': results['labels'] == results['predictions']
    })
    pred_path = f"{output_dir}/predictions_{timestamp}.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"✅ Predicciones guardadas: {pred_path}")

    return json_path, txt_path, pred_path, timestamp
