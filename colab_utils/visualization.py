"""
Visualization module for VideoMAE WLASL
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history_df, save_path=None):
    """Plot training curves (Loss, Accuracy, LR)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history_df['epoch'], history_df['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history_df['epoch'], history_df['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(history_df['epoch'], history_df['lr'], label='Learning Rate', marker='o', color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Curvas guardadas: {save_path}")

    plt.show()


def plot_confusion_matrix(conf_matrix, top_n=30, save_path=None):
    """Plot confusion matrix heatmap."""
    num_classes = conf_matrix.shape[0]

    # If too many classes, show only top N
    if num_classes > top_n:
        class_totals = conf_matrix.sum(axis=1)
        top_classes = np.argsort(class_totals)[-top_n:][::-1]
        conf_matrix_subset = conf_matrix[np.ix_(top_classes, top_classes)]
        title = f"Matriz de Confusión (Top {top_n} clases más frecuentes)"
    else:
        conf_matrix_subset = conf_matrix
        top_classes = range(num_classes)
        title = "Matriz de Confusión (Todas las clases)"

    # Normalize by row
    conf_matrix_norm = conf_matrix_subset.astype('float') / conf_matrix_subset.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)

    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        conf_matrix_norm,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=top_classes,
        yticklabels=top_classes,
        cbar_kws={'label': 'Proporción'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.ylabel('Clase Real', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz guardada: {save_path}")

    plt.show()


def plot_class_performance(per_class_metrics, top_n=20, save_path=None):
    """Plot best and worst performing classes."""
    class_accs = np.array(per_class_metrics['accuracy'])
    class_support = np.array(per_class_metrics['support'])

    # Filter valid classes
    valid_classes = class_support > 0
    class_accs_valid = class_accs[valid_classes]
    class_ids_valid = np.arange(len(class_accs))[valid_classes]

    # Top and bottom N
    sorted_indices = np.argsort(class_accs_valid)
    bottom_indices = sorted_indices[:top_n]
    top_indices = sorted_indices[-top_n:][::-1]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top N best
    top_classes = class_ids_valid[top_indices]
    top_accs = class_accs_valid[top_indices]
    colors_top = ['green' if acc >= 80 else 'yellowgreen' for acc in top_accs]

    axes[0].barh(range(len(top_classes)), top_accs, color=colors_top, alpha=0.7)
    axes[0].set_yticks(range(len(top_classes)))
    axes[0].set_yticklabels([f"Clase {c}" for c in top_classes])
    axes[0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0].set_title(f'Top {len(top_classes)} Mejores Clases', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].set_xlim([0, 105])

    # Bottom N worst
    bottom_classes = class_ids_valid[bottom_indices]
    bottom_accs = class_accs_valid[bottom_indices]
    colors_bottom = ['red' if acc < 50 else 'orange' for acc in bottom_accs]

    axes[1].barh(range(len(bottom_classes)), bottom_accs, color=colors_bottom, alpha=0.7)
    axes[1].set_yticks(range(len(bottom_classes)))
    axes[1].set_yticklabels([f"Clase {c}" for c in bottom_classes])
    axes[1].set_xlabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Top {len(bottom_classes)} Peores Clases', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_xlim([0, 105])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance guardado: {save_path}")

    plt.show()


def plot_accuracy_distribution(per_class_metrics, save_path=None):
    """Plot accuracy distribution histogram."""
    class_accs = np.array(per_class_metrics['accuracy'])
    class_support = np.array(per_class_metrics['support'])

    valid_accs = class_accs[class_support > 0]

    plt.figure(figsize=(12, 6))
    plt.hist(valid_accs, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(valid_accs.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {valid_accs.mean():.2f}%')
    plt.axvline(np.median(valid_accs), color='green', linestyle='--', linewidth=2,
                label=f'Mediana: {np.median(valid_accs):.2f}%')

    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.ylabel('Número de Clases', fontsize=12)
    plt.title('Distribución de Accuracy por Clase', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribución guardada: {save_path}")

    plt.show()


def plot_support_analysis(per_class_metrics, save_path=None):
    """Plot performance vs number of samples."""
    class_accs = np.array(per_class_metrics['accuracy'])
    class_support = np.array(per_class_metrics['support'])

    # Create support bins
    support_bins = [0, 5, 10, 20, 50, 100, 1000]
    support_labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '100+']

    bin_stats = []
    for i in range(len(support_bins)-1):
        mask = (class_support >= support_bins[i]) & (class_support < support_bins[i+1])
        if i == len(support_bins) - 2:
            mask = class_support >= support_bins[i]

        if mask.sum() > 0:
            avg_acc = class_accs[mask].mean()
            count = mask.sum()
        else:
            avg_acc = 0
            count = 0

        bin_stats.append({
            'range': support_labels[i] if i < len(support_labels) else support_labels[-1],
            'avg_accuracy': avg_acc,
            'num_classes': count
        })

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ranges = [s['range'] for s in bin_stats]
    accs = [s['avg_accuracy'] for s in bin_stats]
    counts = [s['num_classes'] for s in bin_stats]

    # Accuracy by bin
    ax1.bar(ranges, accs, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Número de Muestras por Clase', fontsize=12)
    ax1.set_ylabel('Accuracy Promedio (%)', fontsize=12)
    ax1.set_title('Accuracy según Número de Muestras', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    # Classes by bin
    ax2.bar(ranges, counts, color='coral', alpha=0.7)
    ax2.set_xlabel('Número de Muestras por Clase', fontsize=12)
    ax2.set_ylabel('Número de Clases', fontsize=12)
    ax2.set_title('Distribución de Clases', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Análisis guardado: {save_path}")

    plt.show()


def visualize_all_results(results, history_df, output_dir, timestamp):
    """Generate all visualizations."""
    print("\n📊 Generando visualizaciones...\n")

    # Training curves
    curves_path = f"{output_dir}/training_curves_{timestamp}.png"
    plot_training_curves(history_df, curves_path)

    # Confusion matrix
    cm_path = f"{output_dir}/confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(results['confusion_matrix'], top_n=30, save_path=cm_path)

    # Class performance
    perf_path = f"{output_dir}/class_performance_{timestamp}.png"
    plot_class_performance(results['per_class'], top_n=20, save_path=perf_path)

    # Accuracy distribution
    dist_path = f"{output_dir}/accuracy_distribution_{timestamp}.png"
    plot_accuracy_distribution(results['per_class'], save_path=dist_path)

    # Support analysis
    support_path = f"{output_dir}/support_analysis_{timestamp}.png"
    plot_support_analysis(results['per_class'], save_path=support_path)

    print("\n✅ Todas las visualizaciones generadas\n")

    return {
        'training_curves': curves_path,
        'confusion_matrix': cm_path,
        'class_performance': perf_path,
        'accuracy_distribution': dist_path,
        'support_analysis': support_path,
    }
