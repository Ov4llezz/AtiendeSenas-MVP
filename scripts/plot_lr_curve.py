"""
Script para extraer y visualizar la curva de Learning Rate desde logs de TensorBoard
"""
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse


def extract_lr_from_tensorboard(logdir):
    """
    Extrae el historial de Learning Rate desde los logs de TensorBoard

    Args:
        logdir: Directorio con los logs de TensorBoard

    Returns:
        steps: Lista de steps
        lr_values: Lista de valores de LR
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # Obtener tags disponibles
    print(f"\nTags disponibles en {logdir}:")
    print(f"Scalars: {ea.Tags()['scalars']}")

    # Buscar el tag del learning rate (puede variar según cómo se registró)
    lr_tags = [tag for tag in ea.Tags()['scalars'] if 'lr' in tag.lower() or 'learning_rate' in tag.lower()]

    if not lr_tags:
        print(f"No se encontró tag de Learning Rate en {logdir}")
        return None, None

    lr_tag = lr_tags[0]
    print(f"Usando tag: {lr_tag}")

    # Extraer valores
    lr_events = ea.Scalars(lr_tag)
    steps = [event.step for event in lr_events]
    lr_values = [event.value for event in lr_events]

    return steps, lr_values


def plot_lr_curves(runs_dir, output_file='lr_curves.png'):
    """
    Grafica las curvas de LR de todos los runs disponibles
    """
    plt.figure(figsize=(12, 6))

    # Buscar todos los subdirectorios en runs/
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    run_dirs.sort()

    colors = plt.cm.tab10(range(len(run_dirs)))

    for idx, run_name in enumerate(run_dirs):
        run_path = os.path.join(runs_dir, run_name)
        print(f"\nProcesando: {run_name}")

        steps, lr_values = extract_lr_from_tensorboard(run_path)

        if steps and lr_values:
            plt.plot(steps, lr_values, label=run_name, color=colors[idx], linewidth=2, marker='o', markersize=3)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule During Training', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    plt.tight_layout()

    # Guardar gráfico
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico guardado en: {output_file}")
    plt.close()


def plot_single_run(run_path, output_file='lr_curve_single.png'):
    """
    Grafica la curva de LR de un run específico
    """
    print(f"Procesando run: {run_path}")

    steps, lr_values = extract_lr_from_tensorboard(run_path)

    if not steps or not lr_values:
        print("No se pudo extraer información de LR")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, lr_values, linewidth=2, marker='o', markersize=4, color='#2E86AB')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico guardado en: {output_file}")
    plt.close()

    # Imprimir estadísticas
    print(f"\nEstadisticas del Learning Rate:")
    print(f"   - LR inicial: {lr_values[0]:.2e}")
    print(f"   - LR final: {lr_values[-1]:.2e}")
    print(f"   - LR mínimo: {min(lr_values):.2e}")
    print(f"   - LR máximo: {max(lr_values):.2e}")
    print(f"   - Total de steps: {len(steps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extraer y graficar curvas de Learning Rate')
    parser.add_argument('--runs_dir', type=str, default='./runs', help='Directorio con los runs de TensorBoard')
    parser.add_argument('--run_name', type=str, help='Nombre de un run específico (opcional)')
    parser.add_argument('--output', type=str, default='lr_curves.png', help='Archivo de salida')

    args = parser.parse_args()

    if args.run_name:
        # Graficar un run específico
        run_path = os.path.join(args.runs_dir, args.run_name)
        if os.path.exists(run_path):
            plot_single_run(run_path, args.output)
        else:
            print(f"[ERROR] No se encontro el run: {run_path}")
    else:
        # Graficar todos los runs
        if os.path.exists(args.runs_dir):
            plot_lr_curves(args.runs_dir, args.output)
        else:
            print(f"[ERROR] No se encontro el directorio: {args.runs_dir}")
