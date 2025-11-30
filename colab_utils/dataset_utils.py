"""
Dataset verification utilities for WLASL100/300 (V1 and V2)
"""

import os
from pathlib import Path


def verify_dataset(dataset_path, dataset_name):
    """
    Verifica la estructura de un dataset individual.

    Args:
        dataset_path: Ruta al dataset
        dataset_name: Nombre del dataset (ej: "wlasl100", "wlasl100_v2")

    Returns:
        dict con información del dataset
    """
    info = {
        'name': dataset_name,
        'path': dataset_path,
        'exists': False,
        'splits': {},
        'total_videos': 0,
        'errors': []
    }

    if not os.path.exists(dataset_path):
        info['errors'].append(f"❌ Dataset no encontrado en {dataset_path}")
        return info

    info['exists'] = True

    # Verificar estructura de directorios
    required_dirs = ['videos', 'splits']
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            info['errors'].append(f"❌ Directorio '{dir_name}' no encontrado")

    # Verificar splits
    splits_dir = os.path.join(dataset_path, 'splits')
    if os.path.exists(splits_dir):
        # Determinar qué splits esperar según la versión
        is_v2 = '_v2' in dataset_name

        if is_v2:
            # V2: train (combinado), test (usado como val)
            expected_splits = ['train_split.txt', 'test_split.txt']
        else:
            # V1: train, val, test separados
            expected_splits = ['train_split.txt', 'val_split.txt', 'test_split.txt']

        for split_file in expected_splits:
            split_path = os.path.join(splits_dir, split_file)
            if os.path.exists(split_path):
                # Contar líneas (videos)
                with open(split_path, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                split_name = split_file.replace('_split.txt', '')
                info['splits'][split_name] = len(lines)
                info['total_videos'] += len(lines)
            else:
                info['errors'].append(f"❌ Split '{split_file}' no encontrado")

    # Verificar directorio de videos
    videos_dir = os.path.join(dataset_path, 'videos')
    if os.path.exists(videos_dir):
        # Contar archivos .mp4
        video_files = list(Path(videos_dir).rglob('*.mp4'))
        actual_videos = len(video_files)

        # Verificar que coincidan con los splits
        if actual_videos != info['total_videos']:
            info['errors'].append(
                f"⚠️ Videos en disco ({actual_videos}) no coinciden con splits ({info['total_videos']})"
            )

    return info


def verify_all_datasets(data_root="/home/ov4lle/AtiendeSenas-MVP/data"):
    """
    Verifica todos los datasets disponibles (V1 y V2).

    Args:
        data_root: Ruta raíz donde están los datasets
    """
    datasets = [
        ("wlasl100", "WLASL100 - V1 Original"),
        ("wlasl300", "WLASL300 - V1 Original"),
        ("wlasl100_v2", "WLASL100 - V2 (Train+Val Combinados)"),
        ("wlasl300_v2", "WLASL300 - V2 (Train+Val Combinados)")
    ]

    print("=" * 80)
    print(f"{'VERIFICACIÓN DE DATASETS':^80}")
    print("=" * 80)
    print(f"Ruta base: {data_root}\n")

    for dataset_name, description in datasets:
        dataset_path = os.path.join(data_root, dataset_name)
        info = verify_dataset(dataset_path, dataset_name)

        print(f"\n{'─' * 80}")
        print(f"📁 {description}")
        print(f"{'─' * 80}")

        if info['exists']:
            print(f"✅ Dataset encontrado")
            print(f"📂 Ruta: {info['path']}")

            if info['splits']:
                print(f"\n📊 Splits:")
                for split_name, count in info['splits'].items():
                    print(f"  • {split_name:10s}: {count:4d} videos")
                print(f"  {'─' * 40}")
                print(f"  • {'TOTAL':10s}: {info['total_videos']:4d} videos")

            if info['errors']:
                print(f"\n⚠️ Advertencias/Errores:")
                for error in info['errors']:
                    print(f"  {error}")
        else:
            print(f"❌ Dataset NO encontrado")
            if info['errors']:
                for error in info['errors']:
                    print(f"  {error}")

    print(f"\n{'=' * 80}")
    print(f"{'VERIFICACIÓN COMPLETADA':^80}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Prueba local
    verify_all_datasets()
