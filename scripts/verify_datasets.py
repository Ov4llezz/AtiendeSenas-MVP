"""
Script de verificación para confirmar que ambos datasets (WLASL100 y WLASL300)
están correctamente configurados y son compatibles con el pipeline.
"""

import os
import json
import sys

def check_dataset_structure(base_path, dataset_name, expected_num_classes):
    """Verifica la estructura de un dataset"""
    print(f"\n{'='*70}")
    print(f"Verificando {dataset_name}")
    print(f"{'='*70}")

    issues = []

    # Verificar que existe el directorio base
    if not os.path.exists(base_path):
        issues.append(f"[X] Directorio base no existe: {base_path}")
        return issues
    else:
        print(f"[OK] Directorio base encontrado: {base_path}")

    # Verificar estructura de carpetas
    required_dirs = [
        "dataset/train",
        "dataset/val",
        "dataset/test",
        "splits"
    ]

    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if os.path.exists(full_path):
            # Contar videos
            if "dataset" in dir_path:
                split_name = os.path.basename(dir_path)
                videos = [f for f in os.listdir(full_path) if f.endswith('.mp4')]
                print(f"[OK] {dir_path}: {len(videos)} videos")
        else:
            issues.append(f"[X] Directorio faltante: {full_path}")

    # Verificar archivos JSON
    if dataset_name == "WLASL100":
        json_files = ["nslt_100.json", "WLASL_v0.3.json"]
    else:  # WLASL300
        json_files = ["nslt_300.json", "WLASL_v0.3_300.json", "gloss_to_id.json"]

    for json_file in json_files:
        json_path = os.path.join(base_path, json_file)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if json_file.startswith("nslt"):
                    num_videos = len(data)
                    print(f"[OK] {json_file}: {num_videos} videos mapeados")
                elif json_file == "gloss_to_id.json":
                    num_glosses = len(data)
                    print(f"[OK] {json_file}: {num_glosses} glosas")
                    if num_glosses != expected_num_classes:
                        issues.append(f"[X] Número de glosas ({num_glosses}) no coincide con esperado ({expected_num_classes})")
                else:
                    print(f"[OK] {json_file}: válido")
            except Exception as e:
                issues.append(f"[X] Error leyendo {json_file}: {e}")
        else:
            issues.append(f"[X] Archivo JSON faltante: {json_path}")

    # Verificar archivos de splits
    split_files = ["train_split.txt", "val_split.txt", "test_split.txt"]
    for split_file in split_files:
        split_path = os.path.join(base_path, "splits", split_file)
        if os.path.exists(split_path):
            with open(split_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"[OK] {split_file}: {len(lines)} entradas")
        else:
            issues.append(f"[X] Archivo de split faltante: {split_path}")

    return issues


def verify_dataset_loading(base_path, dataset_size):
    """Intenta cargar un dataset usando WLASLDataset"""
    print(f"\n{'='*70}")
    print(f"Verificando carga del dataset...")
    print(f"{'='*70}")

    try:
        # Cambiar al directorio de scripts para importar
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from WLASLDataset import WLASLVideoDataset

        # Intentar cargar train dataset
        print(f"Intentando cargar train dataset desde {base_path}...")
        train_dataset = WLASLVideoDataset(
            split="train",
            base_path=base_path,
            dataset_size=dataset_size
        )

        print(f"[OK] Dataset cargado exitosamente")
        print(f"  - Muestras: {len(train_dataset)}")
        print(f"  - Clases: {dataset_size}")

        # Intentar obtener una muestra
        print(f"\nIntentando leer una muestra...")
        try:
            video, label = train_dataset[0]
            print(f"[OK] Muestra leída exitosamente")
            print(f"  - Video shape: {video.shape}")
            print(f"  - Label: {label.item()}")
            return []
        except Exception as e:
            return [f"[X] Error leyendo muestra: {e}"]

    except Exception as e:
        return [f"[X] Error cargando dataset: {e}"]


def main():
    """Función principal"""
    print("\n" + "="*70)
    print(f"{'VERIFICACIÓN DE DATASETS WLASL100 Y WLASL300':^70}")
    print("="*70)

    all_issues = []

    # Verificar WLASL100
    issues_100 = check_dataset_structure("data/wlasl100", "WLASL100", 100)
    all_issues.extend([(f"WLASL100", issue) for issue in issues_100])

    if not issues_100:
        loading_issues = verify_dataset_loading("data/wlasl100", 100)
        all_issues.extend([("WLASL100", issue) for issue in loading_issues])

    # Verificar WLASL300
    issues_300 = check_dataset_structure("data/wlasl300", "WLASL300", 300)
    all_issues.extend([("WLASL300", issue) for issue in issues_300])

    if not issues_300:
        loading_issues = verify_dataset_loading("data/wlasl300", 300)
        all_issues.extend([("WLASL300", issue) for issue in loading_issues])

    # Resumen final
    print(f"\n{'='*70}")
    print(f"{'RESUMEN DE VERIFICACIÓN':^70}")
    print(f"{'='*70}")

    if not all_issues:
        print("\n[OK] ¡TODOS LOS CHECKS PASARON EXITOSAMENTE!")
        print("[OK] Ambos datasets están correctamente configurados")
        print("[OK] El pipeline está listo para usar")
    else:
        print("\n[X] SE ENCONTRARON PROBLEMAS:\n")
        for dataset, issue in all_issues:
            print(f"  [{dataset}] {issue}")
        print(f"\nTotal de problemas encontrados: {len(all_issues)}")

    print("="*70)


if __name__ == "__main__":
    main()
