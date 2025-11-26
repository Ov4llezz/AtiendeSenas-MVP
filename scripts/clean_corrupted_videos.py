"""
clean_corrupted_videos.py - Detecta y elimina videos corruptos de WLASL100

Este script:
1. Escanea todos los videos en train/val/test
2. Identifica videos corruptos (0 frames o no legibles)
3. Elimina los archivos corruptos
4. Actualiza los archivos de splits
5. Genera reporte detallado

Autor: Rafael Ovalle - Tesis UNAB
"""

import os
import cv2
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple
from tqdm import tqdm


# ============================================================
#   CONFIGURACIÓN
# ============================================================
BASE_PATH = "data/wlasl100"
DATASET_DIR = os.path.join(BASE_PATH, "dataset")
SPLITS_DIR = os.path.join(BASE_PATH, "splits")

SPLITS = ["train", "val", "test"]


# ============================================================
#   VERIFICAR SI UN VIDEO ESTÁ CORRUPTO
# ============================================================
def is_video_corrupted(video_path: str) -> Tuple[bool, str]:
    """
    Verifica si un video está corrupto.

    Returns:
        (is_corrupted, reason): Tupla con booleano y razón
    """
    if not os.path.exists(video_path):
        return True, "FILE_NOT_FOUND"

    try:
        cap = cv2.VideoCapture(video_path)

        # Verificar que se pudo abrir
        if not cap.isOpened():
            cap.release()
            return True, "CANNOT_OPEN"

        # Verificar frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return True, "ZERO_FRAMES"

        # Intentar leer el primer frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return True, "CANNOT_READ_FRAME"

        cap.release()
        return False, "OK"

    except Exception as e:
        return True, f"EXCEPTION: {str(e)}"


# ============================================================
#   ESCANEAR VIDEOS DE UN SPLIT
# ============================================================
def scan_split(split: str) -> dict:
    """
    Escanea todos los videos de un split y detecta corruptos.

    Returns:
        dict con estadísticas y listas de videos
    """
    print(f"\n{'='*60}")
    print(f"ESCANEANDO SPLIT: {split.upper()}")
    print(f"{'='*60}")

    # Rutas
    videos_dir = os.path.join(DATASET_DIR, split)
    split_file = os.path.join(SPLITS_DIR, f"{split}_split.txt")

    # Leer lista de videos del split
    with open(split_file, "r", encoding="utf-8") as f:
        video_list = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Videos en {split}_split.txt: {len(video_list)}")

    # Escanear videos
    corrupted_videos = []
    valid_videos = []
    missing_videos = []
    corruption_reasons = {}

    for video_entry in tqdm(video_list, desc=f"Verificando {split}"):
        # Normalizar nombre de archivo
        filename = os.path.basename(video_entry.replace("\\", "/"))
        video_path = os.path.join(videos_dir, filename)

        # Verificar si existe
        if not os.path.exists(video_path):
            missing_videos.append(filename)
            continue

        # Verificar si está corrupto
        is_corrupted, reason = is_video_corrupted(video_path)

        if is_corrupted:
            corrupted_videos.append(filename)
            corruption_reasons[filename] = reason
        else:
            valid_videos.append(filename)

    # Estadísticas
    stats = {
        "split": split,
        "total_in_split_file": len(video_list),
        "valid": len(valid_videos),
        "corrupted": len(corrupted_videos),
        "missing": len(missing_videos),
        "valid_videos": valid_videos,
        "corrupted_videos": corrupted_videos,
        "missing_videos": missing_videos,
        "corruption_reasons": corruption_reasons
    }

    # Imprimir resumen
    print(f"\n[RESUMEN {split.upper()}]")
    print(f"  Total en split file: {stats['total_in_split_file']}")
    print(f"  [OK] Videos validos:   {stats['valid']}")
    print(f"  [X] Videos corruptos: {stats['corrupted']}")
    print(f"  [?] Videos faltantes: {stats['missing']}")

    return stats


# ============================================================
#   ELIMINAR VIDEOS CORRUPTOS
# ============================================================
def delete_corrupted_videos(stats: dict, dry_run: bool = False) -> int:
    """
    Elimina videos corruptos del sistema de archivos.

    Args:
        stats: Diccionario con estadísticas del split
        dry_run: Si es True, solo simula la eliminación

    Returns:
        Número de archivos eliminados
    """
    split = stats["split"]
    videos_dir = os.path.join(DATASET_DIR, split)
    corrupted_videos = stats["corrupted_videos"]

    deleted_count = 0

    if len(corrupted_videos) == 0:
        print(f"[INFO] No hay videos corruptos para eliminar en {split}")
        return 0

    print(f"\n{'='*60}")
    print(f"{'SIMULANDO' if dry_run else 'ELIMINANDO'} VIDEOS CORRUPTOS - {split.upper()}")
    print(f"{'='*60}")

    for filename in tqdm(corrupted_videos, desc=f"Eliminando {split}"):
        video_path = os.path.join(videos_dir, filename)

        if os.path.exists(video_path):
            if not dry_run:
                try:
                    os.remove(video_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"[ERROR] No se pudo eliminar {filename}: {e}")
            else:
                print(f"[DRY RUN] Eliminaría: {filename}")
                deleted_count += 1

    print(f"[INFO] {'Simulados' if dry_run else 'Eliminados'}: {deleted_count} archivos")

    return deleted_count


# ============================================================
#   ACTUALIZAR ARCHIVO DE SPLIT
# ============================================================
def update_split_file(stats: dict, dry_run: bool = False):
    """
    Actualiza el archivo de split para excluir videos corruptos y faltantes.
    """
    split = stats["split"]
    split_file = os.path.join(SPLITS_DIR, f"{split}_split.txt")

    # Crear nuevo contenido del split (solo videos válidos)
    valid_entries = [f"{split}\\{filename}" for filename in stats["valid_videos"]]

    if dry_run:
        print(f"\n[DRY RUN] Actualizaría {split}_split.txt:")
        print(f"  Entradas originales: {stats['total_in_split_file']}")
        print(f"  Nuevas entradas: {len(valid_entries)}")
        return

    # Backup del archivo original
    backup_file = split_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(split_file):
        import shutil
        shutil.copy(split_file, backup_file)
        print(f"[BACKUP] {split}_split.txt -> {os.path.basename(backup_file)}")

    # Escribir nuevo archivo de split
    with open(split_file, "w", encoding="utf-8") as f:
        for entry in valid_entries:
            f.write(entry + "\n")

    print(f"[UPDATE] {split}_split.txt actualizado: {len(valid_entries)} entradas")


# ============================================================
#   GENERAR REPORTE DETALLADO
# ============================================================
def generate_report(all_stats: list, output_dir: str = "data/wlasl100"):
    """Genera reporte detallado en JSON y TXT"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reporte JSON
    json_report = {
        "timestamp": timestamp,
        "splits": all_stats,
        "summary": {
            "total_valid": sum(s["valid"] for s in all_stats),
            "total_corrupted": sum(s["corrupted"] for s in all_stats),
            "total_missing": sum(s["missing"] for s in all_stats),
        }
    }

    json_path = os.path.join(output_dir, f"cleanup_report_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    print(f"\n[REPORT] JSON guardado en: {json_path}")

    # Reporte TXT
    txt_path = os.path.join(output_dir, f"cleanup_report_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("REPORTE DE LIMPIEZA DE VIDEOS CORRUPTOS\n")
        f.write("="*60 + "\n")
        f.write(f"Fecha: {timestamp}\n\n")

        for stats in all_stats:
            f.write(f"\n{'='*60}\n")
            f.write(f"SPLIT: {stats['split'].upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total en split file: {stats['total_in_split_file']}\n")
            f.write(f"Videos válidos:      {stats['valid']}\n")
            f.write(f"Videos corruptos:    {stats['corrupted']}\n")
            f.write(f"Videos faltantes:    {stats['missing']}\n\n")

            if stats['corrupted_videos']:
                f.write(f"VIDEOS CORRUPTOS ({len(stats['corrupted_videos'])}):\n")
                f.write("-" * 60 + "\n")
                for video in stats['corrupted_videos']:
                    reason = stats['corruption_reasons'].get(video, "UNKNOWN")
                    f.write(f"  - {video} ({reason})\n")
                f.write("\n")

            if stats['missing_videos']:
                f.write(f"VIDEOS FALTANTES ({len(stats['missing_videos'])}):\n")
                f.write("-" * 60 + "\n")
                for video in stats['missing_videos']:
                    f.write(f"  - {video}\n")
                f.write("\n")

        f.write("\n" + "="*60 + "\n")
        f.write("RESUMEN GLOBAL\n")
        f.write("="*60 + "\n")
        f.write(f"Total videos válidos:   {json_report['summary']['total_valid']}\n")
        f.write(f"Total videos corruptos: {json_report['summary']['total_corrupted']}\n")
        f.write(f"Total videos faltantes: {json_report['summary']['total_missing']}\n")

    print(f"[REPORT] TXT guardado en: {txt_path}")

    return json_path, txt_path


# ============================================================
#   FUNCIÓN PRINCIPAL
# ============================================================
def main(dry_run: bool = True, delete_files: bool = True, update_splits: bool = True):
    """
    Función principal de limpieza.

    Args:
        dry_run: Si es True, solo simula sin hacer cambios
        delete_files: Si es True, elimina archivos corruptos
        update_splits: Si es True, actualiza archivos de splits
    """
    print("\n" + "="*60)
    print("LIMPIEZA DE VIDEOS CORRUPTOS - WLASL100")
    print("="*60)
    print(f"Modo: {'DRY RUN (SIMULACIÓN)' if dry_run else 'EJECUCIÓN REAL'}")
    print(f"Eliminar archivos: {delete_files}")
    print(f"Actualizar splits: {update_splits}")
    print("="*60)

    # Escanear todos los splits
    all_stats = []
    for split in SPLITS:
        stats = scan_split(split)
        all_stats.append(stats)

    # Resumen global
    print(f"\n{'='*60}")
    print("RESUMEN GLOBAL")
    print(f"{'='*60}")
    total_valid = sum(s["valid"] for s in all_stats)
    total_corrupted = sum(s["corrupted"] for s in all_stats)
    total_missing = sum(s["missing"] for s in all_stats)

    print(f"[OK] Total videos validos:   {total_valid}")
    print(f"[X] Total videos corruptos: {total_corrupted}")
    print(f"[?] Total videos faltantes: {total_missing}")
    print(f"{'='*60}\n")

    # Eliminar videos corruptos
    if delete_files and total_corrupted > 0:
        for stats in all_stats:
            delete_corrupted_videos(stats, dry_run=dry_run)

    # Actualizar archivos de splits
    if update_splits:
        print(f"\n{'='*60}")
        print(f"{'SIMULANDO' if dry_run else 'ACTUALIZANDO'} ARCHIVOS DE SPLITS")
        print(f"{'='*60}")
        for stats in all_stats:
            update_split_file(stats, dry_run=dry_run)

    # Generar reporte
    if not dry_run:
        print(f"\n{'='*60}")
        print("GENERANDO REPORTES")
        print(f"{'='*60}")
        generate_report(all_stats)

    # Mensaje final
    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO")
    print(f"{'='*60}")

    if dry_run:
        print("\n[!] MODO DRY RUN - NO SE REALIZARON CAMBIOS")
        print("Para ejecutar la limpieza real, ejecuta:")
        print("  python scripts/clean_corrupted_videos.py --execute")
    else:
        print("\n[OK] LIMPIEZA COMPLETADA")
        print(f"   - Videos eliminados: {total_corrupted}")
        print(f"   - Splits actualizados: {len(SPLITS)}")
        print(f"   - Reportes generados en: {BASE_PATH}/")


# ============================================================
#   EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detecta y elimina videos corruptos de WLASL100")
    parser.add_argument("--execute", action="store_true",
                        help="Ejecutar limpieza real (por defecto es dry run)")
    parser.add_argument("--no-delete", action="store_true",
                        help="No eliminar archivos, solo actualizar splits")
    parser.add_argument("--no-update-splits", action="store_true",
                        help="No actualizar archivos de splits")

    args = parser.parse_args()

    main(
        dry_run=not args.execute,
        delete_files=not args.no_delete,
        update_splits=not args.no_update_splits
    )
