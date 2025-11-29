"""
Script para verificar archivos de video corruptos en WLASL300.
Intenta abrir cada video y verificar que se pueda leer al menos un frame.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

def check_video_integrity(video_path):
    """
    Verifica si un video puede abrirse y leer frames.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return False, "No se puede abrir el video"

        # Intentar leer el primer frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, "No se puede leer frames"

        # Verificar propiedades básicas
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        if frame_count == 0:
            return False, "Video sin frames"

        if fps == 0:
            return False, "FPS inválido"

        if width == 0 or height == 0:
            return False, f"Resolución inválida ({width}x{height})"

        return True, f"OK ({frame_count} frames, {fps:.1f} FPS, {width}x{height})"

    except Exception as e:
        return False, f"Error: {str(e)}"

def check_dataset_videos(base_path, split="train"):
    """
    Verifica todos los videos en un split del dataset.
    """
    videos_dir = Path(base_path) / "dataset" / split

    if not videos_dir.exists():
        print(f"[X] Directorio no encontrado: {videos_dir}")
        return

    # Obtener todos los archivos .mp4
    video_files = list(videos_dir.glob("*.mp4"))

    print(f"\n{'='*80}")
    print(f"Verificando videos en: {videos_dir}")
    print(f"Total de archivos .mp4: {len(video_files)}")
    print(f"{'='*80}\n")

    corrupted = []
    valid = []

    for video_path in tqdm(video_files, desc="Verificando videos"):
        is_valid, message = check_video_integrity(video_path)

        if is_valid:
            valid.append((video_path.name, message))
        else:
            corrupted.append((video_path.name, message))
            print(f"\n[X] CORRUPTO: {video_path.name} - {message}")

    # Resumen
    print(f"\n{'='*80}")
    print(f"RESUMEN DE VERIFICACION - {split.upper()}")
    print(f"{'='*80}")
    print(f"Total verificados:  {len(video_files)}")
    print(f"Videos validos:     {len(valid)} ({len(valid)/len(video_files)*100:.1f}%)")
    print(f"Videos corruptos:   {len(corrupted)} ({len(corrupted)/len(video_files)*100:.1f}%)")
    print(f"{'='*80}\n")

    if corrupted:
        print("\nARCHIVOS CORRUPTOS DETECTADOS:")
        print("-" * 80)
        for filename, error in corrupted:
            print(f"  - {filename}: {error}")
        print("-" * 80)

        # Guardar lista de corruptos
        output_file = f"corrupted_videos_{split}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Videos corruptos en {split}:\n\n")
            for filename, error in corrupted:
                f.write(f"{filename} - {error}\n")

        print(f"\n[OK] Lista guardada en: {output_file}")
    else:
        print("\n[OK] Todos los videos estan en buen estado!")

    return valid, corrupted

if __name__ == "__main__":
    # Verificar WLASL300 train
    print("\n" + "="*80)
    print("VERIFICADOR DE VIDEOS CORRUPTOS - WLASL300")
    print("="*80)

    base_path = "data/wlasl300"

    # Verificar train
    valid_train, corrupted_train = check_dataset_videos(base_path, "train")

    print("\nVerificando validation...")
    valid_val, corrupted_val = check_dataset_videos(base_path, "val")

    print("\nVerificando test...")
    valid_test, corrupted_test = check_dataset_videos(base_path, "test")

    # Resumen final
    total_corrupted = len(corrupted_train) + len(corrupted_val) + len(corrupted_test)
    total_videos = len(valid_train) + len(corrupted_train) + len(valid_val) + len(corrupted_val) + len(valid_test) + len(corrupted_test)

    print("\n" + "="*80)
    print("RESUMEN FINAL - WLASL300 COMPLETO")
    print("="*80)
    print(f"Total videos verificados: {total_videos}")
    print(f"Videos corruptos totales: {total_corrupted}")
    print(f"Train corruptos:          {len(corrupted_train)}")
    print(f"Val corruptos:            {len(corrupted_val)}")
    print(f"Test corruptos:           {len(corrupted_test)}")
    print("="*80)
