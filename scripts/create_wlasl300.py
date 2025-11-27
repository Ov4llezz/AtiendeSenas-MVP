"""
Script para crear el dataset WLASL300 desde el dataset completo WLASL_full.
Selecciona las 300 glosas con más videos y crea la estructura necesaria
para el pipeline de entrenamiento.
"""

import os
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ===== CONFIGURACIÓN =====
WLASL_FULL_PATH = r"C:\Users\ov4ll\Desktop\TESIS\WLASL_full"
OUTPUT_BASE = r"data\wlasl300"
NUM_GLOSSES = 300

# Archivos del dataset completo
SAMPLES_JSON = os.path.join(WLASL_FULL_PATH, "samples.json")
DATA_DIR = os.path.join(WLASL_FULL_PATH, "data")

print("=" * 80)
print("Creando dataset WLASL300")
print("=" * 80)

# ===== PASO 1: Cargar y analizar samples.json =====
print("\n[1/7] Cargando samples.json...")
with open(SAMPLES_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]
print(f"Total de videos en WLASL_full: {len(samples)}")

# ===== PASO 2: Contar videos por glosa =====
print("\n[2/7] Contando videos por glosa...")
gloss_counts = Counter()
gloss_videos = defaultdict(list)  # glosa -> lista de videos

for sample in samples:
    if "gloss" in sample and sample["gloss"]:
        gloss = sample["gloss"]["label"]
        gloss_counts[gloss] += 1
        gloss_videos[gloss].append(sample)

print(f"Total de glosas encontradas: {len(gloss_counts)}")

# ===== PASO 3: Seleccionar top 300 glosas =====
print(f"\n[3/7] Seleccionando top {NUM_GLOSSES} glosas con más videos...")
top_glosses = [gloss for gloss, count in gloss_counts.most_common(NUM_GLOSSES)]

print(f"\nTop 10 glosas:")
for i, (gloss, count) in enumerate(gloss_counts.most_common(10), 1):
    print(f"  {i}. {gloss}: {count} videos")

total_videos = sum(gloss_counts[g] for g in top_glosses)
print(f"\nTotal de videos en las top {NUM_GLOSSES} glosas: {total_videos}")

# ===== PASO 4: Crear estructura de directorios =====
print(f"\n[4/7] Creando estructura de directorios en {OUTPUT_BASE}...")
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "videos"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "dataset", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "dataset", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "dataset", "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "splits"), exist_ok=True)

print("[OK] Estructura de directorios creada")

# ===== PASO 5: Copiar videos y crear splits =====
print(f"\n[5/7] Copiando videos y creando splits...")

# Crear mapeo de glosa a ID de clase (0-299)
gloss_to_class_id = {gloss: idx for idx, gloss in enumerate(top_glosses)}

# Estructuras para los archivos de salida
nslt_300 = {}  # video_id -> {subset, action}
train_list = []
val_list = []
test_list = []

# Variables para estadísticas
copied_count = 0
skipped_count = 0
split_counts = {"train": 0, "val": 0, "test": 0}

# Para crear WLASL_v0.3_300.json
wlasl_entries = []
gloss_to_instances = defaultdict(list)

for gloss in top_glosses:
    class_id = gloss_to_class_id[gloss]

    for sample in gloss_videos[gloss]:
        filepath = sample["filepath"]  # Ej: "data/data_0/00335.mp4"
        video_filename = os.path.basename(filepath)  # "00335.mp4"
        video_id = os.path.splitext(video_filename)[0]  # "00335"

        # Ruta completa en WLASL_full
        source_path = os.path.join(WLASL_FULL_PATH, filepath)

        if not os.path.exists(source_path):
            skipped_count += 1
            continue

        # Determinar el split basado en el índice del video
        # Usamos la misma distribución que WLASL original: ~70% train, ~15% val, ~15% test
        # Esto es una simplificación; idealmente usaríamos los splits originales si estuvieran disponibles
        video_idx = int(video_id)
        if video_idx % 10 < 7:
            split = "train"
        elif video_idx % 10 < 8 or video_idx % 10 == 9:
            split = "val"
        else:
            split = "test"

        # Copiar video a la carpeta de videos (opcional, para backup)
        dest_videos = os.path.join(OUTPUT_BASE, "videos", video_filename)
        if not os.path.exists(dest_videos):
            shutil.copy2(source_path, dest_videos)

        # Copiar video a la carpeta del split correspondiente
        dest_split = os.path.join(OUTPUT_BASE, "dataset", split, video_filename)
        if not os.path.exists(dest_split):
            shutil.copy2(source_path, dest_split)
            copied_count += 1
            split_counts[split] += 1

        # Agregar a nslt_300.json
        nslt_300[video_id] = {
            "subset": split,
            "action": [class_id, -1, -1]  # Formato compatible con nslt_100.json
        }

        # Agregar a las listas de splits
        split_entry = f"{split}\\{video_filename}"
        if split == "train":
            train_list.append(split_entry)
        elif split == "val":
            val_list.append(split_entry)
        elif split == "test":
            test_list.append(split_entry)

        # Preparar entrada para WLASL_v0.3_300.json
        instance = {
            "bbox": sample["bounding_box"]["detections"][0]["bounding_box"] if sample.get("bounding_box") and sample["bounding_box"].get("detections") else [0, 0, 1, 1],
            "fps": sample.get("metadata", {}).get("frame_rate", 30),
            "frame_start": 0,
            "frame_end": sample.get("metadata", {}).get("total_frame_count", 60),
            "instance_id": copied_count,
            "signer_id": -1,
            "source": "WLASL",
            "split": split,
            "url": "",
            "variation_id": -1,
            "video_id": video_id
        }
        gloss_to_instances[gloss].append(instance)

print(f"\n[OK] Videos copiados: {copied_count}")
print(f"  [X] Videos no encontrados: {skipped_count}")
print(f"\nDistribución por split:")
print(f"  Train: {split_counts['train']} ({split_counts['train']/copied_count*100:.1f}%)")
print(f"  Val:   {split_counts['val']} ({split_counts['val']/copied_count*100:.1f}%)")
print(f"  Test:  {split_counts['test']} ({split_counts['test']/copied_count*100:.1f}%)")

# ===== PASO 6: Crear archivos JSON de configuración =====
print(f"\n[6/7] Generando archivos JSON de configuración...")

# Guardar nslt_300.json
nslt_path = os.path.join(OUTPUT_BASE, "nslt_300.json")
with open(nslt_path, "w", encoding="utf-8") as f:
    json.dump(nslt_300, f, indent=2, ensure_ascii=False)
print(f"[OK] Creado: {nslt_path}")

# Crear WLASL_v0.3_300.json
for gloss in top_glosses:
    class_id = gloss_to_class_id[gloss]
    wlasl_entries.append({
        "gloss": gloss,
        "instances": gloss_to_instances[gloss]
    })

wlasl_v03_path = os.path.join(OUTPUT_BASE, "WLASL_v0.3_300.json")
with open(wlasl_v03_path, "w", encoding="utf-8") as f:
    json.dump(wlasl_entries, f, indent=2, ensure_ascii=False)
print(f"[OK] Creado: {wlasl_v03_path}")

# Crear archivo de mapeo de glosas a IDs
gloss_mapping_path = os.path.join(OUTPUT_BASE, "gloss_to_id.json")
with open(gloss_mapping_path, "w", encoding="utf-8") as f:
    json.dump(gloss_to_class_id, f, indent=2, ensure_ascii=False)
print(f"[OK] Creado: {gloss_mapping_path}")

# ===== PASO 7: Guardar archivos de splits =====
print(f"\n[7/7] Guardando archivos de splits...")

with open(os.path.join(OUTPUT_BASE, "splits", "train_split.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_list))
print(f"[OK] train_split.txt: {len(train_list)} videos")

with open(os.path.join(OUTPUT_BASE, "splits", "val_split.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(val_list))
print(f"[OK] val_split.txt: {len(val_list)} videos")

with open(os.path.join(OUTPUT_BASE, "splits", "test_split.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(test_list))
print(f"[OK] test_split.txt: {len(test_list)} videos")

# ===== RESUMEN FINAL =====
print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"Dataset: WLASL300")
print(f"Glosas seleccionadas: {NUM_GLOSSES}")
print(f"Videos totales: {copied_count}")
print(f"  - Train: {len(train_list)}")
print(f"  - Val:   {len(val_list)}")
print(f"  - Test:  {len(test_list)}")
print(f"\nArchivos generados:")
print(f"  - {nslt_path}")
print(f"  - {wlasl_v03_path}")
print(f"  - {gloss_mapping_path}")
print(f"  - splits/train_split.txt")
print(f"  - splits/val_split.txt")
print(f"  - splits/test_split.txt")
print(f"\nDataset listo en: {OUTPUT_BASE}")
print("=" * 80)
