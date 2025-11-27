"""
Script para generar un reporte detallado del dataset WLASL100.
Incluye estadísticas de glosas, videos por glosa, y distribución por split.
CORREGIDO: Cuenta solo los videos que realmente existen en las carpetas locales.
"""

import json
import os
from collections import defaultdict, Counter

# Rutas
WLASL100_PATH = "data/wlasl100"
NSLT_100_JSON = os.path.join(WLASL100_PATH, "nslt_100.json")
WLASL_V03_JSON = os.path.join(WLASL100_PATH, "WLASL_v0.3.json")
SPLITS_DIR = os.path.join(WLASL100_PATH, "splits")
OUTPUT_FILE = "WLASL100_DATASET_REPORT.txt"

print("=" * 80)
print("Generando reporte detallado de WLASL100 (CORREGIDO)")
print("=" * 80)

# Cargar nslt_100.json para mapeo video_id -> class_id
print("\n[1/5] Cargando archivos JSON...")
with open(NSLT_100_JSON, 'r', encoding='utf-8') as f:
    nslt_data = json.load(f)

print(f"   - Videos en nslt_100.json: {len(nslt_data)}")

# Cargar WLASL_v0.3.json para mapeo class_id -> gloss
id_to_gloss = {}
if os.path.exists(WLASL_V03_JSON):
    with open(WLASL_V03_JSON, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)

    for entry in wlasl_data:
        gloss = entry['gloss']
        if 'instances' in entry and len(entry['instances']) > 0:
            for instance in entry['instances']:
                video_id = instance['video_id']
                if video_id in nslt_data:
                    class_id = nslt_data[video_id]['action'][0]
                    id_to_gloss[class_id] = gloss
                    break

print(f"   - Glosas mapeadas: {len(id_to_gloss)}")

# Leer archivos de splits para saber qué videos realmente existen
print("\n[2/5] Leyendo archivos de splits...")

splits_data = {'train': [], 'val': [], 'test': []}

for split in ['train', 'val', 'test']:
    split_file = os.path.join(SPLITS_DIR, f"{split}_split.txt")
    if os.path.exists(split_file):
        with open(split_file, 'r', encoding='utf-8') as f:
            # El formato es "train\00623.mp4" o similar
            lines = [line.strip() for line in f if line.strip()]
            splits_data[split] = lines
            print(f"   - {split}: {len(lines)} videos")

# Extraer solo los video IDs de los splits
actual_videos = {}  # video_id -> split
for split, lines in splits_data.items():
    for line in lines:
        # Extraer el nombre del archivo (ej: "train\00623.mp4" -> "00623.mp4")
        filename = os.path.basename(line)
        video_id = os.path.splitext(filename)[0]  # "00623"
        actual_videos[video_id] = split

print(f"   - Total videos locales: {len(actual_videos)}")

# Análisis por glosa usando solo videos que existen localmente
print("\n[3/5] Analizando distribución de videos...")

gloss_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
split_counts = Counter()

for video_id, split in actual_videos.items():
    if video_id in nslt_data:
        class_id = nslt_data[video_id]['action'][0]
        gloss = id_to_gloss.get(class_id, f"Unknown_{class_id}")

        gloss_stats[gloss][split] += 1
        gloss_stats[gloss]['total'] += 1
        split_counts[split] += 1

print(f"   - Glosas con videos: {len(gloss_stats)}")
print(f"   - Train: {split_counts['train']}")
print(f"   - Val: {split_counts['val']}")
print(f"   - Test: {split_counts['test']}")

# Ordenar glosas alfabéticamente
print("\n[4/5] Ordenando estadísticas...")
sorted_glosses = sorted(gloss_stats.items(), key=lambda x: x[0])

# Generar reporte
print(f"\n[5/5] Generando reporte en {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    # Encabezado
    f.write("=" * 80 + "\n")
    f.write("REPORTE DETALLADO DEL DATASET WLASL100\n")
    f.write("Sign Language Recognition - WLASL 100 Classes\n")
    f.write("=" * 80 + "\n\n")

    # Resumen general
    f.write("RESUMEN GENERAL\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total de Glosas (Clases):        {len(gloss_stats)}\n")
    f.write(f"Total de Videos:                 {sum(split_counts.values())}\n")
    f.write(f"  - Training:                    {split_counts['train']} ({split_counts['train']/sum(split_counts.values())*100:.1f}%)\n")
    f.write(f"  - Validation:                  {split_counts['val']} ({split_counts['val']/sum(split_counts.values())*100:.1f}%)\n")
    f.write(f"  - Test:                        {split_counts['test']} ({split_counts['test']/sum(split_counts.values())*100:.1f}%)\n")
    f.write("\n")

    # Estadísticas de distribución
    videos_per_gloss = [stats['total'] for stats in gloss_stats.values()]
    f.write("ESTADISTICAS DE DISTRIBUCION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Videos por glosa (promedio):     {sum(videos_per_gloss)/len(videos_per_gloss):.1f}\n")
    f.write(f"Videos por glosa (minimo):       {min(videos_per_gloss)}\n")
    f.write(f"Videos por glosa (maximo):       {max(videos_per_gloss)}\n")
    f.write(f"Videos por glosa (mediana):      {sorted(videos_per_gloss)[len(videos_per_gloss)//2]}\n")
    f.write("\n\n")

    # Tabla detallada por glosa
    f.write("DETALLE POR GLOSA (ALFABETICO)\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'#':<5} {'Glosa':<25} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}\n")
    f.write("-" * 80 + "\n")

    for idx, (gloss, stats) in enumerate(sorted_glosses, 1):
        f.write(f"{idx:<5} {gloss:<25} {stats['total']:<8} "
                f"{stats['train']:<8} {stats['val']:<8} {stats['test']:<8}\n")

    f.write("=" * 80 + "\n\n")

    # Top 20 glosas con más videos
    f.write("TOP 20 GLOSAS CON MAS VIDEOS\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Rank':<6} {'Glosa':<25} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}\n")
    f.write("-" * 80 + "\n")

    sorted_by_total = sorted(gloss_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:20]
    for rank, (gloss, stats) in enumerate(sorted_by_total, 1):
        f.write(f"{rank:<6} {gloss:<25} {stats['total']:<8} "
                f"{stats['train']:<8} {stats['val']:<8} {stats['test']:<8}\n")

    f.write("=" * 80 + "\n\n")

    # Top 20 glosas con menos videos
    f.write("TOP 20 GLOSAS CON MENOS VIDEOS\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Rank':<6} {'Glosa':<25} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}\n")
    f.write("-" * 80 + "\n")

    sorted_by_total_asc = sorted(gloss_stats.items(), key=lambda x: x[1]['total'])[:20]
    for rank, (gloss, stats) in enumerate(sorted_by_total_asc, 1):
        f.write(f"{rank:<6} {gloss:<25} {stats['total']:<8} "
                f"{stats['train']:<8} {stats['val']:<8} {stats['test']:<8}\n")

    f.write("=" * 80 + "\n\n")

    # Información del dataset
    f.write("INFORMACION DEL DATASET\n")
    f.write("=" * 80 + "\n")
    f.write("Nombre:           WLASL100\n")
    f.write("Fuente:           WLASL (Word-Level American Sign Language)\n")
    f.write("Idioma:           American Sign Language (ASL)\n")
    f.write("Tipo:             Video (Sign Language Recognition)\n")
    f.write("Seleccion:        100 glosas mas comunes para entrenamiento\n")
    f.write("Formato videos:   MP4\n")
    f.write("Resolucion:       Variable (procesado a 224x224 en entrenamiento)\n")
    f.write("FPS:              Variable (muestreado a 16 frames por video)\n")
    f.write("\n")
    f.write("Nota:             Este reporte cuenta solo los videos descargados\n")
    f.write("                  localmente en data/wlasl100/dataset/\n")
    f.write("\n")
    f.write("Estructura del dataset:\n")
    f.write("  data/wlasl100/\n")
    f.write("  ├── dataset/\n")
    f.write("  │   ├── train/          # Videos de entrenamiento\n")
    f.write("  │   ├── val/            # Videos de validacion\n")
    f.write("  │   └── test/           # Videos de prueba\n")
    f.write("  ├── splits/\n")
    f.write("  │   ├── train_split.txt\n")
    f.write("  │   ├── val_split.txt\n")
    f.write("  │   └── test_split.txt\n")
    f.write("  ├── nslt_100.json       # Mapeo video_id -> clase\n")
    f.write("  └── WLASL_v0.3.json     # Metadata del dataset\n")
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("Generado por: scripts/generate_wlasl100_report.py (CORREGIDO)\n")
    f.write("Proyecto: AtiendeSenas - Tesis UNAB\n")
    f.write("Autor: Rafael Ovalle\n")
    f.write("=" * 80 + "\n")

print(f"\n[OK] Reporte generado exitosamente: {OUTPUT_FILE}")
print(f"[OK] Total de glosas: {len(gloss_stats)}")
print(f"[OK] Total de videos: {sum(split_counts.values())}")
print("\n" + "=" * 80)
