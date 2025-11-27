"""
Script para generar un reporte detallado del dataset WLASL300.
Incluye estadísticas de glosas, videos por glosa, y distribución por split.
"""

import json
import os
from collections import defaultdict, Counter

# Rutas
WLASL300_PATH = "data/wlasl300"
NSLT_300_JSON = os.path.join(WLASL300_PATH, "nslt_300.json")
GLOSS_TO_ID_JSON = os.path.join(WLASL300_PATH, "gloss_to_id.json")
OUTPUT_FILE = "WLASL300_DATASET_REPORT.txt"

print("=" * 80)
print("Generando reporte detallado de WLASL300")
print("=" * 80)

# Cargar datos
print("\n[1/4] Cargando archivos JSON...")
with open(NSLT_300_JSON, 'r', encoding='utf-8') as f:
    nslt_data = json.load(f)

with open(GLOSS_TO_ID_JSON, 'r', encoding='utf-8') as f:
    gloss_to_id = json.load(f)

# Invertir el mapeo para tener id -> gloss
id_to_gloss = {v: k for k, v in gloss_to_id.items()}

print(f"   - Videos mapeados: {len(nslt_data)}")
print(f"   - Glosas únicas: {len(gloss_to_id)}")

# Análisis por glosa y split
print("\n[2/4] Analizando distribución de videos...")

# Estructura: glosa -> {'train': count, 'val': count, 'test': count, 'total': count}
gloss_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

# Contadores por split
split_counts = Counter()

for video_id, info in nslt_data.items():
    class_id = info['action'][0]
    split = info['subset']

    gloss = id_to_gloss.get(class_id, f"Unknown_{class_id}")

    gloss_stats[gloss][split] += 1
    gloss_stats[gloss]['total'] += 1
    split_counts[split] += 1

print(f"   - Total videos procesados: {sum(split_counts.values())}")
print(f"   - Train: {split_counts['train']}")
print(f"   - Val: {split_counts['val']}")
print(f"   - Test: {split_counts['test']}")

# Ordenar glosas alfabéticamente
print("\n[3/4] Ordenando estadísticas...")
sorted_glosses = sorted(gloss_stats.items(), key=lambda x: x[0])

# Generar reporte
print(f"\n[4/4] Generando reporte en {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    # Encabezado
    f.write("=" * 80 + "\n")
    f.write("REPORTE DETALLADO DEL DATASET WLASL300\n")
    f.write("Sign Language Recognition - WLASL 300 Classes\n")
    f.write("=" * 80 + "\n\n")

    # Resumen general
    f.write("RESUMEN GENERAL\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total de Glosas (Clases):        300\n")
    f.write(f"Total de Videos:                 {sum(split_counts.values())}\n")
    f.write(f"  - Training:                    {split_counts['train']} ({split_counts['train']/sum(split_counts.values())*100:.1f}%)\n")
    f.write(f"  - Validation:                  {split_counts['val']} ({split_counts['val']/sum(split_counts.values())*100:.1f}%)\n")
    f.write(f"  - Test:                        {split_counts['test']} ({split_counts['test']/sum(split_counts.values())*100:.1f}%)\n")
    f.write("\n")

    # Estadísticas de distribución
    videos_per_gloss = [stats['total'] for stats in gloss_stats.values()]
    f.write("ESTADÍSTICAS DE DISTRIBUCIÓN\n")
    f.write("-" * 80 + "\n")
    f.write(f"Videos por glosa (promedio):     {sum(videos_per_gloss)/len(videos_per_gloss):.1f}\n")
    f.write(f"Videos por glosa (mínimo):       {min(videos_per_gloss)}\n")
    f.write(f"Videos por glosa (máximo):       {max(videos_per_gloss)}\n")
    f.write(f"Videos por glosa (mediana):      {sorted(videos_per_gloss)[len(videos_per_gloss)//2]}\n")
    f.write("\n\n")

    # Tabla detallada por glosa
    f.write("DETALLE POR GLOSA (ALFABÉTICO)\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'#':<5} {'Glosa':<25} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}\n")
    f.write("-" * 80 + "\n")

    for idx, (gloss, stats) in enumerate(sorted_glosses, 1):
        f.write(f"{idx:<5} {gloss:<25} {stats['total']:<8} "
                f"{stats['train']:<8} {stats['val']:<8} {stats['test']:<8}\n")

    f.write("=" * 80 + "\n\n")

    # Top 20 glosas con más videos
    f.write("TOP 20 GLOSAS CON MÁS VIDEOS\n")
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
    f.write("INFORMACIÓN DEL DATASET\n")
    f.write("=" * 80 + "\n")
    f.write("Nombre:           WLASL300\n")
    f.write("Fuente:           WLASL (Word-Level American Sign Language)\n")
    f.write("Origen:           Voxel51/WLASL (HuggingFace)\n")
    f.write("Idioma:           American Sign Language (ASL)\n")
    f.write("Tipo:             Video (Sign Language Recognition)\n")
    f.write("Selección:        Top 300 glosas con más videos del dataset completo\n")
    f.write("Formato videos:   MP4\n")
    f.write("Resolución:       Variable (procesado a 224x224 en entrenamiento)\n")
    f.write("FPS:              Variable (muestreado a 16 frames por video)\n")
    f.write("\n")
    f.write("Estructura del dataset:\n")
    f.write("  data/wlasl300/\n")
    f.write("  ├── dataset/\n")
    f.write("  │   ├── train/          # Videos de entrenamiento\n")
    f.write("  │   ├── val/            # Videos de validación\n")
    f.write("  │   └── test/           # Videos de prueba\n")
    f.write("  ├── splits/\n")
    f.write("  │   ├── train_split.txt\n")
    f.write("  │   ├── val_split.txt\n")
    f.write("  │   └── test_split.txt\n")
    f.write("  ├── nslt_300.json       # Mapeo video_id -> clase\n")
    f.write("  ├── WLASL_v0.3_300.json # Metadata del dataset\n")
    f.write("  └── gloss_to_id.json    # Mapeo glosa -> ID de clase\n")
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("Generado por: scripts/generate_wlasl300_report.py\n")
    f.write("Proyecto: AtiendeSenas - Tesis UNAB\n")
    f.write("Autor: Rafael Ovalle\n")
    f.write("=" * 80 + "\n")

print(f"\n[OK] Reporte generado exitosamente: {OUTPUT_FILE}")
print(f"[OK] Total de glosas: {len(gloss_stats)}")
print(f"[OK] Total de videos: {sum(split_counts.values())}")
print("\n" + "=" * 80)
