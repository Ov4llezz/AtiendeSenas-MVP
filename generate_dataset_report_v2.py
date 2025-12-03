import os
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import cv2

def check_video_integrity(video_path, min_size=1000):
    """
    Verifica la integridad de un video.
    Retorna: ('valid', 'corrupt', 'missing')
    """
    if not os.path.exists(video_path):
        return 'missing'

    # Verificar tamaño del archivo
    file_size = os.path.getsize(video_path)
    if file_size < min_size:
        return 'corrupt'

    # Intentar abrir el video con OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 'corrupt'

        # Intentar leer el primer frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return 'corrupt'

        return 'valid'
    except Exception as e:
        return 'corrupt'

def load_gloss_mapping(json_path, gloss_to_id_path):
    """Carga el mapeo de video_id a gloss desde los archivos JSON."""
    # Cargar el mapeo gloss_name -> gloss_id
    with open(gloss_to_id_path, 'r') as f:
        gloss_to_id = json.load(f)

    # Invertir el mapeo para obtener id -> gloss_name
    id_to_gloss = {v: k for k, v in gloss_to_id.items()}

    # Cargar el mapeo video_id -> video_data
    with open(json_path, 'r') as f:
        video_data = json.load(f)

    # Construir mapeo video_id -> gloss
    video_to_gloss = {}
    for video_id, data in video_data.items():
        # action es una lista, el primer elemento es el gloss_id
        if 'action' in data and len(data['action']) > 0:
            gloss_id = data['action'][0]
            gloss_name = id_to_gloss.get(gloss_id, 'unknown')
            video_to_gloss[video_id] = gloss_name

    return video_to_gloss, gloss_to_id

def analyze_split_with_glosses(split_file, videos_dir, video_to_gloss):
    """Analiza un split y retorna estadísticas por glosa."""
    stats = {
        'total': 0,
        'valid': 0,
        'corrupt': 0,
        'missing': 0,
        'corrupt_videos': [],
        'missing_videos': [],
        'gloss_counts': defaultdict(int)
    }

    if not os.path.exists(split_file):
        return stats

    with open(split_file, 'r') as f:
        lines = f.readlines()

    stats['total'] = len(lines)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 1:
            continue

        video_path_rel = parts[0]

        # Extraer video_id
        if video_path_rel.endswith('.mp4'):
            video_filename = os.path.basename(video_path_rel)
            video_id = video_filename.replace('.mp4', '')
            video_path = os.path.join(videos_dir, video_filename)
            video_name = video_path_rel
        else:
            video_id = video_path_rel
            video_path = os.path.join(videos_dir, f"{video_path_rel}.mp4")
            video_name = f"{video_path_rel}.mp4"

        # Obtener glosa
        gloss = video_to_gloss.get(video_id, 'unknown')

        status = check_video_integrity(video_path)

        if status == 'valid':
            stats['valid'] += 1
            stats['gloss_counts'][gloss] += 1
        elif status == 'corrupt':
            stats['corrupt'] += 1
            stats['corrupt_videos'].append(video_name)
        elif status == 'missing':
            stats['missing'] += 1
            stats['missing_videos'].append(video_name)

    return stats

def generate_detailed_report(dataset_name, dataset_path, output_file):
    """Genera un reporte detallado del dataset."""

    splits_dir = os.path.join(dataset_path, 'splits')
    videos_dir = os.path.join(dataset_path, 'videos')

    # Determinar archivo JSON
    if '100' in dataset_name:
        json_file = os.path.join(dataset_path, 'nslt_100.json')
        num_classes = 100
    else:
        json_file = os.path.join(dataset_path, 'nslt_300.json')
        num_classes = 300

    gloss_to_id_file = os.path.join(dataset_path, 'gloss_to_id.json')

    # Cargar mapeo de glosas
    print(f"Cargando mapeo de glosas desde {json_file}...")
    video_to_gloss, gloss_to_id = load_gloss_mapping(json_file, gloss_to_id_file)

    # Analizar cada split
    splits = ['train', 'val', 'test']
    all_stats = {}

    for split in splits:
        split_file = os.path.join(splits_dir, f"{split}_split.txt")
        print(f"Analizando {split}...")
        all_stats[split] = analyze_split_with_glosses(split_file, videos_dir, video_to_gloss)

    # Combinar estadísticas por glosa
    gloss_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

    for split in splits:
        for gloss, count in all_stats[split]['gloss_counts'].items():
            gloss_stats[gloss][split] = count
            gloss_stats[gloss]['total'] += count

    # Calcular estadísticas de distribución
    totals = [stats['total'] for stats in gloss_stats.values() if stats['total'] > 0]
    if totals:
        avg_videos = sum(totals) / len(totals)
        min_videos = min(totals)
        max_videos = max(totals)
        sorted_totals = sorted(totals)
        median_videos = sorted_totals[len(sorted_totals) // 2]
    else:
        avg_videos = min_videos = max_videos = median_videos = 0

    # Ordenar glosas alfabéticamente
    sorted_glosses = sorted(gloss_stats.items(), key=lambda x: x[0])

    # Top 20 con más y menos videos
    top_glosses = sorted(gloss_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:20]
    bottom_glosses = sorted(gloss_stats.items(), key=lambda x: x[1]['total'])[:20]

    # Calcular totales globales
    total_valid = sum(all_stats[s]['valid'] for s in splits)
    total_corrupt = sum(all_stats[s]['corrupt'] for s in splits)
    total_missing = sum(all_stats[s]['missing'] for s in splits)

    # Generar reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write("=" * 80 + "\n")
        f.write(f"REPORTE DETALLADO DEL DATASET {dataset_name.upper()}\n")
        f.write(f"Sign Language Recognition - WLASL {num_classes} Classes\n")
        f.write("=" * 80 + "\n\n")

        # Resumen general
        f.write("RESUMEN GENERAL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total de Glosas (Clases):        {len(gloss_stats)}\n")
        f.write(f"Total de Videos:                 {total_valid}\n")
        f.write(f"  - Training:                    {all_stats['train']['valid']} "
                f"({all_stats['train']['valid']/total_valid*100:.1f}%)\n")
        f.write(f"  - Validation:                  {all_stats['val']['valid']} "
                f"({all_stats['val']['valid']/total_valid*100:.1f}%)\n")
        f.write(f"  - Test:                        {all_stats['test']['valid']} "
                f"({all_stats['test']['valid']/total_valid*100:.1f}%)\n")
        f.write("\n")

        # Estadísticas de distribución
        f.write("ESTADISTICAS DE DISTRIBUCION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Videos por glosa (promedio):     {avg_videos:.1f}\n")
        f.write(f"Videos por glosa (minimo):       {min_videos}\n")
        f.write(f"Videos por glosa (maximo):       {max_videos}\n")
        f.write(f"Videos por glosa (mediana):      {median_videos}\n")
        f.write("\n\n")

        # Detalle por glosa
        f.write("DETALLE POR GLOSA (ALFABETICO)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'#':<6}{'Glosa':<26}{'Total':<9}{'Train':<9}{'Val':<9}{'Test':<9}\n")
        f.write("-" * 80 + "\n")

        for idx, (gloss, stats) in enumerate(sorted_glosses, 1):
            f.write(f"{idx:<6}{gloss:<26}{stats['total']:<9}{stats['train']:<9}"
                   f"{stats['val']:<9}{stats['test']:<9}\n")

        f.write("=" * 80 + "\n\n")

        # Top 20 con más videos
        f.write("TOP 20 GLOSAS CON MAS VIDEOS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<7}{'Glosa':<26}{'Total':<9}{'Train':<9}{'Val':<9}{'Test':<9}\n")
        f.write("-" * 80 + "\n")

        for idx, (gloss, stats) in enumerate(top_glosses, 1):
            f.write(f"{idx:<7}{gloss:<26}{stats['total']:<9}{stats['train']:<9}"
                   f"{stats['val']:<9}{stats['test']:<9}\n")

        f.write("=" * 80 + "\n\n")

        # Top 20 con menos videos
        f.write("TOP 20 GLOSAS CON MENOS VIDEOS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<7}{'Glosa':<26}{'Total':<9}{'Train':<9}{'Val':<9}{'Test':<9}\n")
        f.write("-" * 80 + "\n")

        for idx, (gloss, stats) in enumerate(bottom_glosses, 1):
            f.write(f"{idx:<7}{gloss:<26}{stats['total']:<9}{stats['train']:<9}"
                   f"{stats['val']:<9}{stats['test']:<9}\n")

        f.write("=" * 80 + "\n\n")

        # Información del dataset
        f.write("INFORMACION DEL DATASET\n")
        f.write("=" * 80 + "\n")
        f.write(f"Nombre:           {dataset_name.upper()}\n")
        f.write(f"Fuente:           WLASL (Word-Level American Sign Language)\n")
        f.write(f"Idioma:           American Sign Language (ASL)\n")
        f.write(f"Tipo:             Video (Sign Language Recognition)\n")
        f.write(f"Seleccion:        {num_classes} glosas mas comunes para entrenamiento\n")
        f.write(f"Formato videos:   MP4\n")
        f.write(f"Resolucion:       Variable (procesado a 224x224 en entrenamiento)\n")
        f.write(f"FPS:              Variable (muestreado a 16 frames por video)\n")
        f.write(f"\n")
        f.write(f"Nota:             Este reporte cuenta solo los videos descargados\n")
        f.write(f"                  localmente en {dataset_path}/\n")
        f.write(f"\n")
        f.write(f"Estructura del dataset:\n")
        f.write(f"  {dataset_path}/\n")
        f.write(f"  ├── videos/             # Todos los videos\n")
        f.write(f"  ├── splits/\n")
        f.write(f"  │   ├── train_split.txt\n")
        f.write(f"  │   ├── val_split.txt\n")
        f.write(f"  │   └── test_split.txt\n")
        f.write(f"  ├── nslt_{num_classes}.json       # Mapeo video_id -> clase\n")
        f.write(f"  └── WLASL_v0.3{'_300' if num_classes == 300 else ''}.json     # Metadata del dataset\n")
        f.write(f"\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generado por: generate_dataset_report_v2.py\n")
        f.write(f"Fecha: {timestamp}\n")
        f.write(f"Proyecto: AtiendeSenas - Tesis UNAB\n")
        f.write(f"Autor: Rafael Ovalle\n")
        f.write("=" * 80 + "\n")

    print(f"Reporte generado: {output_file}")
    return output_file

if __name__ == "__main__":
    # Generar reportes para wlasl100_v2 y wlasl300_v2
    base_path = "data"

    datasets = [
        ("WLASL100_v2", os.path.join(base_path, "wlasl100_v2")),
        ("WLASL300_v2", os.path.join(base_path, "wlasl300_v2"))
    ]

    for dataset_name, dataset_path in datasets:
        output_file = os.path.join(dataset_path, f"{dataset_name.upper()}_DATASET_REPORT.txt")
        print(f"\n{'='*80}")
        print(f"Generando reporte detallado para {dataset_name}...")
        print(f"{'='*80}")
        generate_detailed_report(dataset_name, dataset_path, output_file)

    print("\n¡Todos los reportes generados exitosamente!")
