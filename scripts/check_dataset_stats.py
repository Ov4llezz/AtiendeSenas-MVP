from pathlib import Path
import cv2

# =============================
# Rutas
# =============================
BASE = Path(__file__).resolve().parents[1] / "data" / "wlasl100"
SPLIT_FILE = BASE / "splits" / "train_split.txt"
TRAIN_DIR = BASE / "dataset" / "train"

print("== DIAGNÓSTICO DATASET WLASL100 (TRAIN) ==")

# =============================
# Leer split.txt
# =============================
lines = [
    l.strip()
    for l in SPLIT_FILE.read_text(encoding="utf-8").splitlines()
    if l.strip()
]

expected = len(lines)

# Normalizar nombres (quitar train\ o train/)
names = [Path(l.replace("\\", "/")).name for l in lines]

# =============================
# Archivos existentes / faltantes
# =============================
existing = [n for n in names if (TRAIN_DIR / n).exists()]
missing = [n for n in names if not (TRAIN_DIR / n).exists()]

print(f"Videos esperados (split):  {expected}")
print(f"Videos encontrados:        {len(existing)}")
print(f"Videos faltantes:          {len(missing)}")
print()

# =============================
# Detectar videos corruptos
# =============================
corrupt = []

print("[INFO] Revisando corrupción en videos... esto puede tardar 1-2 minutos.\n")

for filename in existing:
    path = str(TRAIN_DIR / filename)
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()

    # Criterio de corrupción:
    # - 0 frames
    # - no se puede leer el primer frame
    if frame_count == 0 or not ok:
        corrupt.append(filename)

# =============================
# Resultados
# =============================
print("==== RESULTADOS ====")
print(f"Videos faltantes:          {len(missing)}")
print(f"Videos corruptos:          {len(corrupt)}")
print(f"Videos utilizables:        {len(existing) - len(corrupt)}")
print()

print("Ejemplos de faltantes:", missing[:10])
print("Ejemplos de corruptos:", corrupt[:10])

# Guardar lista completa si quieres
out_corrupt = BASE / "corrupt_videos_train.txt"
out_corrupt.write_text("\n".join(corrupt), encoding="utf-8")

out_missing = BASE / "missing_videos_train.txt"
out_missing.write_text("\n".join(missing), encoding="utf-8")

print("\nSe guardaron:")
print(f" - Lista de corruptos en {out_corrupt}")
print(f" - Lista de faltantes en {out_missing}")
print("\n== FIN ==")



