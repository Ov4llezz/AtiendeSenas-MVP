import os
import json
import shutil

VIDEOS_DIR = "videos_100"
OUT_DIR = "dataset_wlasl100"

# Crear carpetas destino
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

# Cargar JSONs
meta = json.load(open("WLASL_v0.3.json"))
allowed_glosses = {c["gloss"] for c in json.load(open("nsltt_100.json"))}

# Mapa video_id → split
video_to_split = {}

for entry in meta:
    gloss = entry["gloss"]
    if gloss not in allowed_glosses:
        continue

    for inst in entry["instances"]:
        vid = inst["video_id"]
        split = inst["split"]
        video_to_split[vid] = split

# Listas de splits (para los .txt)
train_list = []
val_list = []
test_list = []

# Copiar según split y llenar listas
for fname in os.listdir(VIDEOS_DIR):
    if not fname.endswith(".mp4"):
        continue

    video_id = fname.replace(".mp4", "")

    if video_id not in video_to_split:
        continue

    split = video_to_split[video_id]
    src = os.path.join(VIDEOS_DIR, fname)
    dst = os.path.join(OUT_DIR, split, fname)

    shutil.copyfile(src, dst)

    if split == "train":
        train_list.append(fname)
    elif split == "val":
        val_list.append(fname)
    elif split == "test":
        test_list.append(fname)

# Guardar archivos .txt
with open("train_split.txt", "w") as f:
    f.writelines(f"{v}\n" for v in train_list)

with open("val_split.txt", "w") as f:
    f.writelines(f"{v}\n" for v in val_list)

with open("test_split.txt", "w") as f:
    f.writelines(f"{v}\n" for v in test_list)

# Resumen
print("\nSplit completado correctamente:")
print("  Train:", len(train_list))
print("  Val:  ", len(val_list))
print("  Test: ", len(test_list))
print("  Total:", len(train_list) + len(val_list) + len(test_list))
print("\nArchivos generados: train_split.txt, val_split.txt, test_split.txt")


