# Script para actualizar los splits de los datasets V2.
# Reemplaza referencias de val\ a train\ en train_split.txt

import os

def update_splits(base_path):
    # Actualiza train_split.txt para reemplazar val por train
    train_split_path = os.path.join(base_path, 'splits', 'train_split.txt')

    if not os.path.exists(train_split_path):
        print(f"[ERROR] No encontrado: {train_split_path}")
        return False

    # Leer archivo
    with open(train_split_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Contar cambios
    val_count = sum(1 for line in lines if line.startswith('val\\'))

    if val_count == 0:
        print(f"[OK] {base_path}: Ya esta actualizado (0 referencias a val)")
        return True

    # Reemplazar val por train
    updated_lines = []
    for line in lines:
        if line.startswith('val\\'):
            updated_lines.append(line.replace('val\\', 'train\\', 1))
        else:
            updated_lines.append(line)

    # Guardar archivo actualizado
    with open(train_split_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

    print(f"[OK] {base_path}: {val_count} referencias actualizadas de val a train")
    return True

# Actualizar ambos datasets V2
print("="*70)
print("ACTUALIZANDO SPLITS DE DATASETS V2")
print("="*70)

datasets = [
    'data/wlasl100_v2',
    'data/wlasl300_v2'
]

for dataset in datasets:
    if os.path.exists(dataset):
        update_splits(dataset)
    else:
        print(f"[WARN] Dataset no encontrado: {dataset}")

print("\n[OK] Proceso completado")
