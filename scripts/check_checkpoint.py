"""
Script temporal para verificar contenido de checkpoints
"""
import torch
import sys
import glob
import os

# Buscar el último run
runs = sorted(glob.glob("models/checkpoints/run_*"), reverse=True)

if not runs:
    print("No hay runs disponibles")
    sys.exit(1)

latest_run = runs[0]
checkpoint_path = os.path.join(latest_run, "best_model.pt")

print(f"Verificando: {checkpoint_path}")
print("="*60)

if not os.path.exists(checkpoint_path):
    print("ERROR: best_model.pt no existe en este run")
    sys.exit(1)

# Cargar checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Mostrar todas las claves
print(f"\nKeys disponibles en el checkpoint:")
print("-"*60)
for key in checkpoint.keys():
    print(f"  - {key}")

# Mostrar valores importantes
print(f"\nValores:")
print("-"*60)
print(f"epoch:     {checkpoint.get('epoch', 'NO EXISTE')}")
print(f"val_acc:   {checkpoint.get('val_acc', 'NO EXISTE')}")
print(f"val_loss:  {checkpoint.get('val_loss', 'NO EXISTE')}")
print(f"train_acc: {checkpoint.get('train_acc', 'NO EXISTE')}")
print(f"train_loss:{checkpoint.get('train_loss', 'NO EXISTE')}")

print("\n" + "="*60)
