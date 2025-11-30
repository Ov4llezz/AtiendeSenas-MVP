"""
Fix para crear estructura de carpetas V2 consistente
Ejecutar ANTES de entrenar en Colab
"""
import os
from pathlib import Path

# Detectar si estamos en Colab o local
if os.path.exists('/content'):
    BASE = '/content/AtiendeSenas-MVP'
    print("[INFO] Entorno: Google Colab")
else:
    BASE = '.'
    print("[INFO] Entorno: Local")

# Crear estructura unificada
folders = [
    f'{BASE}/models/v2/wlasl100/checkpoints',
    f'{BASE}/models/v2/wlasl100_v2/checkpoints',
    f'{BASE}/models/v2/wlasl300/checkpoints',
    f'{BASE}/runs/v2/wlasl100',
    f'{BASE}/runs/v2/wlasl100_v2',
    f'{BASE}/runs/v2/wlasl300',
    f'{BASE}/results/v2',
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✓ Creado: {folder}")

# Verificar
print("\n[INFO] Verificando estructura:")
if os.path.exists('/content'):
    os.system(f'tree -L 3 {BASE}/models 2>/dev/null || find {BASE}/models -type d | head -20')
else:
    os.system(f'dir /s /b {BASE}\\models\\v2 2>nul | findstr checkpoint')

print("\n✓ Estructura V2 creada correctamente")
print("Ahora los checkpoints aparecerán en el explorador de archivos")
