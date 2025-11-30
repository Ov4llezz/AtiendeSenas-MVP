"""
Workaround para cargar checkpoints de NumPy 2.x en NumPy 1.x
"""
import sys
import numpy as np

# Monkey patch: mapear numpy._core a numpy.core
class _CoreModule:
    """Módulo dummy que redirige numpy._core a numpy.core"""
    def __getattr__(self, name):
        # Intentar obtener de numpy.core primero
        if hasattr(np.core, name):
            return getattr(np.core, name)
        # Si no existe, intentar de numpy directamente
        if hasattr(np, name):
            return getattr(np, name)
        raise AttributeError(f"module 'numpy._core' has no attribute '{name}'")

# Registrar el módulo falso
sys.modules['numpy._core'] = _CoreModule()
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.umath'] = np.core.umath

print("[INFO] NumPy compatibility patch aplicado")

# Ahora importar torch y cargar el checkpoint
import torch

def load_checkpoint_safe(checkpoint_path, map_location='cpu'):
    """
    Carga checkpoint con compatibilidad NumPy 1.x/2.x
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        print("[OK] Checkpoint cargado exitosamente")
        return checkpoint
    except Exception as e:
        print(f"[ERROR] Error al cargar checkpoint: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"{'PROBANDO CARGA DE CHECKPOINT':^70}")
    print(f"{'='*70}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"{'='*70}\n")

    checkpoint = load_checkpoint_safe(args.checkpoint_path)

    print(f"\n{'='*70}")
    print(f"{'INFORMACIÓN DEL CHECKPOINT':^70}")
    print(f"{'='*70}")
    print(f"Keys: {list(checkpoint.keys())}")

    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"Val Accuracy: {checkpoint['val_acc']:.2f}%")
    if 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    if 'model_state_dict' in checkpoint:
        print(f"Model state dict keys: {len(checkpoint['model_state_dict'])} layers")

    print(f"{'='*70}\n")
    print("[OK] El checkpoint se puede cargar correctamente en este entorno")
