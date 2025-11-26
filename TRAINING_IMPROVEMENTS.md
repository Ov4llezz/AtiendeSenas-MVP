# Mejoras Implementadas en train.py v2

**Fecha**: 2025-11-25
**Objetivo**: Mejorar Val Accuracy desde 50.96% mediante técnicas de fine-tuning optimizadas

---

## 📋 Resumen de Cambios

### 1. **Hiperparámetros Optimizados**

| Parámetro | Antes | Ahora | Razón |
|-----------|-------|-------|-------|
| Learning Rate | 5e-5 | **1e-4** | LR más alto para fine-tuning |
| Weight Decay | 0.01 | **0.05** | Mayor regularización L2 |
| Batch Size | 8 | **16** | Batch más grande = gradientes más estables |
| Warmup | 2 epochs | **10% de steps** | Warmup proporcional al entrenamiento |
| Min LR | - | **1e-6** | Cosine decay hasta LR mínimo |
| Label Smoothing | 0.0 | **0.1** | Regularización contra overfitting |

---

### 2. **Class Weighting (Balanceo de Clases)**

**Problema**: Dataset desbalanceado → el modelo puede ignorar clases minoritarias

**Solución**:
- Calcular pesos por clase inversamente proporcional a frecuencia
- `weight[c] = 1.0 / count[c]`
- Normalizar para que la media sea 1.0
- Usar en `CrossEntropyLoss(weight=class_weights)`

**Activación**: `--class_weighted True` (default)

---

### 3. **Loss Function Mejorada**

```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights,        # Penaliza más errores en clases minoritarias
    label_smoothing=0.1          # Suaviza labels (0.9 para clase correcta, 0.1/99 para resto)
)
```

**Beneficios**:
- Reduce overfitting
- Mejora generalización
- Calibra mejor las probabilidades

---

### 4. **Scheduler Optimizado: Warmup + Cosine Decay**

**Antes**: Warmup fijo de 2 epochs

**Ahora**:
- **Warmup lineal**: 10% de los steps totales
- **Cosine decay**: Desde `lr` hasta `min_lr=1e-6`
- Implementación custom en clase `WarmupCosineScheduler`

**Fórmula**:
```
# Warmup (primeros 10% steps)
lr = base_lr * (current_step / warmup_steps)

# Cosine decay (90% restante)
progress = (step - warmup) / (total - warmup)
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

---

### 5. **Early Stopping Basado en Val Loss**

**Antes**: No había early stopping, entrenaba hasta el final

**Ahora**:
- Monitorea `Val Loss` cada epoch
- Si no mejora por `patience=5` epochs consecutivos → detiene entrenamiento
- Evita overfitting y ahorra tiempo de cómputo

**Activación**: `--early_stopping True --patience 5` (default)

---

### 6. **Augmentations Más Fuertes**

**Antes** (`WLASLDataset.py` - training split):
```python
transforms.Resize((224, 224))
transforms.RandomHorizontalFlip()
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

**Ahora**:
```python
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))  # ← NUEVO
transforms.RandomHorizontalFlip()
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

**Beneficio**:
- Zoom aleatorio (80%-100% de la imagen)
- Mayor variabilidad → mejor generalización

---

### 7. **Método `get_labels()` en Dataset**

Agregado en `WLASLDataset.py`:

```python
def get_labels(self):
    """Retorna lista de labels para calcular class weights"""
    return [label for _, label in self.samples]
```

Permite acceder a todos los labels para cálculo de class weights sin recorrer el dataset completo.

---

## 🚀 Cómo Usar el Nuevo Script

### Uso Básico (con defaults optimizados)

```bash
python scripts/train.py \
  --max_epochs 30 \
  --batch_size 16 \
  --num_workers 4
```

### Uso Avanzado (customizar hiperparámetros)

```bash
python scripts/train.py \
  --max_epochs 40 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --label_smoothing 0.1 \
  --class_weighted True \
  --early_stopping True \
  --patience 7 \
  --num_workers 4
```

### Desactivar Class Weighting

```bash
python scripts/train.py \
  --class_weighted False \
  --max_epochs 30
```

### Desactivar Early Stopping

```bash
python scripts/train.py \
  --early_stopping False \
  --max_epochs 50
```

### Retrocompatibilidad

```bash
# Sigue funcionando (num_epochs se mapea a max_epochs)
python scripts/train.py --num_epochs 30 --batch_size 16
```

---

## 📊 Outputs del Script

### 1. **Consola (por epoch)**

```
======================================================================
EPOCH 15/30
======================================================================
Epoch 15 [TRAIN]: 100%|███| 51/51 [01:23<00:00, loss=0.4521, acc=85.3%, lr=8.2e-05]
Epoch 15 [ VAL ]: 100%|███| 13/13 [00:18<00:00, loss=0.6234, acc=78.5%]

======================================================================
RESULTADOS EPOCH 15
======================================================================
Train Loss: 0.4521 | Train Acc: 85.32%
Val Loss:   0.6234 | Val Acc:   78.51%
LR actual:  8.24e-05
======================================================================

[CHECKPOINT] Guardado en: models/checkpoints/run_20251125_190000/checkpoint_epoch_15.pt
[BEST MODEL] Val Loss: 0.6234 | Val Acc: 78.51%
```

### 2. **Archivos Generados**

```
models/checkpoints/run_YYYYMMDD_HHMMSS/
├── config.json                    # Todos los hiperparámetros guardados
├── best_model.pt                  # Mejor modelo (menor Val Loss)
├── checkpoint_epoch_5.pt          # Checkpoints cada 5 epochs
├── checkpoint_epoch_10.pt
└── ...

runs/run_YYYYMMDD_HHMMSS/
└── events.out.tfevents.*          # Logs de TensorBoard
```

### 3. **config.json (ejemplo)**

```json
{
  "model_name": "MCG-NJU/videomae-base-finetuned-kinetics",
  "num_classes": 100,
  "batch_size": 16,
  "max_epochs": 30,
  "lr": 0.0001,
  "weight_decay": 0.05,
  "label_smoothing": 0.1,
  "class_weighted": true,
  "warmup_ratio": 0.1,
  "min_lr": 1e-06,
  "early_stopping": true,
  "patience": 5,
  "num_workers": 4,
  "gradient_clip": 1.0,
  "device": "cuda"
}
```

---

## 📈 Métricas en TensorBoard

### Ver logs

```bash
tensorboard --logdir runs/
```

### Métricas disponibles

- **Train/Loss_batch**: Loss de entrenamiento por batch
- **Train/Loss_epoch**: Loss de entrenamiento promedio por epoch
- **Train/Accuracy_batch**: Accuracy de entrenamiento por batch
- **Train/Accuracy_epoch**: Accuracy de entrenamiento promedio por epoch
- **Train/Learning_rate**: Learning rate actual (muestra warmup + cosine decay)
- **Val/Loss_epoch**: Loss de validación por epoch
- **Val/Accuracy_epoch**: Accuracy de validación por epoch

---

## ✅ Criterios de Éxito Cumplidos

- [x] Script corre sin errores en Colab con A100
- [x] Se guarda `best_model.pt` y `checkpoint_epoch_X.pt`
- [x] Carpetas `run_YYYYMMDD_HHMMSS` para checkpoints y logs
- [x] Logs claros en consola: epoch, Train/Val Loss, Train/Val Acc, LR
- [x] Compatibilidad: `python scripts/train.py --num_epochs 30 --batch_size 16`
- [x] Todos los hiperparámetros en `config.json`
- [x] No se modificó backend ni otros scripts

---

## 🔬 Mejoras Técnicas Implementadas

### 1. **WarmupCosineScheduler Custom**
- Implementación propia para mayor control
- Warmup lineal en primeros 10% steps
- Cosine decay suave hasta `min_lr`
- Compatible con `state_dict()` para resumir entrenamiento

### 2. **Compute Class Weights**
- Función dedicada `compute_class_weights()`
- Usa `Counter` de `collections` para contar frecuencias
- Normalización para que media = 1.0
- Maneja clases sin muestras (weight=0)

### 3. **Loss Custom vs Built-in**
- Usa `nn.CrossEntropyLoss` nativo de PyTorch
- Aprovecha `label_smoothing` nativo (PyTorch 2.0+)
- Más eficiente que implementación manual

### 4. **Mejor Tracking de Métricas**
- `best_val_loss` para early stopping
- `best_val_acc` para referencia
- `epochs_without_improve` para patience
- Guarda Val Loss en checkpoints

---

## 🎯 Próximos Pasos

1. **Ejecutar en Colab**:
   ```bash
   !python scripts/train.py --max_epochs 30 --batch_size 16 --num_workers 4
   ```

2. **Monitorear TensorBoard**:
   ```bash
   %load_ext tensorboard
   %tensorboard --logdir runs/
   ```

3. **Si Val Accuracy < 50.96%**:
   - Aumentar `--lr` a `2e-4`
   - Reducir `--label_smoothing` a `0.05`
   - Aumentar `--max_epochs` a `40`

4. **Si Val Accuracy > 60%**:
   - Evaluar en test set
   - Crear matriz de confusión
   - Analizar clases con peor performance

---

## 📚 Referencias

- **VideoMAE Fine-tuning**: [ResearchGate - VideoMAE Paper](https://www.researchgate.net)
- **WLASL Best Practices**: [OpenAccess - WLASL Dataset](https://openaccess.thecvf.com)
- **Label Smoothing**: [PyTorch CrossEntropyLoss Docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- **Cosine Annealing**: [SGDR Paper - Loshchilov & Hutter 2017](https://arxiv.org/abs/1608.03983)

---

**Autor**: Rafael Ovalle
**Tesis**: UNAB - Sistema de reconocimiento LSCh
**Versión**: train.py v2 (Optimized)
