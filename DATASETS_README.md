# Guía de Uso: Datasets WLASL100 y WLASL300

Este documento describe cómo utilizar los datasets WLASL100 y WLASL300 con el pipeline de entrenamiento y evaluación.

## Tabla de Contenidos

- [Estructura de Datasets](#estructura-de-datasets)
- [Entrenamiento](#entrenamiento)
- [Evaluación](#evaluación)
- [Ejemplos de Comandos](#ejemplos-de-comandos)
- [Configuración Avanzada](#configuración-avanzada)

---

## Estructura de Datasets

### WLASL100

- **Ubicación**: `data/wlasl100/`
- **Clases**: 100 señas
- **Videos**: ~1,000 videos
- **Estructura**:
  ```
  data/wlasl100/
  ├── dataset/
  │   ├── train/          # Videos de entrenamiento
  │   ├── val/            # Videos de validación
  │   └── test/           # Videos de prueba
  ├── splits/
  │   ├── train_split.txt
  │   ├── val_split.txt
  │   └── test_split.txt
  ├── nslt_100.json       # Mapeo video_id → clase
  └── WLASL_v0.3.json     # Metadata del dataset
  ```

### WLASL300

- **Ubicación**: `data/wlasl300/`
- **Clases**: 300 señas (las 300 glosas con más videos del dataset completo)
- **Videos**: 2,790 videos totales
  - Train: 1,960 videos (70.3%)
  - Val: 558 videos (20.0%)
  - Test: 272 videos (9.7%)
- **Estructura**:
  ```
  data/wlasl300/
  ├── dataset/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── splits/
  │   ├── train_split.txt
  │   ├── val_split.txt
  │   └── test_split.txt
  ├── nslt_300.json       # Mapeo video_id → clase (0-299)
  ├── WLASL_v0.3_300.json # Metadata del dataset
  └── gloss_to_id.json    # Mapeo glosa → ID de clase
  ```

---

## Entrenamiento

### Entrenar con WLASL100 (Default)

```bash
python scripts/train.py
```

O explícitamente:

```bash
python scripts/train.py --dataset wlasl100
```

### Entrenar con WLASL300

```bash
python scripts/train.py --dataset wlasl300
```

### Opciones Comunes de Entrenamiento

```bash
python scripts/train.py \
  --dataset wlasl300 \
  --batch_size 16 \
  --max_epochs 30 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --label_smoothing 0.1 \
  --early_stopping true \
  --patience 5
```

### Parámetros Importantes

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset a usar: `wlasl100` o `wlasl300` | `wlasl100` |
| `--batch_size` | Tamaño del batch | `16` |
| `--max_epochs` | Número máximo de epochs | `30` |
| `--lr` | Learning rate inicial | `1e-4` |
| `--weight_decay` | Regularización L2 | `0.05` |
| `--label_smoothing` | Label smoothing | `0.1` |
| `--class_weighted` | Usar pesos por clase | `true` |
| `--early_stopping` | Activar early stopping | `true` |
| `--patience` | Epochs sin mejora antes de parar | `5` |

---

## Evaluación

El script de evaluación automáticamente detecta el dataset utilizado durante el entrenamiento leyendo el `config.json` guardado con el modelo.

### Listar Runs Disponibles

```bash
python scripts/test.py --list-runs
```

Esto mostrará todos los runs de entrenamiento con su información:
- Run ID
- Epoch
- Val Accuracy
- Val Loss
- Número de clases (100 o 300)

### Evaluar Modelo Específico

#### Opción 1: Por Run ID

```bash
python scripts/test.py --run-id 1
```

#### Opción 2: Por Ruta de Checkpoint

```bash
python scripts/test.py --checkpoint_path models/checkpoints/run_20231127_153045/best_model.pt
```

### Resultados de Evaluación

El script genera:
- **Métricas generales**: Accuracy, Precision, Recall, F1-score
- **Top-K Accuracies**: Top-1, Top-3, Top-5
- **Matriz de confusión**: Guardada como imagen PNG
- **Reporte detallado**: JSON y TXT con métricas por clase
- **Ubicación**: `evaluation_results/`

---

## Ejemplos de Comandos

### Ejemplo 1: Entrenamiento Rápido con WLASL100

```bash
python scripts/train.py \
  --dataset wlasl100 \
  --batch_size 16 \
  --max_epochs 10 \
  --lr 1e-4
```

### Ejemplo 2: Entrenamiento Completo con WLASL300

```bash
python scripts/train.py \
  --dataset wlasl300 \
  --batch_size 12 \
  --max_epochs 50 \
  --lr 5e-5 \
  --weight_decay 0.1 \
  --label_smoothing 0.15 \
  --early_stopping true \
  --patience 7
```

### Ejemplo 3: Reanudar Entrenamiento

```bash
python scripts/train.py \
  --dataset wlasl300 \
  --resume models/checkpoints/run_20231127_153045/checkpoint_epoch_15.pt
```

### Ejemplo 4: Evaluar el Último Run

```bash
python scripts/test.py --run-id 1
```

### Ejemplo 5: Evaluar con Batch Size Diferente

```bash
python scripts/test.py \
  --run-id 1 \
  --batch_size 32
```

---

## Configuración Avanzada

### Cambiar el Modelo Base

Por defecto se usa VideoMAE pre-entrenado en Kinetics. Puedes cambiar el modelo:

```bash
python scripts/train.py \
  --dataset wlasl300 \
  --model_name MCG-NJU/videomae-large
```

### Ajustar Data Augmentation

Las augmentations están definidas en `WLASLDataset.py`. Para ajustarlas, modifica:

```python
# Para training
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ...
])
```

### Desactivar Class Weighting

```bash
python scripts/train.py \
  --dataset wlasl300 \
  --class_weighted false
```

### Cambiar Scheduler

El scheduler está configurado en `train.py`. Por defecto usa:
- Warmup: 10% de los steps
- Cosine Annealing hasta `min_lr=1e-6`

Para ajustar:

```bash
python scripts/train.py \
  --dataset wlasl300 \
  --warmup_ratio 0.15 \
  --min_lr 5e-7
```

---

## Notas Importantes

### Compatibilidad Automática

El pipeline detecta automáticamente:
- El número de clases (100 o 300) basándose en `--dataset`
- El `base_path` correcto (`data/wlasl100` o `data/wlasl300`)
- Los archivos JSON correspondientes (`nslt_100.json` o `nslt_300.json`)

### Archivos de Configuración

Cada entrenamiento guarda automáticamente:
- `config.json`: Configuración completa del entrenamiento
- `best_model.pt`: Mejor modelo según validation loss
- `checkpoint_epoch_X.pt`: Checkpoints cada N epochs

### Logs de TensorBoard

Ver progreso del entrenamiento:

```bash
tensorboard --logdir runs/
```

Luego abre: http://localhost:6006

---

## Creación del Dataset WLASL300

Si necesitas regenerar el dataset WLASL300 desde el dataset completo:

```bash
python scripts/create_wlasl300.py
```

Este script:
1. Lee el dataset completo de `C:\Users\ov4ll\Desktop\TESIS\WLASL_full`
2. Selecciona las 300 glosas con más videos
3. Crea la estructura en `data/wlasl300`
4. Genera todos los archivos JSON necesarios
5. Crea los splits train/val/test

---

## Solución de Problemas

### Error: "No se encontraron muestras para Split=train"

- Verifica que existan los videos en `data/wlaslXXX/dataset/train/`
- Verifica que exista `splits/train_split.txt`
- Verifica que `nslt_XXX.json` tenga los video IDs correctos

### Error: "num_labels mismatch"

El modelo se ajusta automáticamente. Si persiste:

```bash
python scripts/train.py --dataset wlasl300  # Asegúrate de especificar el dataset correcto
```

### Problemas de Memoria GPU

Reduce el batch size:

```bash
python scripts/train.py --dataset wlasl300 --batch_size 8
```

O usa gradient accumulation (modifica `train.py`).

---

## Contacto

Para preguntas o problemas:
- Autor: Rafael Ovalle
- Proyecto: Tesis UNAB - Reconocimiento de Lengua de Señas
