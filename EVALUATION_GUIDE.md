# Guía de Evaluación - test.py

**Script**: `scripts/test.py`
**Propósito**: Evaluar el modelo entrenado en el test set de WLASL100 con métricas detalladas

---

## 🎯 Características

### Métricas Calculadas

1. **Accuracy Total**
2. **Precision, Recall, F1-Score** (Macro y Weighted)
3. **Top-K Accuracy** (Top-1, Top-3, Top-5)
4. **Métricas por Clase**:
   - Accuracy por clase
   - Precision por clase
   - Recall por clase
   - F1-Score por clase
   - Support (número de muestras)

### Visualizaciones Generadas

1. **Matriz de Confusión** (heatmap)
   - Si hay >20 clases, muestra solo las 20 con más muestras

2. **Gráfico de Performance por Clase**
   - Top 20 mejores clases
   - Top 20 peores clases

### Reportes Generados

1. **JSON** (`test_results_YYYYMMDD_HHMMSS.json`)
   - Métricas globales
   - Top-K accuracies
   - Métricas por clase
   - Información del checkpoint

2. **TXT** (`test_results_YYYYMMDD_HHMMSS.txt`)
   - Formato legible para humanos
   - Top 10 mejores clases
   - Top 10 peores clases

---

## 🚀 Uso

### Uso Básico (Requerido: checkpoint)

```bash
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/best_model.pt
```

### Uso Completo (con opciones)

```bash
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/best_model.pt \
  --batch_size 16 \
  --num_workers 4 \
  --output_dir evaluation_results \
  --device cuda
```

### En Google Colab

```python
!python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/best_model.pt \
  --batch_size 16 \
  --num_workers 4
```

---

## 📊 Output Esperado

### 1. Consola

```
======================================================================
        EVALUACIÓN EN TEST SET - VideoMAE WLASL100
======================================================================
Device: cuda
Checkpoint: models/checkpoints/run_20251125_190000/best_model.pt
Batch size: 16
======================================================================

[INFO] Cargando checkpoint: models/checkpoints/run_20251125_190000/best_model.pt
[INFO] Modelo: MCG-NJU/videomae-base-finetuned-kinetics
[INFO] Num classes: 100
[INFO] Checkpoint epoch: 23
[INFO] Val Accuracy: 68.45
[INFO] Val Loss: 0.8234

[INFO] Cargando test dataset...
[INFO] Test samples: 117
[INFO] Test batches: 8

[INFO] Evaluando en test set...
Evaluación: 100%|███████████████████| 8/8 [00:35<00:00]

[INFO] Calculando métricas...

======================================================================
                            RESULTADOS
======================================================================
Total Accuracy:      65.81%
Precision (Macro):   64.23%
Recall (Macro):      63.87%
F1-Score (Macro):    63.52%
======================================================================

                        TOP-K ACCURACIES
======================================================================
TOP_1: 65.81%
TOP_3: 84.62%
TOP_5: 91.45%
======================================================================

[INFO] Generando visualizaciones...
[SAVE] Confusion matrix guardada en: evaluation_results/confusion_matrix.png
[SAVE] Class performance guardado en: evaluation_results/class_performance.png

[INFO] Guardando resultados...
[SAVE] JSON results guardado en: evaluation_results/test_results_20251125_192030.json
[SAVE] TXT results guardado en: evaluation_results/test_results_20251125_192030.txt

======================================================================
                      EVALUACIÓN COMPLETADA
======================================================================
Resultados guardados en: evaluation_results/
======================================================================
```

---

### 2. Archivos Generados

```
evaluation_results/
├── test_results_20251125_192030.json    # Métricas en formato JSON
├── test_results_20251125_192030.txt     # Reporte legible
├── confusion_matrix.png                 # Heatmap de matriz de confusión
└── class_performance.png                # Mejores/Peores clases
```

---

## 📄 Ejemplo de Reporte TXT

```
======================================================================
EVALUACIÓN EN TEST SET - VideoMAE WLASL100
======================================================================

Fecha: 20251125_192030
Checkpoint: models/checkpoints/run_20251125_190000/best_model.pt
Epoch: 23
Val Accuracy: 68.45
Val Loss: 0.8234

======================================================================
MÉTRICAS GLOBALES
======================================================================

Total Accuracy:      65.81%

Precision (Macro):   64.23%
Recall (Macro):      63.87%
F1-Score (Macro):    63.52%

Precision (Weighted):66.54%
Recall (Weighted):   65.81%
F1-Score (Weighted): 65.23%

======================================================================
TOP-K ACCURACIES
======================================================================

TOP_1: 65.81%
TOP_3: 84.62%
TOP_5: 91.45%

======================================================================
MÉTRICAS POR CLASE (Top 10 Mejores)
======================================================================

Clase    Acc(%)   Prec(%)  Rec(%)   F1(%)    Support
----------------------------------------------------------------------
42       100.00   100.00   100.00   100.00   3
17       100.00   100.00   100.00   100.00   2
8        100.00   100.00   100.00   100.00   2
56       100.00   100.00   100.00   100.00   1
23       100.00   100.00   100.00   100.00   1
71       100.00   100.00   100.00   100.00   1
34       100.00   100.00   100.00   100.00   1
12       100.00   100.00   100.00   100.00   1
89       100.00   100.00   100.00   100.00   1
55       100.00   100.00   100.00   100.00   1

======================================================================
MÉTRICAS POR CLASE (Top 10 Peores)
======================================================================

Clase    Acc(%)   Prec(%)  Rec(%)   F1(%)    Support
----------------------------------------------------------------------
78       0.00     0.00     0.00     0.00     2
91       0.00     0.00     0.00     0.00     1
45       0.00     0.00     0.00     0.00     1
13       0.00     0.00     0.00     0.00     1
67       0.00     0.00     0.00     0.00     1
88       33.33    50.00    33.33    40.00    3
26       33.33    100.00   33.33    50.00    3
52       50.00    50.00    50.00    50.00    2
39       50.00    100.00   50.00    66.67    2
74       50.00    100.00   50.00    66.67    2
```

---

## 📈 Ejemplo de JSON Output

```json
{
  "timestamp": "20251125_192030",
  "checkpoint_info": {
    "checkpoint_path": "models/checkpoints/run_20251125_190000/best_model.pt",
    "epoch": 23,
    "val_acc": 68.45,
    "val_loss": 0.8234
  },
  "overall_metrics": {
    "total_accuracy": 65.81,
    "precision_macro": 64.23,
    "recall_macro": 63.87,
    "f1_macro": 63.52,
    "precision_weighted": 66.54,
    "recall_weighted": 65.81,
    "f1_weighted": 65.23
  },
  "top_k_accuracies": {
    "top_1": 65.81,
    "top_3": 84.62,
    "top_5": 91.45
  },
  "per_class_metrics": {
    "accuracy": [100.0, 75.0, 80.0, ...],
    "precision": [100.0, 80.0, 85.0, ...],
    "recall": [100.0, 75.0, 80.0, ...],
    "f1": [100.0, 77.5, 82.4, ...],
    "support": [3, 4, 5, ...]
  }
}
```

---

## 🖼️ Visualizaciones

### 1. Confusion Matrix (`confusion_matrix.png`)

- Heatmap de matriz de confusión
- Si >20 clases: muestra solo las 20 con más muestras en test set
- Eje X: Clase predicha
- Eje Y: Clase real
- Color: Azul (más oscuro = más muestras)

### 2. Class Performance (`class_performance.png`)

- **Izquierda**: Top 20 mejores clases (barras verdes)
- **Derecha**: Top 20 peores clases (barras rojas)
- Muestra accuracy por clase
- Solo incluye clases con muestras en test set

---

## 🔧 Argumentos CLI Completos

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--checkpoint_path` | **REQUERIDO** | Ruta al checkpoint (.pt) |
| `--base_path` | `data/wlasl100` | Ruta base del dataset |
| `--batch_size` | `16` | Batch size para evaluación |
| `--num_workers` | `4` | Workers para DataLoader |
| `--output_dir` | `evaluation_results` | Directorio de salida |
| `--device` | `cuda` (si disponible) | Device (cuda/cpu) |

---

## 📝 Casos de Uso Comunes

### 1. Evaluar el mejor modelo de un run

```bash
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/best_model.pt
```

### 2. Evaluar un checkpoint específico (epoch 20)

```bash
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/checkpoint_epoch_20.pt
```

### 3. Comparar múltiples checkpoints

```bash
# Run 1
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/best_model.pt \
  --output_dir evaluation_results/run1

# Run 2
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_200000/best_model.pt \
  --output_dir evaluation_results/run2
```

### 4. Evaluación rápida (batch size mayor)

```bash
python scripts/test.py \
  --checkpoint_path models/checkpoints/run_20251125_190000/best_model.pt \
  --batch_size 32
```

---

## 🧪 Integración con TensorBoard (Opcional)

Si deseas agregar los resultados a TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

# Después de ejecutar test.py y obtener métricas
writer = SummaryWriter(log_dir='runs/test_evaluation')
writer.add_scalar('Test/Accuracy', test_accuracy, 0)
writer.add_scalar('Test/Precision', test_precision, 0)
writer.add_scalar('Test/Recall', test_recall, 0)
writer.add_scalar('Test/F1', test_f1, 0)
writer.close()
```

---

## 🐛 Troubleshooting

### Error: "checkpoint_path is required"

**Solución**: Debes especificar `--checkpoint_path`

```bash
python scripts/test.py --checkpoint_path models/checkpoints/run_*/best_model.pt
```

### Error: "Test dataset not found"

**Solución**: Verifica que `data/wlasl100/dataset/test/` existe y tiene videos

```bash
ls data/wlasl100/dataset/test/*.mp4 | wc -l  # Debe mostrar 117 videos
```

### Advertencia: "config.json no encontrado"

**Solución**: El script usará defaults, pero es mejor tener config.json en la misma carpeta que el checkpoint

### Gráficos no se generan

**Solución**: Instala matplotlib y seaborn

```bash
pip install matplotlib seaborn
```

---

## 📊 Interpretación de Resultados

### Top-K Accuracy

- **Top-1**: Accuracy estándar (predicción correcta en primer lugar)
- **Top-3**: Clase correcta está en las 3 predicciones con mayor probabilidad
- **Top-5**: Clase correcta está en las 5 predicciones con mayor probabilidad

**Ejemplo**:
- Top-1: 65.81% → El modelo acierta directamente en 65.81% de casos
- Top-3: 84.62% → En 84.62% de casos, la clase correcta está entre las 3 más probables
- Top-5: 91.45% → En 91.45% de casos, la clase correcta está entre las 5 más probables

### Macro vs Weighted

- **Macro**: Promedio simple (todas las clases pesan igual)
  - Útil cuando te importa el rendimiento en todas las clases por igual

- **Weighted**: Promedio ponderado por número de muestras
  - Útil cuando las clases están desbalanceadas
  - Refleja mejor el rendimiento global

---

## 🎯 Próximos Pasos Después de Evaluar

1. **Si Test Acc < Val Acc** (ej: Test 60%, Val 68%)
   - Normal: el test set es independiente
   - Si diferencia >10%: posible overfitting

2. **Analizar clases problemáticas**
   - Ver peores clases en reporte TXT
   - Analizar confusion matrix para ver confusiones frecuentes
   - Considerar aumentar datos de clases con peor rendimiento

3. **Mejorar modelo**
   - Si hay clases con 0% accuracy: revisar videos de esas clases
   - Si muchas confusiones entre clases similares: considerar augmentations específicas
   - Comparar con otras configuraciones de entrenamiento

---

**Autor**: Rafael Ovalle - Tesis UNAB
**Versión**: test.py v1.0
