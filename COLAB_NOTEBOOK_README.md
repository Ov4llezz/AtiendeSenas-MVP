# 📓 Guía Completa del Notebook de Google Colab

## 🎯 VideoMAE WLASL - Training & Evaluation Complete Pipeline

Este notebook proporciona un pipeline completo para entrenar y evaluar modelos VideoMAE en los datasets WLASL100/WLASL300 para reconocimiento de lengua de señas.

---

## 📁 Archivos Generados

### 1. **VideoMAE_WLASL_Training_Complete.ipynb**
- Notebook principal para Google Colab
- Secciones 1-4: Setup, configuración, preparación de datos y entrenamiento

### 2. **VideoMAE_Evaluation_Section.txt**
- Secciones 5-8: Evaluación completa, visualizaciones y exportación
- Copiar y pegar estas celdas al final del notebook principal

---

## 🚀 Cómo Usar el Notebook

### **Paso 1: Preparar Datos en Google Drive**

1. Sube tu dataset WLASL a Google Drive en la siguiente estructura:

```
MyDrive/
└── TESIS_WLASL/
    └── data/
        ├── wlasl100/          # Dataset WLASL100 V1
        │   ├── splits/
        │   │   ├── train_split.txt
        │   │   ├── val_split.txt
        │   │   └── test_split.txt
        │   ├── dataset/
        │   │   ├── train/     # Videos de entrenamiento
        │   │   ├── val/       # Videos de validación
        │   │   └── test/      # Videos de test
        │   ├── nslt_100.json
        │   └── WLASL_v0.3.json
        │
        ├── wlasl100_v2/       # Dataset WLASL100 V2 (opcional)
        ├── wlasl300/          # Dataset WLASL300 V1 (opcional)
        └── wlasl300_v2/       # Dataset WLASL300 V2 (opcional)
```

### **Paso 2: Abrir el Notebook en Colab**

1. Sube `VideoMAE_WLASL_Training_Complete.ipynb` a Google Drive
2. Abre con Google Colab
3. Asegúrate de tener GPU activada:
   - **Runtime** → **Change runtime type** → **GPU** (T4, V100, o A100)

### **Paso 3: Agregar Sección de Evaluación**

1. Abre `VideoMAE_Evaluation_Section.txt`
2. Copia todo el contenido
3. Pega al final del notebook principal creando nuevas celdas
4. Guarda el notebook

### **Paso 4: Configurar Experimento**

En la celda de "Configuración del Experimento", ajusta:

```python
# SELECCIONA TU CONFIGURACIÓN:
DATASET_TYPE = "wlasl100"  # o "wlasl300"
VERSION = "v1"             # o "v2"
```

**Opciones Disponibles:**

| DATASET_TYPE | VERSION | Description |
|--------------|---------|-------------|
| `"wlasl100"` | `"v1"` | 100 clases, train/val/test separados, regularización activa |
| `"wlasl100"` | `"v2"` | 100 clases, train+val combinados, sin regularización |
| `"wlasl300"` | `"v1"` | 300 clases, train/val/test separados, regularización activa |
| `"wlasl300"` | `"v2"` | 300 clases, train+val combinados, sin regularización |

### **Paso 5: Ejecutar el Notebook**

1. **Ejecutar todas las celdas:** Runtime → Run all
2. **Montar Google Drive** cuando se solicite
3. **Monitorear progreso:**
   - Barras de progreso en cada epoch
   - Métricas en tiempo real
   - TensorBoard (opcional)

---

## 📊 Resultados Generados

### **Archivos Automáticos:**

Todos los resultados se guardan en `MyDrive/TESIS_WLASL/`:

#### **1. Checkpoints del Modelo** (`models/{version}/{dataset}/checkpoints/run_{timestamp}/`)
- `best_model.pt` - Mejor modelo basado en val loss
- `checkpoint_epoch_X.pt` - Checkpoints cada N epochs
- `config.json` - Configuración del entrenamiento

#### **2. Logs de TensorBoard** (`runs/{version}/{dataset}/run_{timestamp}/`)
- Loss y accuracy por batch y epoch
- Learning rate schedule
- Visualizaciones en tiempo real

#### **3. Resultados de Evaluación** (`results/{version}/{dataset}/`)
- `complete_results_{timestamp}.json` - Todas las métricas en JSON
- `report_{timestamp}.txt` - Reporte legible
- `predictions_{timestamp}.csv` - Predicciones detalladas
- `training_history.csv` - Historial de entrenamiento

#### **4. Visualizaciones** (`results/{version}/{dataset}/`)
- `training_curves_{timestamp}.png` - Loss, accuracy y LR
- `confusion_matrix_{timestamp}.png` - Matriz de confusión
- `class_performance_{timestamp}.png` - Mejores y peores clases
- `accuracy_distribution_{timestamp}.png` - Distribución de accuracy
- `support_analysis_{timestamp}.png` - Análisis por número de muestras

---

## 📈 Métricas Incluidas

### **Métricas Generales:**
- ✅ **Accuracy Total**
- ✅ **Precision** (Macro y Weighted)
- ✅ **Recall** (Macro y Weighted)
- ✅ **F1-Score** (Macro y Weighted)
- ✅ **Top-K Accuracy** (K=1, 3, 5)

### **Métricas Por Clase:**
- ✅ Accuracy por clase
- ✅ Precision por clase
- ✅ Recall por clase
- ✅ F1-Score por clase
- ✅ Support (número de muestras) por clase

### **Análisis Adicionales:**
- ✅ Top 10 mejores clases
- ✅ Top 10 peores clases
- ✅ Estadísticas descriptivas (media, mediana, std, min, max)
- ✅ Análisis por umbral de support
- ✅ Matriz de confusión normalizada
- ✅ Curvas de entrenamiento

---

## ⚙️ Configuraciones de Hiperparámetros

### **V1 (Baseline):**
```python
{
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "class_weighted": True,
    "patience": 5,
    "max_epochs": 30
}
```

**Uso:** Experimentación, tuning, validación científica

### **V2 (Experimental):**
```python
{
    "batch_size": 6,
    "lr": 1e-5,
    "weight_decay": 0.0,
    "label_smoothing": 0.0,
    "class_weighted": False,
    "patience": 10,
    "max_epochs": 30
}
```

**Uso:** Modelo final con máximos datos

---

## 🔧 Personalización Avanzada

### **Modificar Hiperparámetros:**

Después de la celda de configuración, puedes sobrescribir valores:

```python
# Personalizar después de la configuración automática
CONFIG['batch_size'] = 8
CONFIG['max_epochs'] = 50
CONFIG['lr'] = 5e-5
CONFIG['patience'] = 15
```

### **Agregar Data Augmentation Personalizado:**

En la clase `WLASLVideoDataset`, modifica el transform de train:

```python
if split == "train":
    self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Más agresivo
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Agregar rotación
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

### **Cambiar Modelo Base:**

```python
CONFIG['model_name'] = "MCG-NJU/videomae-large-finetuned-kinetics"  # Usar modelo large
```

---

## 📊 Ver Resultados en TensorBoard

### **Opción 1: Directamente en el Notebook**

Ejecuta la celda final:

```python
%load_ext tensorboard
%tensorboard --logdir {CONFIG['logs_dir']}
```

### **Opción 2: TensorBoard.dev (Compartir resultados)**

```python
!tensorboard dev upload --logdir {CONFIG['logs_dir']} \
  --name "VideoMAE WLASL100 V1" \
  --description "Baseline experiment"
```

---

## 💾 Descargar Resultados

### **Descargar Todo:**

```python
# Comprimir resultados
!zip -r results_{timestamp}.zip \
    {CONFIG['results_dir']} \
    {run_checkpoint_dir} \
    {log_dir}

# Descargar
from google.colab import files
files.download(f'results_{timestamp}.zip')
```

### **Descargar Solo Mejor Modelo:**

```python
from google.colab import files
files.download(f"{run_checkpoint_dir}/best_model.pt")
```

---

## 🐛 Troubleshooting

### **Error: "Dataset no encontrado"**
- Verifica que la ruta en `CONFIG['data_root']` sea correcta
- Asegúrate de haber montado Google Drive
- Verifica que los archivos de splits existan

### **Error: "CUDA out of memory"**
- Reduce `batch_size` (ej: 4, 2)
- Reduce `num_workers` (ej: 0, 1)
- Usa GPU con más memoria (V100 o A100)

### **Error: Videos corruptos**
- El dataset automáticamente salta videos corruptos
- Verifica `corrupt_videos_{split}.txt` si existe

### **Entrenamiento muy lento**
- Verifica que estés usando GPU: `print(CONFIG['device'])`
- Reduce `num_workers` si hay cuellos de botella en I/O
- Considera usar un batch_size mayor si la memoria lo permite

---

## 📝 Ejemplo de Flujo Completo

```python
# 1. Configurar
DATASET_TYPE = "wlasl100"
VERSION = "v1"

# 2. Ejecutar todas las celdas
# Runtime → Run all

# 3. Esperar entrenamiento (30-60 min con T4)

# 4. Revisar resultados:
print(f"Test Accuracy: {metrics['total_accuracy']:.2f}%")
print(f"Top-3 Accuracy: {metrics['top_k']['top_3']:.2f}%")

# 5. Descargar modelo
from google.colab import files
files.download(f"{run_checkpoint_dir}/best_model.pt")

# 6. Descargar todos los resultados
!zip -r my_results.zip {CONFIG['results_dir']} {run_checkpoint_dir}
files.download('my_results.zip')
```

---

## 📚 Para tu Tesis

### **Secciones Recomendadas:**

1. **Metodología:**
   - Configuración de hiperparámetros (Sección 2.1)
   - Arquitectura del modelo (VideoMAE)
   - Data augmentation aplicado

2. **Resultados:**
   - Tabla de métricas generales (Sección 5.3)
   - Gráficos de curvas de entrenamiento (Sección 4.4)
   - Matriz de confusión (Sección 6.1)
   - Análisis por clase (Secciones 5.4, 5.5)

3. **Discusión:**
   - Comparación V1 vs V2
   - Análisis de clases difíciles (peores 10)
   - Impacto del número de muestras (Sección 6.4)

### **Figuras para Incluir:**
- ✅ `training_curves_{timestamp}.png`
- ✅ `confusion_matrix_{timestamp}.png`
- ✅ `class_performance_{timestamp}.png`
- ✅ `accuracy_distribution_{timestamp}.png`

### **Tablas para Incluir:**
- ✅ Métricas generales (del reporte TXT)
- ✅ Top-10 mejores y peores clases
- ✅ Comparación V1 vs V2

---

## 🎓 Recomendaciones Finales

1. **Siempre guarda tu configuración:** El notebook guarda automáticamente `config.json`
2. **Usa nombres descriptivos:** Los timestamps ayudan a organizar experimentos
3. **Documenta cambios:** Si modificas hiperparámetros, anótalos
4. **Compara resultados:** Ejecuta V1 primero, luego V2
5. **Backup en Drive:** Todo se guarda automáticamente en tu Drive

---

**¡Éxito con tu tesis!** 🎓✨

Si tienes dudas o necesitas modificaciones, revisa la documentación en `EXPERIMENTS_V1_VS_V2.md`.
