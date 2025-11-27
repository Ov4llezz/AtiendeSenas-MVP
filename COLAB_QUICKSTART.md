# 🚀 Guía Rápida: Google Colab

Esta guía te ayudará a entrenar tus modelos de reconocimiento de lengua de señas en Google Colab.

## 📋 Requisitos Previos

1. Cuenta de Google (para usar Colab)
2. Repositorio de GitHub público (para clonar el código)
3. GPU habilitada en Colab

## 🎯 Pasos Rápidos

### 1. Abrir el Notebook en Colab

**Opción A: Desde GitHub**
1. Sube el archivo `AtiendeSenas_Training_Colab.ipynb` a tu repositorio
2. Ve a [Google Colab](https://colab.research.google.com/)
3. Selecciona `File > Open notebook`
4. Ve a la pestaña `GitHub`
5. Ingresa: `Ov4llezz/AtiendeSenas-MVP`
6. Selecciona el notebook

**Opción B: Directo**
1. Descarga `AtiendeSenas_Training_Colab.ipynb`
2. Ve a [Google Colab](https://colab.research.google.com/)
3. Selecciona `File > Upload notebook`
4. Sube el archivo descargado

### 2. Habilitar GPU

**IMPORTANTE**: Antes de empezar, asegúrate de tener GPU habilitada:

1. Ve a `Runtime > Change runtime type`
2. Selecciona:
   - **Hardware accelerator**: GPU
   - **GPU type**: T4 (o la mejor disponible)
3. Haz clic en `Save`

### 3. Ejecutar el Notebook

Ejecuta las celdas en orden:

```
1. ✅ Verificar GPU
2. 📥 Clonar repositorio
3. 📦 Instalar dependencias
4. 🔍 Verificar datasets
5. 🎓 Entrenar modelo
6. 📊 Evaluar resultados
7. 📈 Visualizar métricas
```

## ⚙️ Configuración Básica

### Para WLASL100 (Recomendado para empezar):

```python
DATASET = "wlasl100"
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 5
```

### Para WLASL300 (Más clases):

```python
DATASET = "wlasl300"
BATCH_SIZE = 12  # Reduce por más clases
MAX_EPOCHS = 50
LEARNING_RATE = 5e-5
PATIENCE = 7
```

### Prueba Rápida (2 epochs):

```python
DATASET = "wlasl100"
BATCH_SIZE = 8
MAX_EPOCHS = 2
```

## 🎬 Comandos Básicos

### Entrenar

```bash
!python scripts/train.py \
    --dataset wlasl100 \
    --batch_size 16 \
    --max_epochs 30
```

### Evaluar

```bash
!python scripts/test.py --list-runs  # Ver modelos disponibles
!python scripts/test.py --run-id 1   # Evaluar modelo más reciente
```

### Verificar Datasets

```bash
!python scripts/verify_datasets.py
```

## 💾 Guardar Resultados

### Opción 1: Descargar Directamente

```python
# Comprimir resultados
!tar -czf models.tar.gz models/checkpoints/
!tar -czf results.tar.gz evaluation_results/

# Descargar desde el panel de archivos (click derecho > Download)
```

### Opción 2: Guardar en Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copiar a Drive
!cp -r models/checkpoints /content/drive/MyDrive/AtiendeSenas/
!cp -r evaluation_results /content/drive/MyDrive/AtiendeSenas/
```

## ⏱️ Tiempos Estimados

| Dataset | Epochs | GPU | Tiempo Aprox. |
|---------|--------|-----|---------------|
| WLASL100 | 30 | T4 | 2-4 horas |
| WLASL300 | 30 | T4 | 6-12 horas |
| WLASL100 | 2 (test) | T4 | 10-15 min |

## 🐛 Solución de Problemas

### ❌ "RuntimeError: CUDA out of memory"

**Solución**:
1. Reduce `BATCH_SIZE`:
   ```python
   BATCH_SIZE = 8  # o incluso 4
   ```
2. Limpia memoria GPU:
   ```python
   import gc
   import torch
   gc.collect()
   torch.cuda.empty_cache()
   ```
3. Reinicia el runtime: `Runtime > Restart runtime`

### ❌ "No GPU available"

**Solución**:
1. Ve a `Runtime > Change runtime type`
2. Selecciona GPU en "Hardware accelerator"
3. Guarda y reconecta

### ❌ Sesión se desconecta

**Causas**:
- Límite de tiempo de Colab (12h gratis)
- Inactividad prolongada

**Solución**:
- Guarda checkpoints frecuentemente
- Usa Google Drive para respaldo automático
- Considera Colab Pro para sesiones más largas

### ❌ "No module named 'transformers'"

**Solución**:
```python
!pip install transformers==4.36.0
```

### ❌ Videos no se encuentran

**Solución**:
1. Verifica que clonaste el repo correctamente
2. Ejecuta:
   ```bash
   !python scripts/verify_datasets.py
   ```
3. Revisa que los videos estén en:
   - `data/wlasl100/dataset/train/`
   - `data/wlasl300/dataset/train/`

## 📊 Monitorear Entrenamiento

### Ver Progreso en Vivo

```python
# TensorBoard (en Colab)
%load_ext tensorboard
%tensorboard --logdir runs/
```

### Ver Uso de GPU

```python
!nvidia-smi
```

## 💡 Tips y Trucos

### 1. Mantener Sesión Activa

Ejecuta esto en la consola del navegador (F12):
```javascript
function ClickConnect(){
    console.log("Keeping alive");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

### 2. Backup Automático

```python
# Ejecuta cada hora
import time
while training:
    time.sleep(3600)  # 1 hora
    !cp -r models/checkpoints /content/drive/MyDrive/backup/
```

### 3. Notificaciones

```python
# Al terminar entrenamiento
from google.colab import auth
# Configura para recibir notificaciones
```

## 📚 Recursos Adicionales

- [Documentación Completa](DATASETS_README.md)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Repositorio GitHub](https://github.com/Ov4llezz/AtiendeSenas-MVP)

## 🎯 Workflow Recomendado

1. **Primera vez**:
   - Prueba rápida (2 epochs, wlasl100)
   - Verifica que todo funciona

2. **Entrenamiento real**:
   - Empieza con wlasl100 (30 epochs)
   - Analiza resultados

3. **Experimentos**:
   - Prueba wlasl300
   - Ajusta hiperparámetros
   - Compara resultados

4. **Producción**:
   - Entrena modelo final
   - Guarda en Google Drive
   - Descarga para deployment

---

**¿Preguntas? Abre un issue en GitHub! 🚀**
