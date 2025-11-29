# Comparación de Experimentos: V1 vs V2

## 📋 Resumen

Este documento describe las diferencias entre las configuraciones V1 (baseline) y V2 (experimental) para el entrenamiento de VideoMAE en WLASL100/300.

---

## 📊 Configuración de Datasets

### **WLASL100 - 100 Clases**

#### **V1 - Configuración Original (Baseline)**

| Split | Videos | Propósito |
|-------|---------|-----------|
| Train | 807 | Entrenamiento |
| Val | 194 | Validación durante entrenamiento |
| Test | 117 | Evaluación final |
| **Total** | **1,118** | |

**Ubicación:** `data/wlasl100/`

**Scripts:** `scripts/`

#### **V2 - Configuración Experimental (Train+Val Combinados)**

| Split | Videos | Propósito |
|-------|---------|-----------|
| Train | 1,001 (807+194) | Entrenamiento (train+val combinados) |
| Val | 117 (test) | Validación durante entrenamiento |
| Test | 117 (test) | Evaluación final |
| **Total** | **1,118** | |

**Ubicación:** `data/wlasl100_v2/`

**Scripts:** `scripts_v2/`

---

### **WLASL300 - 300 Clases**

#### **V1 - Configuración Original (Baseline)**

| Split | Videos | Propósito |
|-------|---------|-----------|
| Train | 1,959 | Entrenamiento |
| Val | 557 | Validación durante entrenamiento |
| Test | 271 | Evaluación final |
| **Total** | **2,787** | |

**Ubicación:** `data/wlasl300/`

**Scripts:** `scripts/`

#### **V2 - Configuración Experimental (Train+Val Combinados)**

| Split | Videos | Propósito |
|-------|---------|-----------|
| Train | 2,516 (1,959+557) | Entrenamiento (train+val combinados) |
| Val | 271 (test) | Validación durante entrenamiento |
| Test | 271 (test) | Evaluación final |
| **Total** | **2,787** | |

**Ubicación:** `data/wlasl300_v2/`

**Scripts:** `scripts_v2/`

---

## ⚙️ Hiperparámetros

| Parámetro | V1 (Baseline) | V2 (Experimental) | Cambio |
|-----------|---------------|-------------------|--------|
| **Batch Size** | 16 | 6 | ⬇️ Reducido (66% menos) |
| **Max Epochs** | 30 | 30 | = Sin cambio |
| **Learning Rate** | 1e-4 | 1e-5 | ⬇️ Reducido (10x menor) |
| **Weight Decay** | 0.05 | 0.0 | ❌ Eliminado |
| **Label Smoothing** | 0.1 | 0.0 | ❌ Desactivado |
| **Class Weighted** | True | False | ❌ Desactivado |
| **Patience** | 5 | 10 | ⬆️ Aumentado (2x) |
| **Warmup Ratio** | 0.1 (10%) | 0.1 (10%) | = Sin cambio |
| **Min LR** | 1e-6 | 1e-6 | = Sin cambio |
| **Gradient Clip** | 1.0 | 1.0 | = Sin cambio |

---

## 🎯 Estrategias de Entrenamiento

### **V1 - Baseline**

- ✅ **Regularización agresiva:**
  - Weight decay: 0.05
  - Label smoothing: 0.1
  - Class weighting activo

- ✅ **Early stopping conservador:**
  - Patience: 5 epochs
  - Basado en validation loss

- ✅ **Batch size estándar:** 16

- ✅ **Learning rate estándar:** 1e-4

### **V2 - Experimental**

- 🔬 **Sin regularización explícita:**
  - Weight decay: 0.0 (confiando en la arquitectura)
  - Label smoothing: 0.0 (confiando en más datos)
  - Class weighting desactivado

- 🔬 **Early stopping más paciente:**
  - Patience: 10 epochs
  - Permite más tiempo para convergencia

- 🔬 **Batch size reducido:** 6
  - Mayor número de actualizaciones de pesos
  - Posible mejor generalización

- 🔬 **Learning rate más bajo:** 1e-5
  - Aprendizaje más fino
  - Complementa mayor cantidad de datos

---

## 🔄 Frame Sampling

**Ambas versiones usan sampling uniforme:**
- Método: `np.linspace(0, frame_count-1, 16)`
- 16 frames uniformemente espaciados
- Cubre toda la duración del video

---

## 🧪 Hipótesis y Justificación

### **¿Por qué V2?**

**Objetivo:** Maximizar datos de entrenamiento para potencialmente mejorar el desempeño del modelo.

#### **1. Más datos de entrenamiento (807 → 1,001)**
- ✅ **Hipótesis:** Más ejemplos → mejor generalización
- ⚠️ **Riesgo:** Sin validación independiente → posible overfitting a test set

#### **2. Eliminación de regularización**
- ✅ **Hipótesis:** Con más datos, menos necesidad de regularización artificial
- 📊 **Razonamiento:** Class weights y label smoothing son útiles con datasets pequeños

#### **3. Batch size reducido (16 → 6)**
- ✅ **Hipótesis:** Más actualizaciones de gradientes → mejor optimización
- 📊 **Razonamiento:** Con más datos, batch size pequeño puede ayudar a explorar mejor el espacio

#### **4. Learning rate reducido (1e-4 → 1e-5)**
- ✅ **Hipótesis:** LR bajo + más datos = ajuste más fino
- 📊 **Razonamiento:** Evita sobrepasar mínimos locales buenos

#### **5. Patience aumentado (5 → 10)**
- ✅ **Hipótesis:** Más datos requieren más tiempo para converger
- 📊 **Razonamiento:** Evita detener prematuramente el entrenamiento

---

## ⚠️ Consideraciones Importantes

### **Ventajas de V2:**
- ✅ **24% más datos de entrenamiento** (807 → 1,001)
- ✅ Aprovecha todos los videos disponibles
- ✅ Potencial de mejor generalización

### **Desventajas de V2:**
- ❌ **No hay validación independiente**
  - El test set se usa para validación durante entrenamiento
  - Viola el principio de "unseen data"
  - Riesgo de overfitting al test set

- ❌ **No se pueden ajustar hiperparámetros**
  - Cualquier tuning usaría el test set
  - Los resultados finales pueden estar sesgados

- ❌ **Entrenamiento más lento**
  - Batch size pequeño → más iteraciones
  - Patience alto → más epochs potenciales

---

## 📁 Estructura de Archivos

```
AtiendeSenas-MVP/
│
├── data/
│   ├── wlasl100/                    # V1 - Dataset original
│   │   ├── splits/
│   │   │   ├── train_split.txt      (807 videos)
│   │   │   ├── val_split.txt        (194 videos)
│   │   │   └── test_split.txt       (117 videos)
│   │   ├── dataset/                 (videos organizados)
│   │   ├── nslt_100.json
│   │   └── WLASL_v0.3.json
│   │
│   ├── wlasl100_v2/                 # V2 - Dataset experimental WLASL100
│   │   ├── splits/
│   │   │   ├── train_split.txt      (1,001 videos = train+val)
│   │   │   ├── val_split.txt        (117 videos = test)
│   │   │   └── test_split.txt       (117 videos = test)
│   │   ├── dataset/ → symlink to wlasl100/dataset/
│   │   ├── videos/ → symlink to wlasl100/videos/
│   │   ├── nslt_100.json
│   │   └── WLASL_v0.3.json
│   │
│   ├── wlasl300/                    # V1 - Dataset original WLASL300
│   │   ├── splits/
│   │   │   ├── train_split.txt      (1,959 videos)
│   │   │   ├── val_split.txt        (557 videos)
│   │   │   └── test_split.txt       (271 videos)
│   │   ├── dataset/                 (videos organizados)
│   │   ├── nslt_300.json
│   │   ├── gloss_to_id.json
│   │   └── WLASL_v0.3_300.json
│   │
│   └── wlasl300_v2/                 # V2 - Dataset experimental WLASL300
│       ├── splits/
│       │   ├── train_split.txt      (2,516 videos = train+val)
│       │   ├── val_split.txt        (271 videos = test)
│       │   └── test_split.txt       (271 videos = test)
│       ├── dataset/ → symlink to wlasl300/dataset/
│       ├── videos/ → symlink to wlasl300/videos/
│       ├── nslt_300.json
│       ├── gloss_to_id.json
│       └── WLASL_v0.3_300.json
│
├── scripts/                         # V1 - Scripts originales
│   ├── train.py
│   ├── test.py
│   └── WLASLDataset.py
│
├── scripts_v2/                      # V2 - Scripts experimentales
│   ├── train.py
│   ├── test.py
│   └── WLASLDataset.py
│
├── models/                          # V1 - Modelos y checkpoints
│   └── checkpoints/
│       └── run_YYYYMMDD_HHMMSS/
│
├── models_v2/                       # V2 - Modelos y checkpoints
│   └── checkpoints/
│       └── run_YYYYMMDD_HHMMSS/
│
├── runs/                            # V1 - TensorBoard logs
│   └── run_YYYYMMDD_HHMMSS/
│
├── runs_v2/                         # V2 - TensorBoard logs
│   └── run_YYYYMMDD_HHMMSS/
│
├── evaluation_results/              # V1 - Resultados evaluación
│   ├── test_results_*.json
│   ├── test_results_*.txt
│   └── *.png
│
└── evaluation_results_v2/           # V2 - Resultados evaluación
    ├── test_results_*.json
    ├── test_results_*.txt
    └── *.png
```

---

## 🚀 Cómo Usar

### **Entrenar Modelo V1 (Baseline)**

```bash
# WLASL100 - Desde el directorio raíz del proyecto
cd scripts
python train.py --dataset wlasl100

# WLASL300 - Desde el directorio raíz del proyecto
cd scripts
python train.py --dataset wlasl300
```

### **Entrenar Modelo V2 (Experimental)**

```bash
# WLASL100_V2 - Desde el directorio raíz del proyecto
cd scripts_v2
python train.py --dataset wlasl100_v2

# WLASL300_V2 - Desde el directorio raíz del proyecto
cd scripts_v2
python train.py --dataset wlasl300_v2
```

### **Evaluar Modelo V1**

```bash
cd scripts
python test.py --list-runs              # Ver runs disponibles
python test.py --run-id 1               # Evaluar run específico

# Para WLASL300, especificar base_path si es necesario
python test.py --run-id 1 --base_path data/wlasl300
```

### **Evaluar Modelo V2**

```bash
cd scripts_v2
python test.py --list-runs              # Ver runs disponibles V2
python test.py --run-id 1               # Evaluar run específico V2

# Para WLASL300_V2, el base_path se detecta automáticamente del checkpoint
python test.py --run-id 1
```

---

## 📈 Comparación de Resultados (A completar después del entrenamiento)

| Métrica | V1 (Baseline) | V2 (Experimental) | Diferencia |
|---------|---------------|-------------------|------------|
| **Test Accuracy** | TBD | TBD | TBD |
| **Top-3 Accuracy** | TBD | TBD | TBD |
| **Top-5 Accuracy** | TBD | TBD | TBD |
| **Precision (Macro)** | TBD | TBD | TBD |
| **Recall (Macro)** | TBD | TBD | TBD |
| **F1-Score (Macro)** | TBD | TBD | TBD |
| **Val Loss Final** | TBD | TBD | TBD |
| **Epochs Completados** | TBD | TBD | TBD |
| **Tiempo de Entrenamiento** | TBD | TBD | TBD |

---

## 🎓 Recomendaciones

### **Cuándo usar V1:**
- ✅ Para experimentar con hiperparámetros
- ✅ Para validación científica rigurosa
- ✅ Cuando necesitas validación independiente

### **Cuándo usar V2:**
- ✅ Como experimento final después de optimizar V1
- ✅ Cuando necesitas maximizar uso de datos disponibles
- ✅ Para comparar impacto de más datos vs regularización

### **Flujo de trabajo recomendado:**
1. **Fase 1:** Experimentar con V1 para encontrar mejores hiperparámetros
2. **Fase 2:** Entrenar modelo final con V2 usando configuración optimizada
3. **Fase 3:** Comparar resultados finales entre V1 y V2

---

## 📝 Notas Adicionales

- Ambas configuraciones mantienen los mismos videos (no hay duplicados)
- Los enlaces simbólicos en V2 ahorran espacio en disco
- Los resultados se guardan en carpetas separadas para evitar confusión
- El frame sampling es idéntico en ambas versiones

---

**Fecha de creación:** 2025-01-29

**Autor:** Rafael Ovalle - Tesis UNAB
