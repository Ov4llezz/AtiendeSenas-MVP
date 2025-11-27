# 🎯 Guía de Hiperparámetros para Entrenamiento VideoMAE

## 📊 Análisis de Datasets

### WLASL100 (Local)
- **Videos totales**: 1,118 (807 train, 194 val, 117 test)
- **Clases**: 100
- **Videos/clase**: 6-19 (promedio: 11.2)
- **Desbalance**: Moderado (ratio 3.2:1)

### WLASL300
- **Videos totales**: 2,790 (1,960 train, 558 val, 272 test)
- **Clases**: 300
- **Videos/clase**: 1-16 (promedio: 9.4)
- **Desbalance**: Alto (ratio 16:1)

---

## 🎯 Recomendaciones por Dataset

### Para WLASL100 (Recomendado para empezar)

#### ✅ Configuración Conservadora (Baseline)
```python
DATASET = "wlasl100"
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
CLASS_WEIGHTED = True
WARMUP_RATIO = 0.1
PATIENCE = 5
GRADIENT_CLIP = 1.0
```

**Justificación:**
- **Batch Size 16**: Balance entre estabilidad y velocidad
  - Dataset pequeño (807 train) → batches pequeños previenen overfitting
  - 807 / 16 ≈ 50 steps por epoch (bueno para convergencia)

- **Learning Rate 1e-4**: Conservador para fine-tuning
  - Modelo pre-entrenado (VideoMAE en Kinetics) necesita ajuste fino
  - Tasa más alta podría destruir features pre-entrenadas

- **Weight Decay 0.05**: Regularización L2 moderada
  - Dataset pequeño → necesitamos prevenir overfitting
  - Valor estándar para Vision Transformers

- **Label Smoothing 0.1**: Reduce overconfidence
  - Con solo ~11 videos por clase, el modelo puede sobre-ajustarse
  - Mejora generalización en datasets pequeños

- **Class Weighted True**: CRÍTICO para desbalance
  - Clases con 6 videos vs 19 videos (ratio 3.2:1)
  - Sin pesos, el modelo ignorará clases minoritarias

- **Warmup 10%**: Inicio gradual
  - 3 epochs de warmup en 30 epochs totales
  - Evita gradientes explosivos al inicio

- **Patience 5**: Early stopping moderado
  - Dataset pequeño → puede oscilar
  - 5 epochs sin mejora es razonable

#### ⚡ Configuración Agresiva (Experimental)
```python
DATASET = "wlasl100"
BATCH_SIZE = 12
MAX_EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
LABEL_SMOOTHING = 0.15
CLASS_WEIGHTED = True
WARMUP_RATIO = 0.15
PATIENCE = 7
GRADIENT_CLIP = 0.5
```

**Cuándo usar:**
- Si baseline sobre-ajusta (val loss sube después de epoch 10-15)
- Si quieres maximizar accuracy a costa de tiempo
- Si tienes GPU potente y tiempo

**Cambios clave:**
- Batch size más pequeño (12) → más actualizaciones, menos overfitting
- LR más bajo (5e-5) → ajuste más fino
- Weight decay más alto (0.1) → más regularización
- Label smoothing más alto (0.15) → menos confianza
- Patience más largo (7) → más oportunidades de mejorar

---

### Para WLASL300 (Más Desafiante)

#### ✅ Configuración Recomendada
```python
DATASET = "wlasl300"
BATCH_SIZE = 12
MAX_EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.08
LABEL_SMOOTHING = 0.15
CLASS_WEIGHTED = True
WARMUP_RATIO = 0.15
PATIENCE = 7
GRADIENT_CLIP = 1.0
```

**Justificación:**
- **Batch Size 12**: Más clases (300) → batches más pequeños
  - 1,960 / 12 ≈ 163 steps por epoch
  - Más actualizaciones por epoch mejora aprendizaje de clases raras

- **Learning Rate 5e-5**: Más conservador
  - 300 clases es complejo → necesita ajuste más cuidadoso
  - Evita colapso en primeras epochs

- **Max Epochs 50**: Más tiempo para aprender
  - 3x más clases que WLASL100
  - Necesita más epochs para converger

- **Weight Decay 0.08**: Regularización moderada-alta
  - Balance entre prevenir overfitting y aprender 300 clases

- **Label Smoothing 0.15**: Mayor suavizado
  - Desbalance alto (1-16 videos/clase)
  - Reduce overconfidence en clases dominantes

- **Class Weighted True**: ESENCIAL
  - Desbalance ratio 16:1 es crítico
  - Sin pesos, 90% del modelo aprenderá solo top 50 clases

- **Patience 7**: Más flexible
  - Con 300 clases, el aprendizaje es más lento
  - Permite más oscilaciones antes de parar

#### 🚀 Configuración Optimista (Si tienes buenos recursos)
```python
DATASET = "wlasl300"
BATCH_SIZE = 16
MAX_EPOCHS = 70
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 0.06
LABEL_SMOOTHING = 0.12
CLASS_WEIGHTED = True
WARMUP_RATIO = 0.12
PATIENCE = 10
GRADIENT_CLIP = 1.5
```

**Cuándo usar:**
- GPU con ≥16GB VRAM (Tesla T4 o mejor)
- Quieres maximizar accuracy
- No te importa tiempo de entrenamiento (6-12 horas)

---

## 🔬 Justificación Científica

### 1. Learning Rate (Crítico)

**Por qué tasas bajas (1e-4, 5e-5)?**
- **Transfer Learning**: Modelo pre-entrenado en Kinetics-400
  - Features visuales ya aprendidas
  - Solo necesitamos adaptarlas a señas
  - LR alto (>1e-3) destruiría estas features

- **Literatura**:
  - VideoMAE paper usa 1e-3 para pre-training desde cero
  - Fine-tuning típicamente usa 10x-100x menor
  - Nuestro caso: 1e-4 a 5e-5 es apropiado

**Referencias:**
- Tong et al. (2022): "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
- Dosovitskiy et al. (2021): Vision Transformer - LR 1e-3 pre-training, 1e-4 fine-tuning

### 2. Batch Size

**Por qué 12-16 y no 32 o 64?**
- **Dataset pequeño**: 807-1,960 videos de entrenamiento
  - Batch grande (32+) → pocas actualizaciones por epoch
  - 807 / 32 = 25 steps/epoch (muy poco)
  - 807 / 16 = 50 steps/epoch (mejor)

- **Desbalance de clases**:
  - Batch pequeño → más probabilidad de incluir clases raras
  - Con batch 32 y 100 clases, muchas clases nunca aparecen en un batch

- **Memory vs Learning**:
  - Batch 16: ~12GB VRAM (Colab T4 OK)
  - Batch 32: ~20GB VRAM (requiere A100)

**Literatura:**
- Masters & Luschi (2018): "Revisiting Small Batch Training for Deep Neural Networks"
  - Batches pequeños generalizan mejor en datasets pequeños

### 3. Weight Decay (Regularización L2)

**Por qué 0.05-0.1?**
- **Prevención de Overfitting**:
  - Videos: 1,118 (WLASL100) vs millones en ImageNet
  - Sin regularización → 100% train acc, 50% val acc

- **Vision Transformers**:
  - ViT usa weight decay 0.05-0.3
  - VideoMAE hereda esta arquitectura

- **Nuestro caso**:
  - 0.05 para WLASL100 (menos clases)
  - 0.08-0.1 para WLASL300 (más complejo)

**Literatura:**
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"

### 4. Label Smoothing

**Por qué 0.1-0.15?**
- **Calibración de confianza**:
  - Modelo tiende a ser overconfident con pocos datos
  - "This is 100% class X" con solo 11 ejemplos es arriesgado

- **Regularización implícita**:
  - Suaviza la distribución objetivo
  - Evita que el modelo memorice ruido

- **Mejora generalización**:
  - Szegedy et al. (2016): +2-3% accuracy en ImageNet
  - En datasets pequeños el efecto es mayor

**Literatura:**
- Szegedy et al. (2016): "Rethinking the Inception Architecture"
- Müller et al. (2019): "When Does Label Smoothing Help?"

### 5. Class Weighting (CRÍTICO)

**Por qué es esencial?**

**WLASL100:**
- Clase más común: 19 videos
- Clase menos común: 6 videos
- Sin pesos → modelo optimiza para clases comunes
- Resultado: 85% accuracy en top-20 clases, 30% en bottom-20

**WLASL300:**
- Clase más común: 16 videos
- Clase menos común: 1 video (!)
- Ratio 16:1 es extremo
- Sin pesos → modelo ignora 150+ clases raras

**Cálculo de pesos:**
```python
weight[class] = 1 / count[class]
normalized_weight[class] = weight[class] / mean(weights)
```

**Literatura:**
- Cui et al. (2019): "Class-Balanced Loss Based on Effective Number of Samples"

### 6. Warmup

**Por qué 10-15% de warmup?**
- **Inicio gradual**:
  - Primeros batches pueden tener gradientes grandes
  - Warmup evita desestabilizar pesos pre-entrenados

- **AdamW + Warmup**:
  - Combinación estándar en Transformers
  - BERT, GPT, ViT todos usan warmup

- **Duración**:
  - 10% (3 epochs en 30) para WLASL100
  - 15% (7-8 epochs en 50) para WLASL300

**Literatura:**
- Goyal et al. (2017): "Accurate, Large Minibatch SGD"

---

## 📈 Estrategias de Experimentación

### Fase 1: Baseline (Primer Intento)
```bash
python scripts/train.py \
    --dataset wlasl100 \
    --batch_size 16 \
    --max_epochs 30 \
    --lr 1e-4
```

**Esperado:**
- Val Accuracy: 40-55%
- Train/Val gap: 10-15%
- Converge en ~20 epochs

**Si obtienes:**
- Val Acc < 40% → Incrementa epochs o reduce LR
- Train/Val gap > 20% → Aumenta regularización (weight decay, label smoothing)
- Val Acc > 55% → ¡Excelente! Prueba WLASL300

### Fase 2: Fine-tuning
```bash
python scripts/train.py \
    --dataset wlasl100 \
    --batch_size 12 \
    --max_epochs 50 \
    --lr 5e-5 \
    --weight_decay 0.08
```

**Esperado:**
- Val Accuracy: 45-60%
- Mejor generalización

### Fase 3: WLASL300
```bash
python scripts/train.py \
    --dataset wlasl300 \
    --batch_size 12 \
    --max_epochs 50 \
    --lr 5e-5 \
    --weight_decay 0.08 \
    --label_smoothing 0.15
```

**Esperado:**
- Val Accuracy: 25-40% (normal con 300 clases)
- Top-5 Accuracy: 50-65%

---

## 🎯 Grid Search Sugerido

### Para WLASL100

| Parámetro | Valores a probar | Prioridad |
|-----------|------------------|-----------|
| Learning Rate | [5e-5, 1e-4, 2e-4] | ⭐⭐⭐ Alta |
| Batch Size | [12, 16] | ⭐⭐ Media |
| Weight Decay | [0.05, 0.08, 0.1] | ⭐⭐ Media |
| Label Smoothing | [0.1, 0.15] | ⭐ Baja |

**Combinación recomendada para explorar:**
1. LR=1e-4, BS=16, WD=0.05 (baseline)
2. LR=5e-5, BS=12, WD=0.08 (conservador)
3. LR=2e-4, BS=16, WD=0.1 (agresivo)

### Para WLASL300

| Parámetro | Valores a probar | Prioridad |
|-----------|------------------|-----------|
| Learning Rate | [3e-5, 5e-5, 8e-5] | ⭐⭐⭐ Alta |
| Batch Size | [8, 12, 16] | ⭐⭐⭐ Alta |
| Weight Decay | [0.06, 0.08, 0.1] | ⭐⭐ Media |
| Max Epochs | [50, 70] | ⭐ Baja |

---

## ⚠️ Señales de Alerta

### Overfitting
- **Síntomas**: Train Acc > 80%, Val Acc < 50%
- **Solución**:
  - ↑ Weight decay (0.05 → 0.1)
  - ↑ Label smoothing (0.1 → 0.15)
  - ↓ Batch size (16 → 12)

### Underfitting
- **Síntomas**: Train Acc < 60%, Val Acc ≈ Train Acc
- **Solución**:
  - ↑ Learning rate (1e-4 → 2e-4)
  - ↑ Max epochs (30 → 50)
  - ↓ Weight decay (0.05 → 0.03)

### Inestabilidad
- **Síntomas**: Loss oscila mucho, no converge
- **Solución**:
  - ↓ Learning rate (1e-4 → 5e-5)
  - ↑ Warmup ratio (0.1 → 0.2)
  - ↓ Gradient clip (1.0 → 0.5)

---

## 📚 Referencias

1. Tong et al. (2022): "VideoMAE: Masked Autoencoders are Data-Efficient Learners"
2. Dosovitskiy et al. (2021): "An Image is Worth 16x16 Words: Transformers for Image Recognition"
3. Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"
4. Szegedy et al. (2016): "Rethinking the Inception Architecture for Computer Vision"
5. Cui et al. (2019): "Class-Balanced Loss Based on Effective Number of Samples"
6. Masters & Luschi (2018): "Revisiting Small Batch Training for Deep Neural Networks"

---

## 🎓 Conclusión

### Recomendación Principal para tu Tesis

**Empezar con WLASL100:**
```python
DATASET = "wlasl100"
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
CLASS_WEIGHTED = True
PATIENCE = 5
```

**Después experimentar con WLASL300:**
```python
DATASET = "wlasl300"
BATCH_SIZE = 12
MAX_EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.08
LABEL_SMOOTHING = 0.15
CLASS_WEIGHTED = True
PATIENCE = 7
```

**¿Por qué esta estrategia?**
1. WLASL100 es más rápido (2-4h vs 8-12h)
2. Permite iterar y entender el modelo
3. Establece baseline para comparar con WLASL300
4. WLASL300 demuestra escalabilidad a más clases

**Métricas a reportar en tesis:**
- Accuracy (total, por clase, top-5)
- Precision, Recall, F1-score (macro y weighted)
- Matriz de confusión
- Curvas de aprendizaje (train/val loss y accuracy)
- Comparación WLASL100 vs WLASL300

---

**Buena suerte con el entrenamiento! 🚀**
