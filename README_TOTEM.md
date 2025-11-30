# 🤖 Tótem LSCh - Sistema de Reconocimiento de Lengua de Señas Chilena

Sistema completo de reconocimiento de señas usando VideoMAE + Chatbot Gemini para orientación en salud pública.

## 📋 Descripción

Este proyecto implementa un tótem de autoatención que permite a personas Sordas comunicarse mediante videos de señas. El sistema:

1. Recibe un video de una seña
2. Detecta la palabra correspondiente con VideoMAE
3. Genera una respuesta empática contextualizada con Gemini
4. Muestra todo en una interfaz minimalista tipo tótem

## 🏗️ Arquitectura

```
AtiendeSenas-MVP/
├── backend/                 # API FastAPI (Python 3.10)
│   ├── main.py             # Endpoint principal
│   ├── config.py           # Configuración
│   ├── modules/            # Módulos separados
│   │   ├── video_ingestion.py
│   │   ├── video_processing.py
│   │   ├── videomae_inference.py
│   │   ├── conversation_history.py
│   │   └── gemini_chatbot.py
│   └── requirements.txt
│
└── frontend/               # React + Vite + TypeScript
    ├── src/
    │   ├── components/
    │   │   ├── VideoUploader.tsx
    │   │   ├── PredictionDisplay.tsx
    │   │   ├── ChatResponseDisplay.tsx
    │   │   ├── LatencyPanel.tsx
    │   │   └── LoadingIndicator.tsx
    │   ├── App.tsx
    │   └── main.tsx
    └── package.json
```

## 🚀 Instalación y Configuración

### **Prerequisitos**

- Python 3.10.0
- Node.js 18+ y npm
- GPU recomendada (para inferencia VideoMAE)
- API Key de Google Gemini

### **1. Backend Setup**

```bash
cd backend

# Activar entorno virtual (si no está activo)
source ../venv_backend/Scripts/activate  # Windows Git Bash
# o
../venv_backend/Scripts/activate.bat     # Windows CMD

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu GEMINI_API_KEY
```

**Archivo `.env` requerido:**

```env
GEMINI_API_KEY=tu_api_key_aqui
MODEL_PATH=../models/v2/wlasl100/checkpoints/run_XXXXXX/best_model.pt
GLOSSES_PATH=../glosas_wlasl100_es.txt
NUM_CLASSES=100
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173
MIN_CONFIDENCE=0.55
```

### **2. Frontend Setup**

```bash
cd frontend

# Instalar dependencias
npm install
```

## ▶️ Ejecución

### **Opción 1: Desarrollo (2 terminales)**

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

El servidor estará en: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

El frontend estará en: `http://localhost:5173`

### **Opción 2: Producción**

**Backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run build
npm run preview
```

## 📡 API Endpoints

### **`POST /api/full-pipeline`**

Procesa un video de seña completo.

**Request:**
- `video`: archivo de video (mp4/mov)
- `history`: (opcional) historial previo

**Response:**
```json
{
  "predicted_word": "PAIN",
  "confidence": 0.87,
  "chatbot_response": "Entiendo que siente dolor. ¿Puede indicarme dónde le duele?",
  "history": ["HELLO", "HELP", "PAIN"],
  "latency_ms": {
    "videomae": 450.2,
    "chatbot": 320.5,
    "total": 770.7
  }
}
```

### **Otros Endpoints:**

- `GET /health` - Health check
- `POST /api/reset-history` - Reiniciar historial
- `GET /api/history` - Obtener historial actual

## 🎯 Flujo del Sistema

1. **Usuario:** Sube video de seña en la interfaz
2. **Backend:**
   - Valida y guarda video temporalmente
   - Extrae 16 frames uniformes
   - Redimensiona a 224x224 y normaliza
   - Inferencia con VideoMAE → palabra + confianza
3. **Decisión:**
   - Si confianza < 0.55 → mensaje fallback
   - Si confianza >= 0.55 → actualizar historial + llamar Gemini
4. **Gemini:** Genera respuesta empática (max 2 oraciones, español chileno)
5. **Frontend:** Muestra palabra, confianza, respuesta y latencias

## ⚙️ Configuración

### **Hiperparámetros del Backend (`.env`)**

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MIN_CONFIDENCE` | 0.55 | Confianza mínima para llamar a Gemini |
| `MAX_UPLOAD_SIZE_MB` | 50 | Tamaño máximo de video |
| `MAX_HISTORY_LENGTH` | 3 | Máximo de palabras en historial |

### **Reset de Historial Automático**

El historial se resetea cuando se detecta:
- Saludos: "HELLO", "HI"
- Despedidas: "THANKS", "THANK YOU", "GOODBYE", "BYE"

## 🛠️ Tecnologías

**Backend:**
- FastAPI
- PyTorch + VideoMAE
- Google Generative AI (Gemini)
- OpenCV

**Frontend:**
- React 18
- Vite
- TypeScript
- TailwindCSS
- Axios

## 📊 Performance

- Latencia típica VideoMAE: ~300-500ms
- Latencia típica Gemini: ~200-400ms
- **Latencia total: <1 segundo** (ideal)

## 🔒 Seguridad

- Validación estricta de archivos (formato, tamaño)
- CORS configurado para orígenes específicos
- API key de Gemini en variables de entorno
- Limpieza automática de archivos temporales
- Sanitización de inputs

## 🐛 Solución de Problemas

### Error: `GEMINI_API_KEY no está configurada`
→ Asegúrate de crear el archivo `.env` con tu API key

### Error: `No se encontró checkpoint`
→ Verifica que `MODEL_PATH` en `.env` apunte al modelo correcto

### Frontend no se conecta al backend
→ Verifica que el backend esté corriendo en `http://localhost:8000`

### CORS Error
→ Verifica que `CORS_ORIGINS` en `.env` incluya `http://localhost:5173`

## 📝 Licencia

Este proyecto es parte de la tesis de Rafael Ovalle - UNAB

## 👥 Contacto

Para preguntas o soporte, contactar a: [email]

---

**Desarrollado con ❤️ para la comunidad Sorda de Chile**
