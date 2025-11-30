"""
Configuración del Backend - Tótem LSCh
Carga variables de entorno y configuración general del sistema
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

class Config:
    """Configuración centralizada del sistema"""

    # === Gemini API ===
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno")

    # === VideoMAE Model ===
    MODEL_PATH = os.getenv("MODEL_PATH", "../models/v2/wlasl100/checkpoints/best_model.pt")
    GLOSSES_PATH = os.getenv("GLOSSES_PATH", "../glosas_wlasl100_es.txt")
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "100"))

    # === Server ===
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

    # === Upload ===
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "mp4,mov").split(",")
    TEMP_UPLOAD_DIR = Path(os.getenv("TEMP_UPLOAD_DIR", "temp_uploads"))

    # Crear directorio temporal si no existe
    TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

    # === Thresholds ===
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

    # === Historial Conversacional ===
    MAX_HISTORY_LENGTH = 3

    # === Palabras especiales para reset de historial ===
    GREETING_WORDS = ["HELLO", "HI"]
    FAREWELL_WORDS = ["THANKS", "THANK YOU", "GOODBYE", "BYE"]


# Instancia global de configuración
config = Config()
