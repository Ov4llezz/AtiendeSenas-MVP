"""
Módulo de Inferencia VideoMAE
Responsabilidades:
- Cargar modelo VideoMAE pre-entrenado (una sola vez)
- Realizar inferencia sobre tensores de video
- Retornar: palabra detectada, confianza, latencia
- Integra el modelo ya entrenado del proyecto
"""

import time
import torch
from pathlib import Path
from typing import Tuple, Dict
from transformers import VideoMAEForVideoClassification

from config import config


class VideoMAEInferenceError(Exception):
    """Error personalizado para problemas de inferencia"""
    pass


class VideoMAEInference:
    """
    Clase singleton para manejar inferencia de VideoMAE.
    Carga el modelo una sola vez en memoria.
    """

    _instance = None
    _model = None
    _device = None
    _id2gloss = None

    def __new__(cls):
        """Patrón Singleton para cargar modelo una sola vez"""
        if cls._instance is None:
            cls._instance = super(VideoMAEInference, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Inicializa el modelo VideoMAE y mapeo de clases"""
        print("[INFO] Inicializando modelo VideoMAE...")

        # Detectar dispositivo
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Usando dispositivo: {self._device}")

        # Cargar mapeo de glosas
        self._load_glosses()

        # Cargar modelo
        try:
            # Cargar arquitectura base
            self._model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics",
                num_labels=config.NUM_CLASSES,
                ignore_mismatched_sizes=True
            )

            # Cargar pesos entrenados si existe checkpoint
            checkpoint_path = Path(config.MODEL_PATH)
            if checkpoint_path.exists():
                print(f"[INFO] Cargando checkpoint desde: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self._device)
                self._model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[INFO] Checkpoint cargado (Epoch: {checkpoint.get('epoch', 'N/A')})")
            else:
                print(f"[WARN] No se encontró checkpoint en {checkpoint_path}")
                print("[WARN] Usando modelo base sin fine-tuning")

            # Mover modelo a dispositivo y modo evaluación
            self._model.to(self._device)
            self._model.eval()

            print("[INFO] Modelo VideoMAE cargado exitosamente")

        except Exception as e:
            raise VideoMAEInferenceError(f"Error al cargar el modelo: {str(e)}")

    def _load_glosses(self):
        """Carga el mapeo de IDs a glosas desde el archivo de traducción"""
        glosses_path = Path(config.GLOSSES_PATH)

        if not glosses_path.exists():
            raise VideoMAEInferenceError(f"Archivo de glosas no encontrado: {glosses_path}")

        self._id2gloss = {}

        try:
            with open(glosses_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Saltar comentarios y líneas vacías
                    if not line or line.startswith('#') or line.startswith('='):
                        continue

                    # Formato: "ID | Glosa (Inglés) | Traducción (Español)"
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3:
                        class_id = int(parts[0])
                        gloss_en = parts[1]
                        gloss_es = parts[2]

                        # Guardar tanto inglés como español
                        self._id2gloss[class_id] = {
                            'english': gloss_en,
                            'spanish': gloss_es
                        }

            print(f"[INFO] Cargadas {len(self._id2gloss)} glosas desde {glosses_path}")

        except Exception as e:
            raise VideoMAEInferenceError(f"Error al cargar glosas: {str(e)}")

    @torch.no_grad()
    def predict(self, video_tensor: torch.Tensor) -> Tuple[str, float, float]:
        """
        Realiza inferencia sobre un tensor de video.

        Args:
            video_tensor: Tensor de video (1, T, C, H, W) = (1, 16, 3, 224, 224)

        Returns:
            Tuple[str, float, float]:
                - palabra detectada (glosa en inglés, ej: "PAIN")
                - confianza (0.0 a 1.0)
                - latencia en milisegundos

        Raises:
            VideoMAEInferenceError: Si hay problemas en la inferencia
        """
        try:
            start_time = time.time()

            # Mover tensor a dispositivo
            video_tensor = video_tensor.to(self._device)

            # Inferencia
            outputs = self._model(pixel_values=video_tensor)
            logits = outputs.logits

            # Obtener predicción
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)

            predicted_id = predicted_class.item()
            confidence_value = confidence.item()

            # Obtener glosa
            if predicted_id in self._id2gloss:
                predicted_word = self._id2gloss[predicted_id]['english']
            else:
                predicted_word = f"UNKNOWN_{predicted_id}"

            # Calcular latencia
            latency_ms = (time.time() - start_time) * 1000

            return predicted_word, confidence_value, latency_ms

        except Exception as e:
            raise VideoMAEInferenceError(f"Error durante la inferencia: {str(e)}")

    def get_spanish_translation(self, gloss_en: str) -> str:
        """
        Obtiene la traducción al español de una glosa.

        Args:
            gloss_en: Glosa en inglés

        Returns:
            str: Traducción al español
        """
        for class_id, glosses in self._id2gloss.items():
            if glosses['english'].upper() == gloss_en.upper():
                return glosses['spanish']

        return gloss_en  # Si no se encuentra, retornar original


# Instancia global del modelo (Singleton)
videomae_model = VideoMAEInference()
