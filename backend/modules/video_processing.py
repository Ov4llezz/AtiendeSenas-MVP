"""
Módulo de Procesamiento de Video
Responsabilidades:
- Extraer frames uniformemente del video
- Redimensionar a 224x224
- Normalizar según estadísticas de VideoMAE
- Convertir a tensor compatible con PyTorch
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple


# Estadísticas de normalización de VideoMAE (ImageNet)
VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD = [0.229, 0.224, 0.225]

# Configuración de frames
NUM_FRAMES = 16
FRAME_SIZE = 224


class VideoProcessingError(Exception):
    """Error personalizado para problemas de procesamiento de video"""
    pass


def extract_frames_uniform(video_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """
    Extrae frames uniformemente distribuidos a lo largo del video.

    Args:
        video_path: Ruta al archivo de video
        num_frames: Número de frames a extraer (default: 16)

    Returns:
        np.ndarray: Array de frames con shape (num_frames, height, width, 3)

    Raises:
        VideoProcessingError: Si hay problemas al leer el video
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise VideoProcessingError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise VideoProcessingError("El video no tiene frames válidos")

    # Calcular índices uniformemente espaciados
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # Convertir de BGR (OpenCV) a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    # Si no se pudieron leer suficientes frames
    if len(frames) == 0:
        raise VideoProcessingError("No se pudieron extraer frames del video")

    # Si faltan frames, duplicar el último
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.array(frames[:num_frames])


def resize_frames(frames: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """
    Redimensiona frames a tamaño cuadrado.

    Args:
        frames: Array de frames (num_frames, H, W, 3)
        size: Tamaño objetivo (default: 224)

    Returns:
        np.ndarray: Frames redimensionados (num_frames, size, size, 3)
    """
    resized_frames = []

    for frame in frames:
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized)

    return np.array(resized_frames)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """
    Normaliza frames según estadísticas de VideoMAE.

    Args:
        frames: Array de frames (num_frames, H, W, 3) con valores 0-255

    Returns:
        np.ndarray: Frames normalizados (num_frames, H, W, 3)
    """
    # Convertir a float32 y escalar a [0, 1]
    frames_float = frames.astype(np.float32) / 255.0

    # Normalizar con mean y std
    mean = np.array(VIDEOMAE_MEAN, dtype=np.float32)
    std = np.array(VIDEOMAE_STD, dtype=np.float32)

    normalized = (frames_float - mean) / std

    return normalized


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Convierte frames a tensor de PyTorch en formato VideoMAE.

    Args:
        frames: Array normalizado (num_frames, H, W, 3)

    Returns:
        torch.Tensor: Tensor con shape (1, T, C, H, W) = (1, 16, 3, 224, 224)
                     Formato: (batch, time, channels, height, width)
    """
    # Transponer de (T, H, W, C) a (T, C, H, W)
    frames_t = np.transpose(frames, (0, 3, 1, 2))

    # Convertir a tensor
    tensor = torch.from_numpy(frames_t).float()

    # Agregar dimensión de batch: (T, C, H, W) -> (1, T, C, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor


def process_video(video_path: Path) -> Tuple[torch.Tensor, dict]:
    """
    Pipeline completo de procesamiento de video.

    Args:
        video_path: Ruta al archivo de video

    Returns:
        Tuple[torch.Tensor, dict]:
            - Tensor listo para inferencia (1, T, C, H, W)
            - Metadata del procesamiento

    Raises:
        VideoProcessingError: Si hay problemas en el procesamiento
    """
    try:
        # 1. Extraer frames uniformemente
        frames = extract_frames_uniform(video_path, num_frames=NUM_FRAMES)

        # 2. Redimensionar a 224x224
        frames_resized = resize_frames(frames, size=FRAME_SIZE)

        # 3. Normalizar
        frames_normalized = normalize_frames(frames_resized)

        # 4. Convertir a tensor
        video_tensor = frames_to_tensor(frames_normalized)

        # Metadata
        metadata = {
            "num_frames": NUM_FRAMES,
            "frame_size": FRAME_SIZE,
            "tensor_shape": list(video_tensor.shape),
            "video_path": str(video_path)
        }

        return video_tensor, metadata

    except Exception as e:
        raise VideoProcessingError(f"Error en el procesamiento del video: {str(e)}")
