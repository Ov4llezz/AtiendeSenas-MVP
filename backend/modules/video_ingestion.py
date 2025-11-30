"""
Módulo de Ingreso de Video
Responsabilidades:
- Recibir videos vía multipart/form-data
- Validar formato, tamaño, duración
- Guardar temporalmente de forma segura
"""

import os
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException
import cv2

from config import config


class VideoIngestionError(Exception):
    """Error personalizado para problemas de ingreso de video"""
    pass


def validate_file_extension(filename: str) -> bool:
    """
    Valida que la extensión del archivo sea permitida.

    Args:
        filename: Nombre del archivo

    Returns:
        True si la extensión es válida
    """
    extension = filename.split('.')[-1].lower()
    return extension in config.ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """
    Valida que el tamaño del archivo no exceda el máximo permitido.

    Args:
        file_size: Tamaño del archivo en bytes

    Returns:
        True si el tamaño es válido
    """
    return file_size <= config.MAX_UPLOAD_SIZE_BYTES


def validate_video_integrity(video_path: Path) -> Tuple[bool, str]:
    """
    Valida que el video no esté corrupto y tenga frames válidos.

    Args:
        video_path: Ruta al archivo de video

    Returns:
        Tupla (es_valido, mensaje_error)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return False, "No se pudo abrir el video. Puede estar corrupto."

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        # Validar que tenga al menos algunos frames
        if frame_count < 5:
            return False, "El video debe tener al menos 5 frames."

        # Validar FPS válido
        if fps <= 0:
            return False, "El video tiene un FPS inválido."

        return True, ""

    except Exception as e:
        return False, f"Error al validar el video: {str(e)}"


async def ingest_video(video_file: UploadFile) -> Path:
    """
    Procesa el ingreso de un video: valida y guarda temporalmente.

    Args:
        video_file: Archivo de video subido

    Returns:
        Path: Ruta al archivo temporal guardado

    Raises:
        HTTPException: Si hay algún problema con el video
    """

    # 1. Validar que se subió un archivo
    if not video_file:
        raise HTTPException(status_code=400, detail="No se recibió ningún archivo de video")

    # 2. Validar extensión
    if not validate_file_extension(video_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Use: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

    # 3. Leer contenido del archivo
    try:
        contents = await video_file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {str(e)}")

    # 4. Validar tamaño
    file_size = len(contents)
    if not validate_file_size(file_size):
        max_mb = config.MAX_UPLOAD_SIZE_MB
        raise HTTPException(
            status_code=400,
            detail=f"El archivo excede el tamaño máximo permitido ({max_mb} MB)"
        )

    # 5. Generar nombre único para el archivo temporal
    file_extension = video_file.filename.split('.')[-1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    temp_path = config.TEMP_UPLOAD_DIR / unique_filename

    # 6. Guardar archivo temporalmente
    try:
        with open(temp_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

    # 7. Validar integridad del video
    is_valid, error_message = validate_video_integrity(temp_path)
    if not is_valid:
        # Eliminar archivo temporal si no es válido
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=error_message)

    return temp_path


def cleanup_temp_file(file_path: Path) -> None:
    """
    Elimina un archivo temporal de forma segura.

    Args:
        file_path: Ruta al archivo a eliminar
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"[WARN] No se pudo eliminar el archivo temporal {file_path}: {e}")
