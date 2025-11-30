"""
Backend API - Tótem LSCh (Lengua de Señas Chilena)
Sistema de reconocimiento de señas + chatbot Gemini

Endpoint principal: POST /api/full-pipeline
"""

import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from modules import (
    ingest_video,
    cleanup_temp_file,
    process_video,
    videomae_model,
    ConversationHistory,
    gemini_chatbot
)


# === Modelos de datos ===

class LatencyInfo(BaseModel):
    """Información de latencias del pipeline"""
    videomae: float
    chatbot: float
    total: float


class PipelineResponse(BaseModel):
    """Respuesta del endpoint /api/full-pipeline"""
    predicted_word: str
    confidence: float
    chatbot_response: str
    history: List[str]
    latency_ms: LatencyInfo


# === Inicialización de FastAPI ===

app = FastAPI(
    title="Tótem LSCh API",
    description="API de reconocimiento de Lengua de Señas + Chatbot Gemini",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global de historial conversacional
# En producción, esto debería ser por sesión de usuario
conversation_history = ConversationHistory()


# === Endpoints ===

@app.get("/")
async def root():
    """Endpoint raíz - Health check"""
    return {
        "status": "online",
        "service": "Tótem LSCh API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Endpoint de health check detallado"""
    return {
        "status": "healthy",
        "model_loaded": videomae_model is not None,
        "gemini_configured": config.GEMINI_API_KEY is not None,
        "history_length": len(conversation_history)
    }


@app.post("/api/full-pipeline", response_model=PipelineResponse)
async def full_pipeline(
    video: UploadFile = File(..., description="Video de seña (mp4/mov)"),
    history: Optional[str] = Form(None, description="Historial previo (JSON string)")
):
    """
    Pipeline completo de procesamiento de video de señas.

    Flujo:
    1. Recibe video del usuario
    2. Procesa y extrae frames
    3. Realiza inferencia con VideoMAE
    4. Si confianza >= 0.55:
       - Actualiza historial
       - Genera respuesta con Gemini
    5. Si confianza < 0.55:
       - Retorna mensaje fallback
    6. Retorna resultado estructurado con latencias

    Args:
        video: Archivo de video (mp4 o mov)
        history: Historial previo serializado (opcional)

    Returns:
        PipelineResponse: Resultado completo del pipeline
    """

    temp_video_path = None
    pipeline_start = time.time()

    try:
        # === 1. INGRESO DE VIDEO ===
        print(f"\n[PIPELINE] Procesando video: {video.filename}")

        temp_video_path = await ingest_video(video)
        print(f"[OK] Video guardado temporalmente: {temp_video_path}")

        # === 2. PROCESAMIENTO DE VIDEO ===
        print("[PIPELINE] Procesando frames...")

        video_tensor, metadata = process_video(temp_video_path)
        print(f"[OK] Video procesado: {metadata['tensor_shape']}")

        # === 3. INFERENCIA VIDEOMAE ===
        print("[PIPELINE] Ejecutando inferencia VideoMAE...")

        predicted_word, confidence, videomae_latency = videomae_model.predict(video_tensor)

        print(f"[OK] Predicción: {predicted_word} (confianza: {confidence:.2%})")
        print(f"[OK] Latencia VideoMAE: {videomae_latency:.2f}ms")

        # === 4. DECISIÓN BASADA EN CONFIANZA ===

        chatbot_latency = 0.0
        chatbot_response = ""

        if confidence < config.MIN_CONFIDENCE:
            # Confianza baja → No llamar a Gemini
            print(f"[WARN] Confianza baja ({confidence:.2%}). Usando mensaje fallback.")

            chatbot_response = gemini_chatbot.generate_low_confidence_response()
            # NO actualizar historial

        else:
            # Confianza suficiente → Procesar normalmente

            # === 5. ACTUALIZAR HISTORIAL ===
            print("[PIPELINE] Actualizando historial conversacional...")

            updated_history = conversation_history.update(predicted_word, confidence)
            print(f"[OK] Historial actualizado: {updated_history}")

            # === 6. GENERAR RESPUESTA CON GEMINI ===
            print("[PIPELINE] Generando respuesta con Gemini...")

            try:
                chatbot_response, chatbot_latency = gemini_chatbot.generate_response(
                    current_word=predicted_word,
                    history=updated_history
                )
                print(f"[OK] Respuesta generada: {chatbot_response[:50]}...")
                print(f"[OK] Latencia Gemini: {chatbot_latency:.2f}ms")

            except Exception as e:
                print(f"[ERROR] Fallo al generar respuesta con Gemini: {e}")
                chatbot_response = gemini_chatbot.generate_error_fallback("api_error")
                chatbot_latency = 0.0

        # === 7. CALCULAR LATENCIAS TOTALES ===
        total_latency = (time.time() - pipeline_start) * 1000

        print(f"[OK] Latencia total del pipeline: {total_latency:.2f}ms")

        # === 8. CONSTRUIR RESPUESTA ===
        response = PipelineResponse(
            predicted_word=predicted_word,
            confidence=confidence,
            chatbot_response=chatbot_response,
            history=conversation_history.get_history(),
            latency_ms=LatencyInfo(
                videomae=videomae_latency,
                chatbot=chatbot_latency,
                total=total_latency
            )
        )

        print("[PIPELINE] ✓ Pipeline completado exitosamente\n")

        return response

    except HTTPException:
        # Re-raise HTTPExceptions (errores de validación)
        raise

    except Exception as e:
        # Errores inesperados
        print(f"[ERROR] Error inesperado en el pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

    finally:
        # === 9. LIMPIEZA ===
        # Siempre limpiar archivo temporal, incluso si hay errores
        if temp_video_path:
            cleanup_temp_file(temp_video_path)
            print(f"[CLEANUP] Archivo temporal eliminado")


@app.post("/api/reset-history")
async def reset_history():
    """
    Reinicia el historial conversacional.

    Returns:
        dict: Confirmación de reset
    """
    conversation_history.reset()
    return {
        "status": "success",
        "message": "Historial conversacional reiniciado",
        "history": []
    }


@app.get("/api/history")
async def get_history():
    """
    Obtiene el historial conversacional actual.

    Returns:
        dict: Historial actual
    """
    return {
        "history": conversation_history.get_history(),
        "length": len(conversation_history)
    }


# === Ejecución ===

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print(" INICIANDO SERVIDOR - TÓTEM LSCh API ".center(70))
    print("="*70)
    print(f"Host: {config.HOST}")
    print(f"Puerto: {config.PORT}")
    print(f"CORS Origins: {config.CORS_ORIGINS}")
    print(f"Confianza mínima: {config.MIN_CONFIDENCE}")
    print("="*70 + "\n")

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
