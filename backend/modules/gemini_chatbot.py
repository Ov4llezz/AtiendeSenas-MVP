"""
Módulo Chatbot Gemini
Responsabilidades:
- Conectar con API de Gemini
- Construir prompts contextualizados
- Generar respuestas empáticas en español chileno
- Manejar errores con fallbacks seguros
"""

import time
import google.generativeai as genai
from typing import Tuple, List

from config import config


class GeminiChatbotError(Exception):
    """Error personalizado para problemas con Gemini"""
    pass


class GeminiChatbot:
    """Clase para manejar interacciones con Gemini"""

    def __init__(self):
        """Inicializa cliente de Gemini"""
        try:
            # Configurar API key desde variable de entorno
            genai.configure(api_key=config.GEMINI_API_KEY)

            # Configurar modelo
            self.model = genai.GenerativeModel('gemini-pro')

            # Configuración de generación
            self.generation_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 150,  # Limitar a ~2 oraciones
            }

            print("[INFO] Cliente Gemini inicializado correctamente")

        except Exception as e:
            raise GeminiChatbotError(f"Error al inicializar Gemini: {str(e)}")

    def _build_prompt(self, current_word: str, history: List[str]) -> str:
        """
        Construye el prompt para Gemini según especificaciones.

        Args:
            current_word: Palabra actual detectada (glosa en inglés)
            history: Lista de palabras previas

        Returns:
            str: Prompt completo para Gemini
        """
        # Contexto del historial
        if history and len(history) > 0:
            history_context = f"Palabras previas del usuario: {', '.join(history)}."
        else:
            history_context = "Esta es la primera palabra de la conversación."

        # Prompt estructurado
        prompt = f"""Eres un asistente virtual de salud en un tótem de autoatención en Chile.
Tu rol es ayudar a personas Sordas que usan Lengua de Señas.

IMPORTANTE - Reglas obligatorias:
1. Responde en ESPAÑOL DE CHILE (no español formal europeo)
2. Usa máximo 2 oraciones cortas
3. Sé empático, cálido y cercano
4. NO uses tecnicismos médicos
5. NO des diagnósticos ni recomendaciones de medicamentos
6. Si la palabra no es clara, pide amablemente más contexto
7. Si detectas saludo, da la bienvenida cálidamente
8. Si detectas agradecimiento o despedida, despídete cordialmente

Contexto conversacional:
{history_context}

Palabra actual del usuario: "{current_word}"

Genera una respuesta empática y útil, recordando que estás en un contexto de orientación en salud pública chilena.

Respuesta:"""

        return prompt

    def generate_response(self, current_word: str, history: List[str]) -> Tuple[str, float]:
        """
        Genera respuesta usando Gemini.

        Args:
            current_word: Palabra detectada actualmente
            history: Historial de palabras previas

        Returns:
            Tuple[str, float]:
                - Respuesta generada por Gemini
                - Latencia en milisegundos

        Raises:
            GeminiChatbotError: Si hay problemas al generar respuesta
        """
        try:
            start_time = time.time()

            # Construir prompt
            prompt = self._build_prompt(current_word, history)

            # Llamar a Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            # Extraer texto de respuesta
            if response and response.text:
                chatbot_response = response.text.strip()
            else:
                raise GeminiChatbotError("Gemini no devolvió una respuesta válida")

            # Calcular latencia
            latency_ms = (time.time() - start_time) * 1000

            return chatbot_response, latency_ms

        except Exception as e:
            raise GeminiChatbotError(f"Error al generar respuesta: {str(e)}")

    def generate_low_confidence_response(self) -> str:
        """
        Genera respuesta cuando la confianza es baja.

        Returns:
            str: Mensaje de fallback
        """
        return "No pude reconocer la seña claramente. ¿Puede repetirla, por favor?"

    def generate_error_fallback(self, error_type: str = "general") -> str:
        """
        Genera respuesta de fallback cuando hay errores.

        Args:
            error_type: Tipo de error

        Returns:
            str: Mensaje de fallback seguro
        """
        fallback_messages = {
            "timeout": "Disculpe, tuve un problema de conexión. ¿Puede intentarlo nuevamente?",
            "api_error": "Lo siento, tengo dificultades técnicas en este momento. Por favor, intente más tarde.",
            "general": "Disculpe, ocurrió un problema. ¿Puede repetir su consulta?"
        }

        return fallback_messages.get(error_type, fallback_messages["general"])


# Instancia global del chatbot
gemini_chatbot = GeminiChatbot()
