"""
Módulo de Historial Conversacional
Responsabilidades:
- Mantener historial de últimas 3 glosas
- Detectar saludos/despedidas para reset
- Actualizar historial con nuevas glosas
"""

from typing import List
from config import config


class ConversationHistory:
    """Gestiona el historial de conversación con el usuario"""

    def __init__(self):
        """Inicializa historial vacío"""
        self.history: List[str] = []

    def should_reset(self, word: str) -> bool:
        """
        Determina si se debe resetear el historial basado en la palabra detectada.

        Args:
            word: Palabra detectada (en inglés)

        Returns:
            bool: True si se debe resetear el historial
        """
        word_upper = word.upper()

        # Reset en saludos
        if word_upper in config.GREETING_WORDS:
            return True

        # Reset en despedidas
        if word_upper in config.FAREWELL_WORDS:
            return True

        return False

    def reset(self) -> None:
        """Reinicia el historial conversacional"""
        self.history = []

    def add_word(self, word: str) -> None:
        """
        Agrega una palabra al historial.
        Mantiene máximo 3 palabras (elimina la más antigua si es necesario).

        Args:
            word: Palabra a agregar (glosa en inglés)
        """
        self.history.append(word)

        # Mantener solo las últimas 3
        if len(self.history) > config.MAX_HISTORY_LENGTH:
            self.history = self.history[-config.MAX_HISTORY_LENGTH:]

    def get_history(self) -> List[str]:
        """
        Obtiene el historial actual.

        Returns:
            List[str]: Lista de palabras en el historial
        """
        return self.history.copy()

    def get_history_string(self) -> str:
        """
        Obtiene el historial como string para el prompt del chatbot.

        Returns:
            str: Historial formateado como string separado por comas
        """
        if not self.history:
            return "Ninguna conversación previa"

        return ", ".join(self.history)

    def update(self, word: str, confidence: float) -> List[str]:
        """
        Actualiza el historial con una nueva palabra.

        Args:
            word: Palabra detectada
            confidence: Confianza de la predicción

        Returns:
            List[str]: Historial actualizado
        """
        # Verificar si se debe resetear
        if self.should_reset(word):
            self.reset()

        # Solo agregar si la confianza es suficiente
        if confidence >= config.MIN_CONFIDENCE:
            self.add_word(word)

        return self.get_history()

    def __len__(self) -> int:
        """Retorna la longitud del historial"""
        return len(self.history)

    def __str__(self) -> str:
        """Representación en string del historial"""
        return f"ConversationHistory({len(self.history)} palabras: {self.history})"
