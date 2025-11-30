/**
 * Componente ChatResponseDisplay
 * Muestra la respuesta empática del chatbot Gemini
 */

import React from 'react';

interface ChatResponseDisplayProps {
  response: string;
}

const ChatResponseDisplay: React.FC<ChatResponseDisplayProps> = ({ response }) => {
  return (
    <div className="w-full max-w-3xl mx-auto mt-6">
      <div className="bg-white border-2 border-blue-200 rounded-xl p-8 shadow-lg">
        {/* Icono del asistente */}
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
              <svg
                className="w-7 h-7 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
            </div>
          </div>

          {/* Contenido */}
          <div className="flex-1">
            <div className="text-sm font-medium text-gray-500 mb-2">
              Asistente de Salud
            </div>
            <div className="text-2xl text-gray-800 leading-relaxed">
              {response}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatResponseDisplay;
