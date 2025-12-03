/**
 * Componente principal App
 * Orquesta todo el flujo del tótem de autoatención
 */

import { useState } from 'react';
import axios from 'axios';
import VideoUploader from './components/VideoUploader';
import PredictionDisplay from './components/PredictionDisplay';
import ChatResponseDisplay from './components/ChatResponseDisplay';
import LatencyPanel from './components/LatencyPanel';
import LoadingIndicator from './components/LoadingIndicator';
import { PipelineResponse } from './types';
import logo from './logo.png';

// URL del backend
const API_BASE_URL = 'http://localhost:8000';

function App() {
  // Estados
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handler: Cuando el usuario selecciona un video
  const handleVideoSelect = async (file: File) => {
    setSelectedFile(file);
    setError(null);

    // Enviar automáticamente al backend
    await processVideo(file);
  };

  // Función principal: Enviar video al backend
  const processVideo = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Crear FormData
      const formData = new FormData();
      formData.append('video', file);

      // Llamar al endpoint
      const response = await axios.post<PipelineResponse>(
        `${API_BASE_URL}/api/full-pipeline`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 60000, // 60 segundos
        }
      );

      // Guardar resultado
      setResult(response.data);

    } catch (err: any) {
      console.error('Error al procesar video:', err);

      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ECONNABORTED') {
        setError('El servidor tardó demasiado en responder. Por favor, intente nuevamente.');
      } else if (err.message) {
        setError(`Error: ${err.message}`);
      } else {
        setError('Hubo un problema al procesar el video. Por favor intente nuevamente.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handler: Resetear y subir otro video
  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  // Handler: Limpiar historial de conversación
  const handleClearHistory = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/reset-history`);
      // Resetear resultado para reflejar el historial limpio
      if (result) {
        setResult({ ...result, history: [] });
      }
    } catch (err) {
      console.error('Error al limpiar historial:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 py-8 px-4 relative">
      {/* Logo en esquina superior izquierda */}
      <div className="absolute top-6 left-6 z-10">
        <img
          src={logo}
          alt="Logo AtiendeSenas"
          className="h-32 w-auto object-contain"
        />
      </div>

      {/* Header */}
      <header className="text-center mb-8">
        <h1 className="text-5xl font-bold text-gray-800 mb-2">
          Tótem de Autoatención
        </h1>
        <p className="text-xl text-gray-600">
          Sistema de Reconocimiento de Lengua de Señas Chilena
        </p>
      </header>

      {/* Panel de latencias (solo si hay resultado) */}
      {result && <LatencyPanel latency={result.latency_ms} />}

      {/* Contenedor principal */}
      <main className="container mx-auto max-w-5xl">
        {/* Uploader de video */}
        <VideoUploader
          onVideoSelect={handleVideoSelect}
          disabled={isLoading}
        />

        {/* Indicador de carga */}
        {isLoading && <LoadingIndicator />}

        {/* Error */}
        {error && (
          <div className="w-full max-w-3xl mx-auto mt-6">
            <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <svg
                  className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div>
                  <h3 className="text-lg font-semibold text-red-800 mb-1">
                    Error
                  </h3>
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Resultados */}
        {result && !isLoading && (
          <>
            {/* Palabra detectada */}
            <PredictionDisplay
              word={result.predicted_word}
              confidence={result.confidence}
            />

            {/* Respuesta del chatbot */}
            <ChatResponseDisplay response={result.chatbot_response} />

            {/* Historial (opcional, para debugging) */}
            {result.history && result.history.length > 0 && (
              <div className="w-full max-w-3xl mx-auto mt-4">
                <div className="bg-gray-100 rounded-lg p-4 text-sm">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-gray-700">
                      Historial de conversación:
                    </div>
                    <button
                      onClick={handleClearHistory}
                      className="px-4 py-2 text-sm font-semibold text-white rounded-lg transition-colors duration-200 shadow-sm hover:opacity-90"
                      style={{ backgroundColor: '#1E4B7D' }}
                    >
                      Limpiar historial
                    </button>
                  </div>
                  <div className="text-gray-600">
                    {result.history.join(' → ')}
                  </div>
                </div>
              </div>
            )}

            {/* Botón para subir otro video */}
            <div className="w-full max-w-3xl mx-auto mt-6 text-center">
              <button
                onClick={handleReset}
                className="px-8 py-3 text-white font-semibold rounded-lg transition-colors duration-200 shadow-md hover:opacity-90"
                style={{ backgroundColor: '#1E4B7D' }}
              >
                Subir otro video
              </button>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center mt-12 text-gray-500 text-sm">
        <p>Tótem LSCh v1.0.0 - Sistema de Reconocimiento de Señas</p>
        <p className="mt-1">VideoMAE + Gemini Chatbot</p>
        <p className="mt-4 text-lg font-medium italic" style={{ color: '#95B5CF' }}>
          Entender y ser entendido
        </p>
      </footer>
    </div>
  );
}

export default App;
