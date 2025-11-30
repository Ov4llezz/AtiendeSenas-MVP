/**
 * Componente LoadingIndicator
 * Indicador de carga mientras se procesa el video
 */

import React from 'react';

const LoadingIndicator: React.FC = () => {
  return (
    <div className="w-full max-w-3xl mx-auto mt-8">
      <div className="bg-blue-50 border-2 border-blue-300 rounded-xl p-8">
        <div className="flex flex-col items-center justify-center">
          {/* Spinner */}
          <div className="relative w-16 h-16 mb-4">
            <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
          </div>

          {/* Texto */}
          <h3 className="text-xl font-semibold text-blue-900 mb-2">
            Procesando video...
          </h3>
          <p className="text-blue-700 text-center">
            Analizando la seña y generando respuesta
          </p>

          {/* Puntos animados */}
          <div className="flex gap-2 mt-4">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingIndicator;
