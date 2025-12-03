/**
 * Componente LoadingIndicator
 * Indicador de carga mientras se procesa el video
 */

import React from 'react';

const LoadingIndicator: React.FC = () => {
  return (
    <div className="w-full max-w-3xl mx-auto mt-8">
      <div className="bg-gray-50 border-2 rounded-xl p-8" style={{ borderColor: '#95B5CF' }}>
        <div className="flex flex-col items-center justify-center">
          {/* Spinner */}
          <div className="relative w-16 h-16 mb-4">
            <div className="absolute inset-0 border-4 rounded-full" style={{ borderColor: '#95B5CF' }}></div>
            <div className="absolute inset-0 border-4 rounded-full border-t-transparent animate-spin" style={{ borderColor: '#1E4B7D' }}></div>
          </div>

          {/* Texto */}
          <h3 className="text-xl font-semibold mb-2" style={{ color: '#1E4B7D' }}>
            Procesando video...
          </h3>
          <p className="text-gray-700 text-center">
            Analizando la seña y generando respuesta
          </p>

          {/* Puntos animados */}
          <div className="flex gap-2 mt-4">
            <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: '#1E4B7D', animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: '#1E4B7D', animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: '#1E4B7D', animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingIndicator;
