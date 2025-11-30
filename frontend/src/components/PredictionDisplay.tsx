/**
 * Componente PredictionDisplay
 * Muestra la palabra detectada y su nivel de confianza con colores
 */

import React from 'react';

interface PredictionDisplayProps {
  word: string;
  confidence: number;
}

const PredictionDisplay: React.FC<PredictionDisplayProps> = ({ word, confidence }) => {
  // Determinar color según nivel de confianza
  const getConfidenceColor = (conf: number): string => {
    if (conf >= 0.75) return 'text-green-600 bg-green-50 border-green-300';
    if (conf >= 0.55) return 'text-yellow-600 bg-yellow-50 border-yellow-300';
    return 'text-red-600 bg-red-50 border-red-300';
  };

  const getConfidenceLabel = (conf: number): string => {
    if (conf >= 0.75) return 'Alta';
    if (conf >= 0.55) return 'Media';
    return 'Baja';
  };

  const confidencePercent = (confidence * 100).toFixed(1);
  const colorClass = getConfidenceColor(confidence);
  const label = getConfidenceLabel(confidence);

  return (
    <div className="w-full max-w-3xl mx-auto mt-8">
      <div className={`border-2 rounded-xl p-6 ${colorClass} transition-all duration-300`}>
        {/* Etiqueta */}
        <div className="text-sm font-medium mb-2 uppercase tracking-wide opacity-75">
          Palabra Detectada
        </div>

        {/* Palabra */}
        <div className="text-5xl font-bold mb-4">
          {word}
        </div>

        {/* Confianza */}
        <div className="flex items-center justify-between">
          <div className="text-lg">
            <span className="font-medium">Confianza:</span>{' '}
            <span className="font-bold">{confidencePercent}%</span>
          </div>
          <div className="px-4 py-1 bg-white bg-opacity-50 rounded-full text-sm font-semibold">
            {label}
          </div>
        </div>

        {/* Barra de progreso */}
        <div className="mt-4 w-full bg-white bg-opacity-50 rounded-full h-3 overflow-hidden">
          <div
            className="h-full bg-current transition-all duration-500"
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default PredictionDisplay;
