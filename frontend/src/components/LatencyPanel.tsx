/**
 * Componente LatencyPanel
 * Panel pequeño en esquina superior derecha mostrando latencias
 */

import React from 'react';
import { LatencyInfo } from '../types';

interface LatencyPanelProps {
  latency: LatencyInfo;
}

const LatencyPanel: React.FC<LatencyPanelProps> = ({ latency }) => {
  return (
    <div className="fixed top-4 right-4 bg-white border border-gray-300 rounded-lg shadow-md p-3 text-xs w-48">
      <div className="font-semibold text-gray-700 mb-2 flex items-center gap-1">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        Tiempos de Respuesta
      </div>

      <div className="space-y-1">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">VideoMAE:</span>
          <span className="font-mono font-medium text-gray-900">
            {latency.videomae.toFixed(0)} ms
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">Chatbot:</span>
          <span className="font-mono font-medium text-gray-900">
            {latency.chatbot.toFixed(0)} ms
          </span>
        </div>

        <div className="border-t border-gray-200 pt-1 mt-1">
          <div className="flex justify-between items-center">
            <span className="text-gray-700 font-semibold">Total:</span>
            <span className="font-mono font-bold text-blue-600">
              {latency.total.toFixed(0)} ms
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LatencyPanel;
