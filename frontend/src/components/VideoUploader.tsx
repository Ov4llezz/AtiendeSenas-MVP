/**
 * Componente VideoUploader
 * Recuadro central para cargar y previsualizar videos de señas
 */

import React, { useRef, useState } from 'react';

interface VideoUploaderProps {
  onVideoSelect: (file: File) => void;
  disabled: boolean;
}

const VideoUploader: React.FC<VideoUploaderProps> = ({ onVideoSelect, disabled }) => {
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) return;

    // Validar tipo de archivo
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    if (!allowedTypes.includes(file.type)) {
      alert('Formato no soportado. Use un video mp4 o mov.');
      return;
    }

    // Validar tamaño (máx 50MB)
    const maxSizeMB = 50;
    if (file.size > maxSizeMB * 1024 * 1024) {
      alert(`El archivo excede el tamaño máximo permitido (${maxSizeMB} MB)`);
      return;
    }

    // Crear preview
    const url = URL.createObjectURL(file);
    setVideoPreview(url);
    setFileName(file.name);

    // Notificar al componente padre
    onVideoSelect(file);
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Input oculto */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />

      {/* Recuadro de video */}
      <div
        onClick={handleClick}
        className={`
          relative border-4 border-dashed rounded-xl overflow-hidden
          bg-gray-50 transition-all duration-200
          ${disabled ? 'border-gray-300 cursor-not-allowed opacity-60' : 'cursor-pointer hover:opacity-90'}
          ${videoPreview ? 'aspect-video' : 'h-96'}
        `}
        style={!disabled ? { borderColor: '#95B5CF' } : {}}
      >
        {videoPreview ? (
          /* Preview del video */
          <video
            src={videoPreview}
            controls
            className="w-full h-full object-contain bg-black"
          />
        ) : (
          /* Placeholder */
          <div className="flex flex-col items-center justify-center h-full p-8">
            <svg
              className="w-24 h-24 mb-4 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            <h3 className="text-2xl font-semibold text-gray-700 mb-2">
              Toque aquí para cargar un video
            </h3>
            <p className="text-gray-500 text-center">
              Formatos soportados: MP4, MOV
              <br />
              Tamaño máximo: 50 MB
            </p>
          </div>
        )}
      </div>

      {/* Nombre del archivo */}
      {fileName && (
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-600">
            Archivo seleccionado: <span className="font-medium">{fileName}</span>
          </p>
        </div>
      )}
    </div>
  );
};

export default VideoUploader;
