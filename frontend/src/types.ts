/**
 * Tipos TypeScript para la API del Tótem LSCh
 */

export interface LatencyInfo {
  videomae: number;
  chatbot: number;
  total: number;
}

export interface PipelineResponse {
  predicted_word: string;
  confidence: number;
  chatbot_response: string;
  history: string[];
  latency_ms: LatencyInfo;
}

export interface ApiError {
  detail: string;
}
