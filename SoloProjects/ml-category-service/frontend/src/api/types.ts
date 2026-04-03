// Mirrors backend Pydantic models in app/schemas.py

export interface CategoryPrediction {
  category: string
  confidence: number
}

export interface PredictionResponse {
  best_category: string
  best_confidence: number
  latency_ms: number
  predictions: CategoryPrediction[]
}
