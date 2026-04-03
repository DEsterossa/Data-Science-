from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    description: str = Field(..., min_length=1)

class CategoryPrediction(BaseModel):
    category: str
    confidence: float

class PredictionResponse(BaseModel):
    best_category: str
    best_confidence: float
    latency_ms: float
    predictions: List[CategoryPrediction]