from pydantic import BaseModel, Field, field_validator


class PredictionItem(BaseModel):
    category: str
    confidence: float


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Product title")

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be empty")
        return value


class PredictResponse(BaseModel):
    best_category: str
    best_confidence: float
    latency_ms: float
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str