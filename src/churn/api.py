import logging
import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from src.churn.predict import ChurnPredictor, get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn usando rede neural MLP em PyTorch.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    """Dados de entrada esperados para predição de churn."""

    gender: str = Field(example="Female")
    SeniorCitizen: int = Field(example=0, ge=0, le=1)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=12, ge=0)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="Yes")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="Yes")
    StreamingMovies: str = Field(example="Yes")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=70.35, ge=0)
    TotalCharges: float = Field(example=845.50, ge=0)


class PredictionResponse(BaseModel):
    """Resposta retornada pela API após a predição."""

    churn_prediction: int
    churn_probability: float
    risk_level: str
    model_name: str
    model_version: str


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    """Adiciona o tempo de processamento no header da resposta."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


@app.get("/")
def root() -> dict[str, str]:
    """Endpoint inicial da API."""
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health(
    predictor: Annotated[ChurnPredictor, Depends(get_predictor)],
) -> dict[str, str | bool]:
    """Endpoint de saúde da API."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_name": "pytorch_mlp",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerData,
    predictor: Annotated[ChurnPredictor, Depends(get_predictor)],
) -> dict[str, float | int | str]:
    """Realiza predição de churn para um cliente."""
    logger.info("Recebida requisição de predição")

    input_data = customer.model_dump()
    result = predictor.predict(input_data)

    logger.info("Predição finalizada: %s", result)

    return result