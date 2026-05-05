import json
import logging
from functools import lru_cache

import joblib
import pandas as pd
import torch

from src.churn.config import MODELS_DIR, PREPROCESSOR_PATH
from src.churn.train_mlp import ChurnMLP

logger = logging.getLogger(__name__)

MLP_MODEL_PATH = MODELS_DIR / "mlp_model.pt"
MLP_METADATA_PATH = MODELS_DIR / "mlp_metadata.json"


class ChurnPredictor:
    """Classe responsável por carregar o modelo e realizar predições de churn."""

    def __init__(self) -> None:
        if not MLP_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MLP_MODEL_PATH}. "
                "Execute primeiro: python -m src.churn.train_mlp"
            )

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor não encontrado em {PREPROCESSOR_PATH}. "
                "Execute primeiro: python -m src.churn.train_mlp"
            )

        if not MLP_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadados não encontrados em {MLP_METADATA_PATH}. "
                "Execute primeiro: python -m src.churn.train_mlp"
            )

        logger.info("Carregando preprocessor")
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        logger.info("Carregando metadados da MLP")
        with MLP_METADATA_PATH.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        logger.info("Carregando modelo MLP")
        checkpoint = torch.load(MLP_MODEL_PATH, map_location=torch.device("cpu"))

        input_size = int(checkpoint["input_size"])
        self.model = ChurnMLP(input_size=input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, input_data: dict) -> dict[str, float | int | str]:
        """Recebe um cliente em formato dict e retorna predição de churn."""
        input_df = pd.DataFrame([input_data])

        processed_data = self.preprocessor.transform(input_df)
        input_tensor = torch.tensor(processed_data, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()

        prediction = int(probability >= 0.5)
        risk_level = self._get_risk_level(probability)

        return {
            "churn_prediction": prediction,
            "churn_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "model_name": "pytorch_mlp",
            "model_version": "1.0.0",
        }

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Classifica o risco com base na probabilidade."""
        if probability >= 0.70:
            return "alto"
        if probability >= 0.40:
            return "medio"
        return "baixo"


@lru_cache
def get_predictor() -> ChurnPredictor:
    """Carrega o preditor uma única vez e reutiliza nas próximas requisições."""
    return ChurnPredictor()