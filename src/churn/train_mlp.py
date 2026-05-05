import json
import logging
import random

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.churn.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.churn.data import clean_telco_data, load_raw_data, split_features_target
from src.churn.preprocessing import build_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

MLP_MODEL_PATH = MODELS_DIR / "mlp_model.pt"
MLP_METADATA_PATH = MODELS_DIR / "mlp_metadata.json"


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fixa seeds para melhorar a reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChurnMLP(nn.Module):
    """Rede neural MLP simples para classificação binária de churn."""

    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass da rede."""
        return self.network(x).squeeze(1)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calcula métricas principais de classificação binária."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Cria DataLoader do PyTorch."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: ChurnMLP,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Treina o modelo por uma época."""
    model.train()
    losses = []

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def evaluate_loss(model: ChurnMLP, dataloader: DataLoader, loss_fn: nn.Module) -> float:
    """Calcula loss de validação."""
    model.eval()
    losses = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            losses.append(loss.item())

    return float(np.mean(losses))


def predict_probabilities(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    """Gera probabilidades de churn."""
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).numpy()

    return probabilities


def train_mlp() -> pd.DataFrame:
    """Treina uma MLP em PyTorch, registra no MLflow e salva o modelo."""
    set_seed()

    logger.info("Carregando dataset bruto")
    raw_df = load_raw_data()

    logger.info("Limpando dataset")
    df = clean_telco_data(raw_df)

    logger.info("Separando features e target")
    X, y = split_features_target(df)

    logger.info("Separando treino e teste com estratificação")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("Criando e aplicando preprocessor")
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    input_size = X_train_processed.shape[1]

    batch_size = 64
    learning_rate = 0.001
    max_epochs = 100
    patience = 10

    train_loader = create_dataloader(
        X_train_processed,
        y_train_np,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        X_test_processed,
        y_test_np,
        batch_size=batch_size,
        shuffle=False,
    )

    model = ChurnMLP(input_size=input_size)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("tech_challenge_churn_mlp")

    with mlflow.start_run(run_name="pytorch_mlp"):
        mlflow.log_param("model_name", "pytorch_mlp")
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("random_state", RANDOM_STATE)

        logger.info("Iniciando treinamento da MLP")

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_loss = evaluate_loss(model, val_loader, loss_fn)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            logger.info(
                "Epoch %s | train_loss=%.4f | val_loss=%.4f",
                epoch,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping acionado na epoch %s", epoch)
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        y_proba = predict_probabilities(model, X_test_processed)
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = calculate_metrics(y_test_np, y_pred, y_proba)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
            },
            MLP_MODEL_PATH,
        )

        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        metadata = {
            "model_name": "pytorch_mlp",
            "input_size": int(input_size),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
        }

        with MLP_METADATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        mlflow.log_artifact(str(MLP_MODEL_PATH))
        mlflow.log_artifact(str(PREPROCESSOR_PATH))
        mlflow.log_artifact(str(MLP_METADATA_PATH))

    results_df = pd.DataFrame([{"model_name": "pytorch_mlp", **metrics}])

    print("\nResultado da MLP:")
    print(results_df.to_string(index=False))

    logger.info("Modelo MLP salvo em: %s", MLP_MODEL_PATH)
    logger.info("Preprocessor salvo em: %s", PREPROCESSOR_PATH)
    logger.info("Metadados salvos em: %s", MLP_METADATA_PATH)

    return results_df


if __name__ == "__main__":
    train_mlp()