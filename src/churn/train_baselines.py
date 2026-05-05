import json
import logging

import joblib
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.churn.config import (
    BASELINE_RESULTS_PATH,
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


def calculate_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calcula métricas principais de classificação binária."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def get_models() -> dict[str, object]:
    """Retorna modelos baseline para comparação."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_and_evaluate_baselines() -> pd.DataFrame:
    """Treina modelos baseline, registra no MLflow e salva resultados."""
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

    logger.info("Criando preprocessor")
    preprocessor = build_preprocessor(X_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("tech_challenge_churn_baselines")

    results = []

    for model_name, model in get_models().items():
        logger.info("Treinando modelo baseline: %s", model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred

            metrics = calculate_metrics(y_test, y_pred, y_proba)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            result = {"model_name": model_name, **metrics}
            results.append(result)

            logger.info("Resultado %s: %s", model_name, metrics)

    logger.info("Treinando preprocessor final para salvar")
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)

    with BASELINE_RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    logger.info("Resultados salvos em: %s", BASELINE_RESULTS_PATH)
    logger.info("Preprocessor salvo em: %s", PREPROCESSOR_PATH)

    print("\nResultados dos baselines:")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    train_and_evaluate_baselines()