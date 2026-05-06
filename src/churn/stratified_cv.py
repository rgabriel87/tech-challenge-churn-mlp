import json
import logging

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.churn.config import MODELS_DIR, RANDOM_STATE
from src.churn.data import clean_telco_data, load_raw_data, split_features_target
from src.churn.preprocessing import build_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

STRATIFIED_CV_RESULTS_PATH = MODELS_DIR / "stratified_cv_results.json"


def get_models() -> dict[str, object]:
    """Retorna modelos para validação cruzada estratificada."""
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


def run_stratified_cross_validation() -> pd.DataFrame:
    """Executa validação cruzada estratificada para os modelos baseline."""
    logger.info("Carregando dataset bruto")
    raw_df = load_raw_data()

    logger.info("Limpando dataset")
    df = clean_telco_data(raw_df)

    logger.info("Separando features e target")
    X, y = split_features_target(df)

    logger.info("Configurando StratifiedKFold")
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    mlflow.set_experiment("tech_challenge_churn_stratified_cv")

    results = []

    for model_name, model in get_models().items():
        logger.info("Executando validação cruzada estratificada: %s", model_name)

        preprocessor = build_preprocessor(X)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=f"{model_name}_stratified_cv"):
            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            result = {
                "model_name": model_name,
                "cv_folds": 5,
                "accuracy_mean": float(cv_results["test_accuracy"].mean()),
                "accuracy_std": float(cv_results["test_accuracy"].std()),
                "precision_mean": float(cv_results["test_precision"].mean()),
                "precision_std": float(cv_results["test_precision"].std()),
                "recall_mean": float(cv_results["test_recall"].mean()),
                "recall_std": float(cv_results["test_recall"].std()),
                "f1_mean": float(cv_results["test_f1"].mean()),
                "f1_std": float(cv_results["test_f1"].std()),
                "roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
                "roc_auc_std": float(cv_results["test_roc_auc"].std()),
                "pr_auc_mean": float(cv_results["test_pr_auc"].mean()),
                "pr_auc_std": float(cv_results["test_pr_auc"].std()),
            }

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv_strategy", "StratifiedKFold")
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", RANDOM_STATE)

            for metric_name, metric_value in result.items():
                if metric_name != "model_name":
                    mlflow.log_metric(metric_name, metric_value)

            results.append(result)

            logger.info("Resultado CV %s: %s", model_name, result)

    results_df = pd.DataFrame(results).sort_values(by="roc_auc_mean", ascending=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with STRATIFIED_CV_RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    logger.info("Resultados da validação cruzada salvos em: %s", STRATIFIED_CV_RESULTS_PATH)

    print("\nResultados da validação cruzada estratificada:")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    run_stratified_cross_validation()