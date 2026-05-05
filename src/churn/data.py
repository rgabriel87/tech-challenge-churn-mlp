import pandas as pd

from src.churn.config import ID_COLUMN, RAW_DATA_PATH, TARGET_COLUMN


def load_raw_data() -> pd.DataFrame:
    """Carrega o dataset bruto de churn."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado em: {RAW_DATA_PATH}. "
            "Verifique se o CSV está em data/raw/."
        )

    return pd.read_csv(RAW_DATA_PATH)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza básica no dataset Telco Churn."""
    df_clean = df.copy()

    if ID_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[ID_COLUMN])

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())

    df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    return df_clean


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variáveis explicativas e variável alvo."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y