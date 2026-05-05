from src.churn.data import clean_telco_data, load_raw_data


def test_raw_dataset_has_expected_columns() -> None:
    """Verifica se o dataset bruto possui colunas essenciais."""
    df = load_raw_data()

    expected_columns = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    assert expected_columns.issubset(set(df.columns))


def test_clean_dataset_has_numeric_target() -> None:
    """Verifica se a coluna alvo foi convertida para formato numérico."""
    raw_df = load_raw_data()
    clean_df = clean_telco_data(raw_df)

    assert clean_df["Churn"].isin([0, 1]).all()