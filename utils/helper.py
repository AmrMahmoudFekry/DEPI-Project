import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


REQUIRED_COLUMNS = [
    'credit_amount', 'business_age_months', 'monthly_income_avg',
    'total_deposits_3m', 'revenue_volatility_3m', 'request_ratio',
    'dti_monthly', 'nsf_count_3m', 'negative_days_3m',
    'owner_percentage', 'owner_credit_score'
]


class SMEFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer for the model's feature engineering logic."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        return create_features(df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
    )
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df).copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required input columns: {', '.join(missing)}"
        )

    eps = 1e-5

    if 'credit_income_ratio' not in df.columns:
        df['credit_income_ratio'] = (
            df['credit_amount'] / (df['monthly_income_avg'] + eps)
        )

    if 'deposit_coverage' not in df.columns:
        df['deposit_coverage'] = (
            df['total_deposits_3m'] / (df['credit_amount'] + eps)
        )

    if 'income_deposit_ratio' not in df.columns:
        df['income_deposit_ratio'] = (
            df['monthly_income_avg'] / (df['total_deposits_3m'] / 3 + eps)
        )

    if 'revenue_stability' not in df.columns:
        df['revenue_stability'] = 1.0 / (df['revenue_volatility_3m'] + eps)

    if 'age_stability_score' not in df.columns:
        vol = df['revenue_volatility_3m'].clip(0.0, 1.0)
        df['age_stability_score'] = (
            df['business_age_months'] * (1 - vol)
        )

    if 'financial_stress' not in df.columns:
        df['financial_stress'] = (
            df['dti_monthly'] * df['request_ratio']
        )

    if 'nsf_risk_ratio' not in df.columns:
        df['nsf_risk_ratio'] = (
            df['nsf_count_3m'] / (df['business_age_months'] + eps)
        )

    if 'negative_activity_ratio' not in df.columns:
        df['negative_activity_ratio'] = (
            df['negative_days_3m'] / 90.0
        )

    if 'owner_reliability' not in df.columns:
        df['owner_reliability'] = (
            (df['owner_credit_score'] / 850.0) *
            (df['owner_percentage'] / 100.0)
        )

    if 'nsf_dti_interaction' not in df.columns:
        df['nsf_dti_interaction'] = (
            df['nsf_count_3m'] * df['dti_monthly']
        )

    if 'deposit_nsf_ratio' not in df.columns:
        df['deposit_nsf_ratio'] = (
            df['total_deposits_3m'] / (df['nsf_count_3m'] + 1)
        )

    if 'liquidity_stress_ratio' not in df.columns:
        df['liquidity_stress_ratio'] = (
            df['deposit_coverage'] / (df['financial_stress'] + eps)
        )

    return df



def prepare_input(data):
    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    else:
        input_df = pd.DataFrame(data)

    input_df = standardize_columns(input_df)
    input_df = create_features(input_df)
    return input_df