import pandas as pd

from utils.history_manager import load_history


def get_dashboard_stats():
    """
    Summary stats for the sidebar/dashboard, now sourced from the
    Supabase 'prediction_history' table instead of a local CSV.
    """
    df = load_history()

    if df.empty:
        return {
            "total_predictions": 0,
            "high_risk": 0,
            "avg_confidence": 0
        }

    total = len(df)

    high_risk = len(
        df[df["risk_label"] == "HIGH RISK"]
    )

    # Guard against NaN when confidence column exists but is empty
    confidence_series = df["confidence"].dropna() if "confidence" in df.columns else pd.Series(dtype=float)

    if len(confidence_series) == 0:
        avg_conf = 0
    else:
        avg_conf = round(confidence_series.mean(), 1)

    return {
        "total_predictions": total,
        "high_risk": high_risk,
        "avg_confidence": avg_conf
    }
