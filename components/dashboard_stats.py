import pandas as pd
import os


def get_dashboard_stats():

    if not os.path.exists(
        "prediction_history.csv"
    ):
        return {
            "total_predictions": 0,
            "high_risk": 0,
            "avg_confidence": 0
        }

    df = pd.read_csv(
        "prediction_history.csv"
    )

    # Handle empty CSV (headers only, no rows)
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
    confidence_series = df["confidence"].dropna()

    if len(confidence_series) == 0:
        avg_conf = 0
    else:
        avg_conf = round(confidence_series.mean(), 1)

    return {
        "total_predictions": total,
        "high_risk": high_risk,
        "avg_confidence": avg_conf
    }