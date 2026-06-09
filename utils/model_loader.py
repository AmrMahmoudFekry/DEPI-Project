import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from utils.helper import REQUIRED_COLUMNS


# =========================================================
# MODEL LOADING
# =========================================================
MODEL_PATH = Path(__file__).resolve().parent.parent / "pipeline.pkl"
_PIPELINE_CACHE = None


def load_pipeline() -> joblib:
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. "
            f"Run `python modeling_pipeline.py` first."
        )

    try:
        _PIPELINE_CACHE = joblib.load(MODEL_PATH)
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to load pipeline.pkl because it was built with a local "
            "transformer class. Delete the old pipeline.pkl and rerun "
            "`python modeling_pipeline.py` to rebuild it."
        ) from exc

    return _PIPELINE_CACHE


# =========================================================
# BILINGUAL RISK LABELS & EXPLANATIONS
# =========================================================
RISK_TEXTS = {
    "en": {
        "low_label":          "LOW RISK",
        "medium_label":       "MEDIUM RISK",
        "high_label":         "HIGH RISK",
        "low_explanation":    (
            "The business demonstrates strong financial stability "
            "with minimal risk exposure."
        ),
        "medium_explanation": (
            "Moderate financial risk detected. "
            "Some indicators require monitoring."
        ),
        "high_explanation":   (
            "High financial risk detected. "
            "Multiple indicators suggest instability."
        ),
    },
    "ar": {
        "low_label":          "مخاطر منخفضة",
        "medium_label":       "مخاطر متوسطة",
        "high_label":         "مخاطر عالية",
        "low_explanation":    (
            "تُظهر الشركة استقرارًا ماليًا قويًا "
            "مع تعرض محدود للمخاطر."
        ),
        "medium_explanation": (
            "تم رصد مخاطر مالية متوسطة. "
            "بعض المؤشرات تتطلب المراقبة."
        ),
        "high_explanation":   (
            "تم رصد مخاطر مالية مرتفعة. "
            "مؤشرات متعددة تشير إلى عدم الاستقرار."
        ),
    }
}


# =========================================================
# VALIDATION HELPER
# =========================================================
def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Raise KeyError if any required input column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
    if missing:
        raise KeyError(
            f"Missing required model input features: {', '.join(missing)}"
        )


# =========================================================
# SINGLE PREDICTION
# =========================================================
def predict_risk(pipeline, dataframe, lang="en"):
    """
    Pass the raw dataframe directly to the pipeline.
    The pipeline's SMEFeatureEngineer step handles all
    feature engineering internally — do NOT reorder or
    drop columns before calling predict.
    """
    _validate_required_columns(dataframe)

    prediction  = pipeline.predict(dataframe)[0]
    probability = pipeline.predict_proba(dataframe)[0][1]

    risk_score = round(probability * 100, 2)
    confidence = round(max(probability, 1 - probability) * 100, 1)

    tx = RISK_TEXTS.get(lang, RISK_TEXTS["en"])

    if risk_score < 40:
        color       = "green"
        risk_label  = tx["low_label"]
        explanation = tx["low_explanation"]
    elif risk_score < 70:
        color       = "orange"
        risk_label  = tx["medium_label"]
        explanation = tx["medium_explanation"]
    else:
        color       = "red"
        risk_label  = tx["high_label"]
        explanation = tx["high_explanation"]

    result = {
        "prediction":    int(prediction),
        "risk_score":    risk_score,
        "confidence":    confidence,
        "risk_label":    risk_label,
        "explanation":   explanation,
        "color":         color,
        "risk_label_en": (
            "LOW RISK"    if risk_score < 40
            else "MEDIUM RISK" if risk_score < 70
            else "HIGH RISK"
        ),
    }

    return result, prediction


# =========================================================
# BATCH PREDICTION
# =========================================================
def predict_batch_risk(pipeline, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Batch scoring — English labels only (written to CSV report).
    Pass raw dataframe straight to the pipeline; feature engineering
    happens inside the pipeline's SMEFeatureEngineer step.
    """
    _validate_required_columns(dataframe)

    df            = dataframe.copy()
    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions   = pipeline.predict(df)
    risk_scores   = np.round(probabilities * 100, 2)
    confidences   = np.round(
        np.maximum(probabilities, 1 - probabilities) * 100, 1
    )

    labels = []
    for score in risk_scores:
        if score < 40:
            labels.append("LOW RISK")
        elif score < 70:
            labels.append("MEDIUM RISK")
        else:
            labels.append("HIGH RISK")

    result_df               = dataframe.copy()
    result_df["prediction"] = predictions
    result_df["risk_score"] = risk_scores
    result_df["confidence"] = confidences
    result_df["risk_label"] = labels

    return result_df
