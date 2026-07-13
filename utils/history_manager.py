import pandas as pd
import streamlit as st
from datetime import datetime

from utils.db import get_supabase_client


HISTORY_TABLE = "prediction_history"

# Columns returned when the table is empty / unreachable, so callers
# that do history_df["risk_score"] etc. don't crash on a bare empty df.
EMPTY_HISTORY_COLUMNS = ["timestamp", "risk_score", "confidence", "risk_label"]


def save_prediction(data: dict):
    """
    Persist a single prediction result to the Supabase 'prediction_history' table.
    `data` is expected to contain: risk_score, confidence, risk_label.
    """
    supabase = get_supabase_client()

    row = {
        "timestamp": datetime.now().isoformat(),
        **data
    }

    try:
        supabase.table(HISTORY_TABLE).insert(row).execute()
    except Exception as e:
        st.error(f"Unable to save prediction to the database: {e}")


@st.cache_data(ttl=60)  # short cache so a new prediction shows up quickly after rerun
def load_history() -> pd.DataFrame:
    """
    Load full prediction history from Supabase, most recent first.
    Returns an empty (but correctly-columned) DataFrame on any failure
    so downstream code (`history_df["risk_score"].mean()`, etc.) doesn't break.
    """
    supabase = get_supabase_client()

    try:
        response = (
            supabase.table(HISTORY_TABLE)
            .select("*")
            .order("timestamp", desc=True)
            .execute()
        )
        if response.data:
            return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Unable to load prediction history from the database: {e}")

    return pd.DataFrame(columns=EMPTY_HISTORY_COLUMNS)


def clear_history():
    """
    Delete every row from the Supabase 'prediction_history' table.
    Also clears the load_history() cache so the UI reflects the change
    immediately instead of showing stale cached rows for up to 60s.
    """
    supabase = get_supabase_client()
    try:
        supabase.table(HISTORY_TABLE).delete().neq("timestamp", "").execute()
        load_history.clear()
    except Exception as e:
        st.error(f"Unable to clear prediction history in the database: {e}")
        raise
