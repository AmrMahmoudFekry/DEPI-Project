import pandas as pd
import os
from datetime import datetime


HISTORY_FILE = "prediction_history.csv"


def save_prediction(data):

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data
    }

    df = pd.DataFrame([row])

    if os.path.exists(HISTORY_FILE):
        old_df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(HISTORY_FILE, index=False)


def load_history():

    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()