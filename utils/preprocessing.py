# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

ENCODER_PATH = "encoders"

def clean_and_encode_data(df, encoders=None, save_encoders=False):
    """
    Cleans and encodes the dataset for machine learning.

    Parameters:
    - df: DataFrame
    - encoders: dict of fitted LabelEncoders (optional, used for inference)
    - save_encoders: bool, whether to save fitted encoders
    """
    df = df.copy()

    # Drop columns that won't be used
    drop_cols = ['contact', 'month', 'day_of_week', 'pdays', 'poutcome']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if save_encoders and not os.path.exists(ENCODER_PATH):
        os.makedirs(ENCODER_PATH)
    if encoders is None:  # Training mode: create new encoders
        encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            if save_encoders:
                joblib.dump(le, os.path.join(ENCODER_PATH, f"{col}_encoder.pkl"))
    else:  # Use provided or loaded encoders
        for col in df.select_dtypes(include=['object']).columns:
            if col not in encoders:
                raise ValueError(f"Encoder for column '{col}' not found in provided encoders.")
            le = encoders[col]
            # Handle unseen labels by assigning -1
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    return df, encoders