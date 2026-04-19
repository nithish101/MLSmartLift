"""
Train a Gradient Boosting regression model for SmartLift rep prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "smartlift_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "exercise_encoder.pkl")
DATA_PATH = os.path.join(MODEL_DIR, "training_data.csv")

FEATURE_COLS = [
    "exercise_encoded",
    "weight",
    "rir",
    "bodyweight",
    "num_sets",
    "frequency",
    "volume_score",
    "progression_trend",
]


def train_model():
    """Train the SmartLift prediction model."""
    if not os.path.exists(DATA_PATH):
        print("Training data not found. Generating...")
        from ml.generate_data import generate_dataset
        generate_dataset()

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")

    # Encode exercise
    le = LabelEncoder()
    df["exercise_encoded"] = le.fit_transform(df["exercise"])

    X = df[FEATURE_COLS]
    y = df["reps"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )

    print("Training Gradient Boosting model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Model Performance ---")
    print(f"MAE:  {mae:.2f} reps")
    print(f"R²:   {r2:.4f}")

    # Feature importance
    importances = model.feature_importances_
    print(f"\n--- Feature Importance ---")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"  {name:25s} {imp:.4f}")

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Encoder saved → {ENCODER_PATH}")

    return model, le


if __name__ == "__main__":
    train_model()
