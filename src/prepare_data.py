
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "athletes.csv"

# Baseline features
V1 = [
    "age",
    "height",
    "weight",
    "gender_male",
    "backsq",
    "candj",
    "snatch"
]

# Engineered features
V2 = V1 + [
    "bmi",
    "backsq_to_weight",
    "candj_to_weight",
    "snatch_to_weight",
    "strength_total"
]


def prepare_data(limit=30000):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    columns = [
        "athlete_id", "gender", "age",
        "height", "weight", "backsq",
        "candj", "snatch", "deadlift"
    ]

    df = df[columns].copy()

    # Convert numeric fields
    for col in columns:
        if col != "gender":
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Require valid IDs and prediction targets
    df = df.dropna(
        subset=["athlete_id", "deadlift"]
    )

    df = df[
        df["deadlift"].between(50, 1000)
    ].copy()

    # Retain plausible ages, allowing missing ages
    df.loc[
        ~df["age"].between(14, 90), "age"
    ] = np.nan

    # Remove duplicate athlete records
    df = df.drop_duplicates(
        subset="athlete_id"
    )

    # Mark implausible predictor values as missing
    bounds = {
        "height": (48, 90),
        "weight": (75, 500),
        "backsq": (20, 1000),
        "candj": (10, 700),
        "snatch": (10, 600)
    }

    for col, (low, high) in bounds.items():
        df.loc[
            ~df[col].between(low, high), col
        ] = np.nan

    # Encode gender
    df["gender_male"] = df["gender"].map({
        "Male": 1.0,
        "Female": 0.0
    })

    # Feature engineering: Version 2
    df["bmi"] = (
        df["weight"] * 703
        / df["height"].pow(2)
    )

    for col in ["backsq", "candj", "snatch"]:
        df[f"{col}_to_weight"] = (
            df[col] / df["weight"]
        )

    df["strength_total"] = df[
        ["backsq", "candj", "snatch"]
    ].sum(axis=1, min_count=3)

    df = df.replace(
        [np.inf, -np.inf], np.nan
    )

    # Reproducible sample
    df = df.sample(
        frac=1, random_state=42
    )

    if limit is not None:
        df = df.head(limit)

    df = df.copy()

    df["athlete_id"] = (
        df["athlete_id"].astype("int64")
    )

    # Static timestamps for Feast demonstration
    df["event_timestamp"] = pd.Timestamp(
        "2025-01-01", tz="UTC"
    )

    df["created_timestamp"] = pd.Timestamp(
        "2025-01-02", tz="UTC"
    )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = prepare_data()

    print("Cleaned dataset:", df.shape)
    print("\nFeature V1:", V1)
    print("\nFeature V2:", V2)

    print("\nMissing values:")
    print(df[V2].isnull().sum())

    print("\nTarget summary:")
    print(df["deadlift"].describe())

    print("\nFirst 5 rows:")
    print(
        df[
            ["athlete_id", "deadlift"] + V1
        ].head().to_string()
    )
