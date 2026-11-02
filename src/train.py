
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from feast import FeatureStore
from codecarbon import EmissionsTracker

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_data import prepare_data, V1, V2

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CONFIGS = {
    "A": {
        "n_estimators": 50,
        "max_depth": 8,
        "min_samples_leaf": 5
    },
    "B": {
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_leaf": 2
    }
}


def make_plots(y_true, y_pred, name):
    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.25, s=10)

    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())

    ax.plot([low, high], [low, high], "r--")
    ax.set_xlabel("Actual deadlift (lb)")
    ax.set_ylabel("Predicted deadlift (lb)")
    ax.set_title(f"Actual vs Predicted: {name}")

    fig.tight_layout()
    path1 = OUT / f"{name}_actual_vs_predicted.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)

    # Residual distribution
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residuals, bins=40)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (lb)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution: {name}")

    fig.tight_layout()
    path2 = OUT / f"{name}_residuals.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)

    return path1, path2


def run_experiments():
    df = prepare_data(limit=30000)

    store = FeatureStore(
        repo_path=str(ROOT / "feature_repo")
    )

    train_ids, test_ids = train_test_split(
        df["athlete_id"].to_numpy(),
        test_size=0.2,
        random_state=42
    )

    target = df.set_index("athlete_id")["deadlift"]

    # Use SQLite for MLflow experiment tracking
    mlflow.set_tracking_uri(
        f"sqlite:///{ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("athlete_feature_versions")

    results = []

    for version, columns in [
        ("v1", V1),
        ("v2", V2)
    ]:
        view = f"athlete_{version}"

        retrieved = store.get_historical_features(
            entity_df=df[
                ["athlete_id", "event_timestamp"]
            ],
            features=[
                f"{view}:{col}" for col in columns
            ]
        ).to_df().set_index("athlete_id")

        X_train = retrieved.loc[train_ids, columns]
        X_test = retrieved.loc[test_ids, columns]

        y_train = target.loc[train_ids]
        y_test = target.loc[test_ids]

        for config_name, params in CONFIGS.items():
            name = f"{version}_{config_name}"
            print(f"\nRunning experiment: {name}")

            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "feature_version": version,
                    "algorithm": "RandomForestRegressor",
                    "feature_count": len(columns),
                    **params
                })

                model = Pipeline([
                    ("imputer", SimpleImputer(
                        strategy="median"
                    )),
                    ("regressor", RandomForestRegressor(
                        **params,
                        random_state=42,
                        n_jobs=2
                    ))
                ])

                tracker = EmissionsTracker(
                    project_name=name,
                    output_dir=str(OUT),
                    output_file=f"emissions_{name}.csv",
                    log_level="error",
                    allow_multiple_runs=True
                )

                start = time.perf_counter()
                tracker.start()

                try:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                finally:
                    emissions = tracker.stop()

                duration = time.perf_counter() - start

                metrics = {
                    "mae": mean_absolute_error(
                        y_test, predictions
                    ),
                    "rmse": np.sqrt(
                        mean_squared_error(
                            y_test, predictions
                        )
                    ),
                    "r2": r2_score(
                        y_test, predictions
                    ),
                    "duration_seconds": duration,
                    "emissions_kg_co2e": (
                        emissions if emissions is not None
                        else np.nan
                    )
                }

                mlflow.log_metrics({
                    k: float(v)
                    for k, v in metrics.items()
                    if np.isfinite(v)
                })

                plots = make_plots(
                    y_test.to_numpy(),
                    predictions,
                    name
                )

                for plot in plots:
                    mlflow.log_artifact(str(plot))

                
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    skops_trusted_types=[
                        "numpy.dtype",
                        "sklearn.tree._tree.Tree"
                    ]
                )


                results.append({
                    "experiment": name,
                    "feature_version": version,
                    "configuration": config_name,
                    **metrics
                })

                print(
                    f"MAE: {metrics['mae']:.2f} lb"
                )
                print(
                    f"RMSE: {metrics['rmse']:.2f} lb"
                )
                print(
                    f"R²: {metrics['r2']:.4f}"
                )
                print(
                    f"Duration: {duration:.2f} sec"
                )
                print(
                    "Emissions: "
                    f"{metrics['emissions_kg_co2e']:.8f} kg CO2e"
                )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUT / "comparison.csv",
        index=False
    )

    print("\nAll experiments completed!")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_experiments()
