
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

df = pd.read_csv(OUT / "comparison.csv")

# Convert emissions from kg to mg for readability
df["emissions_mg"] = df["emissions_kg_co2e"] * 1_000_000

metrics = {
    "mae": ("Mean Absolute Error", "MAE (lb)"),
    "rmse": ("Root Mean Squared Error", "RMSE (lb)"),
    "r2": ("R-squared", "R²"),
    "duration_seconds": ("Execution Duration", "Seconds"),
    "emissions_mg": ("Estimated Carbon Emissions", "mg CO₂e")
}

for metric, (title, ylabel) in metrics.items():
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        df["experiment"],
        df[metric],
        color=["#487F82", "#84B1B0", "#487F82", "#84B1B0"]
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Experiment")

    ax.bar_label(bars, fmt="%.3f", padding=3)

    ax.set_ylim(0, df[metric].max() * 1.2)

    fig.tight_layout()
    fig.savefig(
        OUT / f"compare_{metric}.png",
        dpi=150
    )
    plt.close(fig)

print("Comparison charts saved successfully.")
