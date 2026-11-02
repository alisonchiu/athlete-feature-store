
from pathlib import Path
import sys

from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from prepare_data import prepare_data, V1, V2
from feature_repo.features import (
    athlete,
    athlete_v1,
    athlete_v2
)

DATA = ROOT / "data"
REPO = ROOT / "feature_repo"


def register_features():
    # Load and prepare dataset
    df = prepare_data(limit=30000)

    print(f"Prepared {len(df)} athlete records")

    # Create Parquet feature datasets
    for name, columns in [
        ("athlete_v1", V1),
        ("athlete_v2", V2)
    ]:
        output_path = DATA / f"{name}.parquet"

        selected = [
            "athlete_id",
            "event_timestamp",
            "created_timestamp"
        ] + columns

        df[selected].to_parquet(
            output_path,
            index=False
        )

        print(f"Saved: {output_path.name}")

    # Register features with Feast
    store = FeatureStore(repo_path=str(REPO))

    store.apply([
        athlete,
        athlete_v1,
        athlete_v2
    ])

    print("\nFeature views registered successfully!")

    for view in store.list_feature_views():
        print(f"- {view.name}")


if __name__ == "__main__":
    register_features()
