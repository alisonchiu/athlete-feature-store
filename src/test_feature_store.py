
from pathlib import Path
import sys

from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_data import prepare_data, V1, V2

# Connect to the Feast feature store
store = FeatureStore(
    repo_path=str(ROOT / "feature_repo")
)

# Load the same prepared dataset
df = prepare_data(limit=30000)

# Select 5 athletes for testing
entity_df = df[
    ["athlete_id", "event_timestamp"]
].head(5)

print("Testing historical feature retrieval...")

for version, columns in [
    ("v1", V1),
    ("v2", V2)
]:
    feature_view = f"athlete_{version}"

    features = [
        f"{feature_view}:{col}"
        for col in columns
    ]

    retrieved = store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    print(f"\nFeature version: {version}")
    print("Shape:", retrieved.shape)
    print(retrieved.head().to_string())

print("\nFeature retrieval test completed!")
