
from pathlib import Path
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Entity used to join feature records
athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"]
)

# Version 1: Baseline features
V1 = [
    "age",
    "height",
    "weight",
    "gender_male",
    "backsq",
    "candj",
    "snatch"
]

# Version 2: Engineered features
V2 = V1 + [
    "bmi",
    "backsq_to_weight",
    "candj_to_weight",
    "snatch_to_weight",
    "strength_total"
]


def create_feature_view(name, columns):
    source = FileSource(
        path=str(
            (DATA / f"{name}.parquet").resolve()
        ),
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp"
    )

    return FeatureView(
        name=name,
        entities=[athlete],
        ttl=timedelta(days=3650),
        schema=[
            Field(name=col, dtype=Float64)
            for col in columns
        ],
        source=source,
        online=True
    )


athlete_v1 = create_feature_view("athlete_v1", V1)
athlete_v2 = create_feature_view("athlete_v2", V2)
