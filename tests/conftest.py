"""Shared fixtures: a small synthetic catalog that needs no network access."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from glider_dashboard.catalog import DataCatalog
from glider_dashboard.datasets import MAX_TIME, MIN_TIME, TIME_END, TIME_START, VOTO_INSTITUTION


def synthetic_mission(start: str, n_profiles: int = 6, mixed_layer: float = 20.0) -> pl.DataFrame:
    """Up/down profiles to 60 m with a warm mixed layer above ``mixed_layer``."""
    depth_down = np.arange(0.0, 60.0, 1.0)
    rows = []
    t0 = pd.Timestamp(start)
    for profile in range(1, n_profiles + 1):
        descending = profile % 2 == 1
        depth = depth_down if descending else depth_down[::-1]
        temperature = np.where(depth < mixed_layer, 15.0, 5.0)
        times = t0 + pd.to_timedelta(profile * 3600 + np.arange(len(depth)) * 10, unit="s")
        rows.append(
            pd.DataFrame(
                {
                    "time": times,
                    "depth": depth,
                    "pressure": depth,
                    "temperature": temperature,
                    "salinity": 7.0 + depth / 10,
                    "profile_num": float(profile),
                    "profile_direction": 1.0 if descending else -1.0,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    return pl.from_pandas(df).with_columns(pl.exclude("time").cast(pl.Float32))


@pytest.fixture
def make_mission():
    return synthetic_mission


MISSIONS = {
    # nrt id: (start, end, basin, has delayed version)
    "nrt_SEA001_M1": ("2024-01-01", "2024-01-20", "Bornholm Basin", True),
    "nrt_SEA002_M2": ("2024-03-01", "2024-03-20", "Bornholm Basin", False),
    "nrt_SEA003_M3": ("2024-05-01", "2024-05-20", "Skagerrak, Kattegat", False),
}


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "project": "SAMBA",
            "basin": [m[2] for m in MISSIONS.values()],
            TIME_START: pd.to_datetime([m[0] for m in MISSIONS.values()]),
            TIME_END: pd.to_datetime([m[1] for m in MISSIONS.values()]),
            "deployment_id": [1, 2, 3],
        },
        index=pd.Index(list(MISSIONS), name="datasetID"),
    )


@pytest.fixture
def voto_datasets():
    rows = {}
    for nrt_id, (start, end, _, has_delayed) in MISSIONS.items():
        row = {
            MIN_TIME: pd.Timestamp(start),
            MAX_TIME: pd.Timestamp(end),
            "institution": VOTO_INSTITUTION,
            "metadata": f"https://example.org/info/{nrt_id}/index",
        }
        rows[nrt_id] = row
        if has_delayed:
            # delayed-mode data usually extends past the end of the nrt data
            rows[nrt_id.replace("nrt_", "delayed_")] = row | {MAX_TIME: pd.Timestamp(end) + pd.Timedelta(days=5)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "datasetID"
    return df


@pytest.fixture
def data_dir(tmp_path, voto_datasets):
    """Parquet files for the missions, as written by ``glider-dashboard download``."""
    for nrt_id, (start, *_rest) in MISSIONS.items():
        dsid = nrt_id.replace("nrt_", "delayed_") if MISSIONS[nrt_id][3] else nrt_id
        frame = synthetic_mission(start)
        frame.write_parquet(tmp_path / f"{dsid}.parquet")
        frame.write_parquet(tmp_path / f"{dsid}_small.parquet")
    return tmp_path


@pytest.fixture
def catalog(data_dir, metadata, voto_datasets):
    return DataCatalog(data_dir, metadata, metadata, voto_datasets)
