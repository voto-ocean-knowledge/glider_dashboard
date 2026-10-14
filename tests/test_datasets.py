import pandas as pd
import pytest

from glider_dashboard import datasets
from glider_dashboard.config import SelectionConfig


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bornholm Basin, Eastern Gotland Basin", "Bornholm Basin"),
        ("Northern Baltic Proper", "Eastern Gotland"),
        ("Eastern Gotland Basin, Northern Baltic Proper", "Eastern Gotland"),
        ("Kattegat, Skagerrak", "Skagerrak, Kattegat"),
        ("\\u00c5land Sea", "Åland Sea"),
        ("Arkona Basin", "Arkona Basin"),
        (float("nan"), "undefined"),
    ],
)
def test_simplify_basin(raw, expected):
    assert datasets.simplify_basin(raw) == expected


def test_mission_and_preferred_ids():
    assert datasets.mission_id("SEA067_M15") == "nrt_SEA067_M15"
    assert datasets.mission_id("delayed_SEA067_M15") == "nrt_SEA067_M15"
    available = ["nrt_A_M1", "delayed_A_M1", "nrt_B_M2"]
    assert datasets.preferred_dataset_id("nrt_A_M1", available) == "delayed_A_M1"
    assert datasets.preferred_dataset_id("nrt_B_M2", available) == "nrt_B_M2"


def test_select_missions_filters(metadata):
    selection = SelectionConfig(basins=["Bornholm Basin"], latest=None, exclude_dataset_ids=[])
    assert list(datasets.select_missions(metadata, selection).index) == [
        "nrt_SEA001_M1",
        "nrt_SEA002_M2",
    ]

    selection.latest = 1
    assert list(datasets.select_missions(metadata, selection).index) == ["nrt_SEA002_M2"]

    selection = SelectionConfig(basins=[], years=[2023], latest=None)
    assert datasets.select_missions(metadata, selection).empty


def test_select_missions_explicit_ids_override_filters(metadata):
    selection = SelectionConfig(
        dataset_ids=["SEA003_M3", "delayed_SEA001_M1", "nrt_SEA999_M9"],
        basins=["Bornholm Basin"],
        exclude_dataset_ids=["nrt_SEA001_M1"],
    )
    assert list(datasets.select_missions(metadata, selection).index) == ["nrt_SEA003_M3"]


def test_drop_overlaps_keeps_longest_mission_per_start_date():
    meta = pd.DataFrame(
        {
            datasets.TIME_START: pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00", "2024-02-01 00:00"]),
            datasets.TIME_END: pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10"]),
        },
        index=["short", "long", "other"],
    )
    assert sorted(datasets.drop_overlaps_fast(meta).index) == ["long", "other"]


def test_erddap_table_falls_back_to_cache(tmp_path, monkeypatch):
    import erddapy

    table = pd.DataFrame({"x": [1]}, index=pd.Index(["ds"], name="datasetID"))
    monkeypatch.setattr(erddapy.ERDDAP, "to_pandas", lambda self, **kw: table)
    backup = tmp_path / "table.pkl"
    datasets.load_erddap_table("https://example.org/erddap", "allDatasets", backup)
    assert backup.is_file()

    def offline(self, **kw):
        raise OSError("no network")

    monkeypatch.setattr(erddapy.ERDDAP, "to_pandas", offline)
    cached = datasets.load_erddap_table("https://example.org/erddap", "allDatasets", backup)
    pd.testing.assert_frame_equal(cached, table)

    with pytest.raises(RuntimeError, match="no cached copy"):
        datasets.load_erddap_table("https://example.org/erddap", "x", tmp_path / "missing.pkl")
