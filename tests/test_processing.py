import numpy as np
import polars as pl
import pytest

from glider_dashboard.processing import add_dive_column, concat_missions, mld_profile


def test_concat_missions_keeps_profile_numbers_unique(make_mission):
    a = make_mission("2024-01-01", n_profiles=4).lazy()
    b = make_mission("2024-02-01", n_profiles=3).drop("salinity").lazy()
    combined = concat_missions([a, b]).collect()
    assert combined["profile_num"].max() == 7
    assert combined["profile_num"].n_unique() == 7
    assert combined["salinity"].null_count() == b.collect().height


def test_add_dive_column():
    df = pl.DataFrame({"profile_num": [1.0, 2.0], "profile_direction": [1.0, -1.0]})
    assert add_dive_column(df)["dives"].to_list() == [1.0, 2.5]


@pytest.mark.parametrize("profile", [1, 2], ids=["descending", "ascending"])
def test_mld_profile_finds_mixed_layer(make_mission, profile):
    df = make_mission("2024-01-01", mixed_layer=20.0).filter(pl.col("profile_num") == profile)
    result = mld_profile(df, "temperature", thresh=0.3, ref_depth=5, verbose=False)
    assert result["mld"].item() == pytest.approx(20.0)


def test_mld_profile_without_reference_depth_data(make_mission):
    df = make_mission("2024-01-01").filter(
        (pl.col("profile_num") == 1) & (pl.col("depth") > 30)
    )
    result = mld_profile(df, "temperature", thresh=0.3, ref_depth=5, verbose=False)
    assert np.isnan(result["mld"].item())
