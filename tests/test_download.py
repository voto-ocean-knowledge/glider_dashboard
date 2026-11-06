from urllib.parse import unquote

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from glider_dashboard import download


def tabledap_file(path, start, stop, temperature=None, actual_range=(3.0, 20.0)):
    """Small NetCDF file shaped like an ERDDAP tabledap response."""
    time = pd.date_range(start, stop, freq="1h", inclusive="left")
    if temperature is None:
        temperature = np.linspace(*actual_range, len(time))
    ds = xr.Dataset(
        {"temperature": ("row", np.asarray(temperature, dtype="float32"))},
        coords={"time": ("row", time)},
    )
    ds["temperature"].attrs["actual_range"] = np.array(actual_range, dtype="float32")
    ds.to_netcdf(path)


def test_check_actual_ranges_detects_zero_filled_variable(tmp_path):
    tabledap_file(tmp_path / "good.nc", "2026-01-01", "2026-01-02")
    tabledap_file(tmp_path / "bad.nc", "2026-01-01", "2026-01-02", temperature=np.zeros(24))
    with xr.open_dataset(tmp_path / "good.nc") as ds:
        assert download.check_actual_ranges(ds) == []
    with xr.open_dataset(tmp_path / "bad.nc") as ds:
        assert download.check_actual_ranges(ds) == ["temperature"]


def test_needs_refresh_only_when_server_has_newer_nrt_data(tmp_path):
    parquet = tmp_path / "nrt_X.parquet"
    end = pd.Timestamp("2026-10-14 07:09:00")
    assert download.needs_refresh("nrt_X", end, parquet)  # nothing local yet

    pl.DataFrame({"time": [end - pd.Timedelta(hours=1), end]}).write_parquet(parquet)
    assert not download.needs_refresh("nrt_X", end, parquet)
    assert download.needs_refresh("nrt_X", end + pd.Timedelta(hours=2), parquet)
    assert not download.needs_refresh("delayed_X", end + pd.Timedelta(hours=2), parquet)


@pytest.fixture
def fake_erddap(monkeypatch):
    """Serve time windows of a 20-day dataset; the 3rd window has no data."""
    requests = []

    def fake_download_file(url, path):
        requests.append(url)
        query = unquote(url)
        start = pd.Timestamp(float(query.split("time>=")[1].split("&")[0]), unit="s")
        stop = pd.Timestamp(float(query.split("time<")[1].split("&")[0]), unit="s")
        if len(requests) == 3:
            raise download.DownloadError(404, url)
        tabledap_file(path, start, stop)

    monkeypatch.setattr(download, "download_file", fake_download_file)
    return requests


def test_long_datasets_are_downloaded_in_chunks(tmp_path, fake_erddap):
    target = tmp_path / "delayed_X.nc"
    start, stop = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-21")
    download.download_tabledap("https://example.org/erddap", "delayed_X", target, start, stop, 7)

    assert len(fake_erddap) == 3
    with xr.open_dataset(target) as ds:
        # 7 + 7 days of hourly data, the empty third window is skipped
        assert ds.sizes["row"] == 14 * 24
        assert ds["time"].to_index().is_monotonic_increasing
    assert sorted(p.name for p in tmp_path.iterdir()) == ["delayed_X.nc"]  # chunks cleaned up
