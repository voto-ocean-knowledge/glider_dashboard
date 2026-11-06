"""Download glider data from ERDDAP and convert it to parquet.

For every selected mission one NetCDF file is downloaded (the delayed-mode
dataset if it exists, else the near-real-time one) and converted into two
parquet files read by the dashboard:

``<dataset_id>.parquet``
    Full resolution data (resampled to 1 s if sampled faster).
``<dataset_id>_small.parquet``
    Data averaged into 0.2 m depth bins and 3 h periods, used for statistics
    and zoomed-out views.

Existing files are reused. Near-real-time missions are downloaded again only
when ERDDAP has newer data than the local copy (the mission is still ongoing).

Long missions are requested from ERDDAP in time chunks: very large single
NetCDF responses have been seen to come back with some variables silently
zero-filled. Every downloaded file is checked against the ``actual_range``
attributes ERDDAP reports for its variables.
"""

from __future__ import annotations

import itertools
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import urllib3
import xarray

from . import datasets
from .config import Config

logger = logging.getLogger(__name__)

_TIMEOUT = urllib3.Timeout(connect=120, read=300)

#: ADCP variables merged into delayed-mode datasets, as {new name: ADCP file name}.
ADCP_VARIABLES = {
    "u": "velocity_N_DAC_reference_sb_corrected",
    "v": "velocity_E_DAC_reference_sb_corrected",
    "glider_speed_through_water": "speed_through_water",
    "shear_E_mean": "shear_E_mean",
    "shear_N_mean": "shear_N_mean",
}


class DownloadError(OSError):
    """HTTP request that did not return 200."""

    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


class CorruptDownloadError(ValueError):
    """Downloaded values contradict the ranges ERDDAP reports for them."""


def download_file(url: str, path: Path) -> None:
    """Stream ``url`` to ``path``.

    The data is written to a temporary ``.part`` file first, so an interrupted
    download never leaves a truncated file that would later be mistaken for a
    complete one.
    """
    tmp_path = path.with_name(path.name + ".part")
    http = urllib3.PoolManager(timeout=_TIMEOUT)
    with http.request("GET", url, preload_content=False) as resp:
        if resp.status != 200:
            raise DownloadError(resp.status, url)
        with open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(resp, out_file)
    tmp_path.replace(path)


def local_max_time(parquet_file: Path):
    """Last timestamp in a converted parquet file, ``None`` if it does not exist."""
    if not parquet_file.is_file():
        return None
    return pl.scan_parquet(parquet_file).select(pl.col("time").max()).collect().item()


def needs_refresh(dataset_id: str, server_max_time, parquet_file: Path) -> bool:
    """True for nrt datasets for which ERDDAP has newer data than the local copy.

    Delayed-mode datasets are final and never refreshed.
    """
    if not dataset_id.startswith("nrt"):
        return False
    local = local_max_time(parquet_file)
    if local is None or pd.isna(server_max_time):
        return local is None
    # converted data is resampled to whole seconds, allow for that
    return pd.Timestamp(local) < server_max_time - pd.Timedelta(minutes=1)


def tabledap_url(server: str, dataset_id: str, start=None, stop=None) -> str:
    """URL to download a tabledap dataset as NetCDF, optionally a time window.

    The window is ``start <= time < stop``.
    """
    from erddapy import ERDDAP  # erddapy contacts the network when imported

    e = ERDDAP(server=server, protocol="tabledap", response="nc")
    e.dataset_id = dataset_id
    constraints = {}
    if start is not None:
        constraints["time>="] = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%SZ")
    if stop is not None:
        constraints["time<"] = pd.Timestamp(stop).strftime("%Y-%m-%dT%H:%M:%SZ")
    return e.get_download_url(constraints=constraints or None)


def check_actual_ranges(ds: xarray.Dataset) -> list[str]:
    """Variables whose values fall outside the ``actual_range`` attribute.

    ERDDAP writes the true range of every variable into ``actual_range``, so
    values outside it (e.g. a variable that is all zeros) mean the download
    is corrupt.
    """
    bad = []
    for name, var in ds.data_vars.items():
        actual_range = var.attrs.get("actual_range")
        if actual_range is None or not np.issubdtype(var.dtype, np.number):
            continue
        values = var.values
        if values.size == 0 or np.all(np.isnan(values.astype(float))):
            continue
        lo, hi = float(actual_range[0]), float(actual_range[-1])
        tolerance = 1e-3 + 0.01 * abs(hi - lo)
        if np.nanmin(values) < lo - tolerance or np.nanmax(values) > hi + tolerance:
            bad.append(name)
    return bad


def _download_checked(url: str, path: Path) -> None:
    """Download a NetCDF file and verify its values, see :func:`check_actual_ranges`."""
    download_file(url, path)
    with xarray.open_dataset(path) as ds:
        bad = check_actual_ranges(ds)
    if bad:
        path.unlink()
        raise CorruptDownloadError(f"{path.name}: values outside actual_range for {bad}")


def download_tabledap(
    server: str, dataset_id: str, path: Path, start, stop, chunk_days: float | None
) -> None:
    """Download a whole tabledap dataset to ``path``, in time chunks if it is long.

    Chunks are downloaded separately and then concatenated along ``row``.
    """
    span = pd.Timestamp(stop) - pd.Timestamp(start)
    if not chunk_days or pd.isna(span) or span <= pd.Timedelta(days=chunk_days):
        _download_checked(tabledap_url(server, dataset_id), path)
        return

    edges = list(pd.date_range(start, stop, freq=pd.Timedelta(days=chunk_days)))
    edges[0] = pd.Timestamp(start).floor("s")
    edges.append(pd.Timestamp(stop).floor("s") + pd.Timedelta(seconds=1))
    chunk_files = []
    try:
        for i, (chunk_start, chunk_stop) in enumerate(itertools.pairwise(edges)):
            chunk_file = path.with_name(f"{path.stem}.chunk{i:03d}.nc")
            logger.info(
                "  %s chunk %d/%d (%s to %s)",
                dataset_id, i + 1, len(edges) - 1, chunk_start.date(), chunk_stop.date(),
            )
            try:
                _download_checked(
                    tabledap_url(server, dataset_id, chunk_start, chunk_stop), chunk_file
                )
            except DownloadError as err:
                if err.status == 404:  # ERDDAP: no data in this time window
                    continue
                raise
            chunk_files.append(chunk_file)

        chunks = [xarray.open_dataset(f) for f in chunk_files]
        try:
            combined = xarray.concat(
                chunks, dim="row", data_vars="minimal", coords="minimal", compat="override"
            )
            for var in combined.variables.values():
                # ranges of the first chunk do not describe the whole dataset
                var.attrs.pop("actual_range", None)
            tmp_path = path.with_name(path.name + ".part")
            combined.to_netcdf(tmp_path)
        finally:
            for chunk in chunks:
                chunk.close()
        tmp_path.replace(path)
    finally:
        for chunk_file in chunk_files:
            chunk_file.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# VOTO
# --------------------------------------------------------------------------- #


def download_voto_dataset(config: Config, dataset_id: str, start, stop, refresh: bool) -> None:
    """Download a VOTO dataset (and its ADCP file for delayed-mode data)."""
    nc_file = config.data_dir / f"{dataset_id}.nc"
    if nc_file.is_file() and not refresh:
        logger.info("%s already downloaded", nc_file.name)
    else:
        logger.info("Downloading %s", dataset_id)
        download_tabledap(
            config.voto.server, dataset_id, nc_file, start, stop, config.voto.chunk_days
        )

    if dataset_id.startswith("delayed_"):
        mission = dataset_id.removeprefix("delayed_")
        adcp_file = config.data_dir / f"{mission}_adcp_proc.nc"
        if not adcp_file.is_file():
            try:
                download_file(f"{config.voto.adcp_files_url}/{mission}_adcp_proc.nc", adcp_file)
            except OSError as err:
                logger.info("No ADCP data for %s (%s)", dataset_id, err)


def _merge_adcp(ds: xarray.Dataset, adcp_file: Path) -> xarray.Dataset:
    """Interpolate ADCP velocities onto the glider time/depth samples."""
    with xarray.open_dataset(adcp_file) as adcp:
        adcp = (
            adcp.set_index({"profile_index": "time"})
            .drop_duplicates(dim="profile_index")
            .dropna(dim="profile_index", subset=["profile_index"])
        )
        interpolated = adcp.interp(
            profile_index=ds["time"], depth=ds["depth"], method="linear"
        ).reset_coords("profile_index")
        for name, adcp_name in ADCP_VARIABLES.items():
            ds[name] = interpolated[adcp_name]
    return ds


def convert_voto_dataset(config: Config, dataset_id: str) -> None:
    """Convert a downloaded VOTO NetCDF file to the full and small parquet files."""
    nc_file = config.data_dir / f"{dataset_id}.nc"
    mission = dataset_id.removeprefix("delayed_")
    adcp_file = config.data_dir / f"{mission}_adcp_proc.nc"

    logger.info("Converting %s to parquet", dataset_id)
    with xarray.open_dataset(nc_file, drop_variables="ad2cp_time") as ds:
        if "time" not in ds["depth"].dims:
            # some datasets come from ERDDAP with a "row" instead of "time" dimension
            ds = ds.swap_dims({"row": "time"})
        ds = ds.drop_duplicates(dim="time").load()
    if adcp_file.is_file():
        ds = _merge_adcp(ds, adcp_file)

    df = ds.to_pandas().sort_index()
    if df.index.diff().mean() < np.timedelta64(600, "ms"):
        df = df.resample("1s").mean()
    pl.from_dataframe(df.astype(np.float32)).write_parquet(
        config.data_dir / f"{dataset_id}.parquet"
    )
    small_frame(df).write_parquet(config.data_dir / f"{dataset_id}_small.parquet")


def small_frame(df: pd.DataFrame) -> pl.DataFrame:
    """Average a full-resolution mission into 0.2 m depth x 3 h bins."""
    df = df.copy()
    maxdepth = df["depth"].quantile(q=0.99)
    df["depth"] = pd.cut(x=df["depth"], bins=np.arange(0, maxdepth, 0.2), include_lowest=False)
    ds = df.groupby([df["depth"], pd.Grouper(freq="3h")]).mean().to_xarray()
    ds["depth"] = [interval.mid for interval in ds.depth.values]
    # Keep the mean (fractional) profile number of each bin: integer profile
    # numbers per time bin would undercount the total number of profiles shown
    # in the statistics.
    profile_num = ds["profile_num"].mean(dim="depth").values
    xv, _ = np.meshgrid(profile_num, np.arange(len(ds.indexes["depth"])))
    ds["profile_num"] = (["depth", "time"], xv)
    return pl.from_dataframe(ds.to_dataframe().reset_index("depth").astype(np.float32))


def update_voto(config: Config) -> list[str]:
    """Download and convert all selected VOTO missions.

    Returns
    -------
    list of str
        Dataset IDs that failed.
    """
    metadata = datasets.load_voto_metadata(config)
    all_datasets = datasets.load_voto_datasets(config)
    missions = datasets.select_missions(metadata, config.selection)
    missions = missions[missions.index.isin(all_datasets.index)]
    logger.info("%d missions selected: %s", len(missions), ", ".join(missions.index))

    failed = []
    for nrt_id in missions.index:
        dataset_id = datasets.preferred_dataset_id(nrt_id, all_datasets.index)
        start, stop = all_datasets.loc[dataset_id, [datasets.MIN_TIME, datasets.MAX_TIME]]
        parquet_file = config.data_dir / f"{dataset_id}.parquet"
        try:
            refresh = needs_refresh(dataset_id, stop, parquet_file)
            download_voto_dataset(config, dataset_id, start, stop, refresh)
            if refresh or not parquet_file.is_file():
                convert_voto_dataset(config, dataset_id)
        except Exception:
            logger.exception("Failed to prepare %s", dataset_id)
            failed.append(dataset_id)
    return failed


# --------------------------------------------------------------------------- #
# IOOS GDAC
# --------------------------------------------------------------------------- #


def update_gdac(config: Config) -> list[str]:
    """Download and convert the GDAC datasets selected in ``config.gdac``.

    Returns
    -------
    list of str
        Dataset IDs that failed.
    """
    gdac_datasets = datasets.load_gdac_datasets(config)
    failed = []
    for dataset_id in gdac_datasets.index:
        nc_file = config.data_dir / f"{dataset_id}.nc"
        try:
            if not nc_file.is_file():
                from erddapy import ERDDAP

                info_url = ERDDAP(server=config.gdac.server).get_info_url(
                    dataset_id=dataset_id, response="csv"
                )
                info = pl.read_csv(info_url)
                n_variables = info.filter(pl.col("Row Type") == "variable").height
                if n_variables > config.gdac.max_variables:
                    logger.warning("Skipping %s: %d variables", dataset_id, n_variables)
                    continue
                logger.info("Downloading %s", dataset_id)
                download_file(tabledap_url(config.gdac.server, dataset_id), nc_file)

            full = config.data_dir / f"{dataset_id}.parquet"
            small = config.data_dir / f"{dataset_id}_small.parquet"
            if not (full.is_file() and small.is_file()):
                logger.info("Converting %s to parquet", dataset_id)
                with xarray.open_dataset(nc_file) as ds:
                    df = ds.to_pandas().sort_index()
                df.to_parquet(full)
                df.to_parquet(small)
        except Exception:
            logger.exception("Failed to prepare %s", dataset_id)
            failed.append(dataset_id)
    return failed


def update_data(config: Config) -> list[str]:
    """Bring the local data directory up to date with the configuration.

    Returns
    -------
    list of str
        Dataset IDs that could not be downloaded or converted.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    failed = update_voto(config)
    if config.gdac.enabled:
        failed += update_gdac(config)
    return failed
