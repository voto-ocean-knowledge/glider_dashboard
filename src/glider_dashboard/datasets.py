"""ERDDAP dataset tables and mission selection.

VOTO publishes one *mission* in up to two versions: a near-real-time dataset
(``nrt_<glider>_<mission>``) and, once processed, a higher resolution
delayed-mode dataset (``delayed_<glider>_<mission>``). Missions are identified
by their nrt ID throughout the dashboard (that is how the metadata table is
indexed); the data actually loaded is the delayed version when it exists, see
:func:`preferred_dataset_id`.

Nothing in this module touches the network at import time.
"""

from __future__ import annotations

import logging
import pickle
import pprint
from pathlib import Path

import pandas as pd

from .config import Config, GdacConfig, SelectionConfig

logger = logging.getLogger(__name__)

TIME_START = "time_coverage_start (UTC)"
TIME_END = "time_coverage_end (UTC)"
MIN_TIME = "minTime (UTC)"
MAX_TIME = "maxTime (UTC)"

VOTO_INSTITUTION = "Voice of the Ocean Foundation"


# --------------------------------------------------------------------------- #
# Loading tables from ERDDAP
# --------------------------------------------------------------------------- #


def load_erddap_table(server: str, dataset_id: str, backup_file: Path) -> pd.DataFrame:
    """Download a tabledap dataset as a DataFrame indexed by ``datasetID``.

    A pickled copy is written to ``backup_file`` on success and read back when
    the server cannot be reached, so the dashboard keeps working offline.
    """
    from erddapy import ERDDAP  # erddapy contacts the network when imported

    e = ERDDAP(server=server, protocol="tabledap", response="csv")
    e.dataset_id = dataset_id
    try:
        df = e.to_pandas(index_col="datasetID", date_format="%f")
    except Exception as err:
        if not backup_file.is_file():
            raise RuntimeError(
                f"Could not download '{dataset_id}' from {server} and no "
                f"cached copy exists at {backup_file}"
            ) from err
        logger.warning("Could not update %s from ERDDAP (%s), using cache", dataset_id, err)
        with open(backup_file, "rb") as f:
            return pickle.load(f)
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_file, "wb") as f:
        pickle.dump(df, f)
    return df


def _parse_time_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Parse UTC timestamp columns into timezone-naive datetimes (still UTC)."""
    for column in columns:
        df[column] = pd.to_datetime(df[column], utc=True).dt.tz_convert(None)
    return df


def load_voto_datasets(config: Config) -> pd.DataFrame:
    """Load the VOTO ``allDatasets`` table (one row per ERDDAP dataset)."""
    df = load_erddap_table(
        config.voto.server, "allDatasets", config.data_dir / "allDatasetsVOTO.pkl"
    )
    return _parse_time_columns(df, [MIN_TIME, MAX_TIME])


def load_voto_metadata(config: Config) -> pd.DataFrame:
    """Load the VOTO mission metadata table (one row per mission, nrt IDs)."""
    df = load_erddap_table(
        config.voto.server, "meta_metadata_table", config.data_dir / "metadata.pkl"
    )
    return prepare_metadata(df)


def load_gdac_datasets(config: Config) -> pd.DataFrame:
    """Load and filter the IOOS GDAC ``allDatasets`` table."""
    df = load_erddap_table(
        config.gdac.server, "allDatasets", config.data_dir / "allDatasetsGDAC.pkl"
    )
    df = df.drop(index="allDatasets", errors="ignore")
    df = _parse_time_columns(df, [MIN_TIME, MAX_TIME])
    return filter_gdac_datasets(df, config.gdac)


def filter_gdac_datasets(df: pd.DataFrame, gdac: GdacConfig) -> pd.DataFrame:
    """Restrict the GDAC table to glider datasets of the configured period."""
    if "cdm_data_type" in df:
        df = df[df["cdm_data_type"] == "TrajectoryProfile"]
    if gdac.year is not None:
        df = df[df[MIN_TIME].dt.year == gdac.year]
    if gdac.month is not None:
        df = df[df[MIN_TIME].dt.month == gdac.month]
    df = df.iloc[: gdac.max_datasets]
    # Some institutions' files currently fail to download from the GDAC.
    return df[~df["institution"].isin(gdac.exclude_institutions)]


# --------------------------------------------------------------------------- #
# Metadata cleaning
# --------------------------------------------------------------------------- #


def simplify_basin(basin) -> str:
    """Map the free-text ``basin`` attribute onto a SAMBA observatory name."""
    if not isinstance(basin, str):
        return "undefined"  # empty basin arrives as NaN
    first = basin.split(",")[0]
    if first == "Bornholm Basin":
        return "Bornholm Basin"
    if first == "Eastern Gotland Basin" or basin in (
        "Northern Baltic Proper, Eastern Gotland Basin",
        "Northern Baltic Proper",
    ):
        return "Eastern Gotland"
    if first in ("Skagerrak", "Kattegat"):
        return "Skagerrak, Kattegat"
    if first == "Western Gotland Basin":
        return "Western Gotland"
    if first in ("Åland Sea", "\\u00c5land Sea", "\\u00c3\\u0085land Sea"):
        return "Åland Sea"
    return basin


def prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw VOTO metadata table for display and filtering."""
    metadata = metadata.copy()
    for column in ("optics_serial", "irradiance_serial", "altimeter_serial", "glider_serial"):
        if column in metadata:
            metadata[column] = metadata[column].apply(pprint.pformat)
    metadata["basin"] = metadata["basin"].apply(simplify_basin)
    return _parse_time_columns(metadata, [TIME_START, TIME_END])


# --------------------------------------------------------------------------- #
# Mission selection
# --------------------------------------------------------------------------- #


def mission_id(dataset_id: str) -> str:
    """Normalise a dataset or mission name to the nrt mission ID.

    ``SEA067_M15``, ``nrt_SEA067_M15`` and ``delayed_SEA067_M15`` all map to
    ``nrt_SEA067_M15``.
    """
    name = dataset_id.removeprefix("delayed_").removeprefix("nrt_")
    return "nrt_" + name


def preferred_dataset_id(nrt_id: str, available_ids) -> str:
    """Return the delayed-mode ID for a mission if it exists, else the nrt ID."""
    delayed_id = "delayed_" + nrt_id.removeprefix("nrt_")
    return delayed_id if delayed_id in available_ids else nrt_id


def select_missions(metadata: pd.DataFrame, selection: SelectionConfig) -> pd.DataFrame:
    """Filter the metadata table down to the missions in ``selection``.

    Returns the selected rows sorted by mission start time.
    """
    if selection.dataset_ids:
        wanted = [mission_id(dsid) for dsid in selection.dataset_ids]
        missing = sorted(set(wanted) - set(metadata.index))
        if missing:
            logger.warning("Requested missions not found in metadata: %s", missing)
        selected = metadata.loc[metadata.index.isin(wanted)]
    else:
        mask = pd.Series(True, index=metadata.index)
        if selection.project:
            mask &= metadata["project"] == selection.project
        if selection.basins:
            mask &= metadata["basin"].isin(selection.basins)
        if selection.years:
            mask &= metadata[TIME_START].dt.year.isin(selection.years)
        selected = metadata[mask]

    selected = selected.drop(selection.exclude_dataset_ids, errors="ignore")
    selected = selected.sort_values(TIME_START)
    if selection.latest:
        selected = selected.iloc[-selection.latest :]
    return selected


def drop_overlaps_fast(metadata: pd.DataFrame) -> pd.DataFrame:
    """Keep one mission per start date, preferring the longest one.

    Used to avoid drawing several gliders on top of each other when a whole
    observatory is shown.
    """
    duration = metadata[TIME_END] - metadata[TIME_START]
    startdate = metadata[TIME_START].dt.date
    order = pd.DataFrame({"startdate": startdate, "duration": duration}).sort_values(
        ["startdate", "duration"], ascending=[True, False]
    )
    remaining = order[["startdate"]].drop_duplicates().index
    return metadata.loc[remaining]
