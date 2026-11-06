"""The set of locally available glider data the dashboard works with.

A :class:`DataCatalog` bundles the ERDDAP tables with lazily scanned parquet
files from the data directory. It is built explicitly (see
:func:`get_catalog`), never at import time, so importing this package — for
example to build the documentation — does not download or read any data.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import pandas as pd
import polars as pl
import polars.selectors as cs

from . import datasets
from .config import Config, load_config

logger = logging.getLogger(__name__)


class DataCatalog:
    """Metadata tables and lazily loaded mission data.

    Missions are keyed by their nrt ID (GDAC datasets by their own ID). Only
    missions whose parquet files exist in ``data_dir`` are included; run
    ``glider-dashboard download`` to fetch them.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the ``<dataset_id>[_small].parquet`` files.
    all_metadata : pandas.DataFrame
        Complete VOTO metadata table, shown on the metadata page.
    metadata : pandas.DataFrame
        Metadata of the selected missions.
    voto_datasets : pandas.DataFrame
        VOTO ``allDatasets`` table.
    gdac_datasets : pandas.DataFrame, optional
        Filtered GDAC ``allDatasets`` table, if GDAC data is enabled.

    Attributes
    ----------
    mission_ids : list of str
        Missions with local data, in metadata order.
    missions : pandas.DataFrame
        ``allDatasets`` rows of :attr:`mission_ids`, with institution and the
        time coverage of the data actually shown (the delayed-mode dataset
        often covers a longer period than the nrt one).
    metadata : pandas.DataFrame
        VOTO metadata of the missions with local data.
    variables : list of str
        Sorted union of all variables found in the data.
    """

    def __init__(
        self,
        data_dir,
        all_metadata: pd.DataFrame,
        metadata: pd.DataFrame,
        voto_datasets: pd.DataFrame,
        gdac_datasets: pd.DataFrame | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.all_metadata = all_metadata
        self.voto_datasets = voto_datasets
        self.gdac_datasets = gdac_datasets
        if gdac_datasets is None:
            self.all_datasets = voto_datasets
        else:
            self.all_datasets = pd.concat([voto_datasets, gdac_datasets])

        self._frames: dict[str, pl.LazyFrame] = {}
        voto_ids = [
            nrt_id
            for nrt_id in metadata.index
            if nrt_id in voto_datasets.index and self._scan(self.data_id(nrt_id))
        ]
        gdac_ids = [
            dsid
            for dsid in (gdac_datasets.index if gdac_datasets is not None else [])
            if self._scan(dsid, gdac=True)
        ]
        self.metadata = metadata.loc[voto_ids]
        self.mission_ids = voto_ids + gdac_ids
        self.missions = self.all_datasets.loc[self.mission_ids].copy()
        time_columns = [datasets.MIN_TIME, datasets.MAX_TIME]
        shown = self.all_datasets.loc[[self.data_id(m) for m in self.mission_ids], time_columns]
        self.missions[time_columns] = shown.to_numpy()

        if self._frames:
            schema = pl.concat(self._frames.values(), how="diagonal_relaxed").collect_schema()
            self.variables = sorted(schema.names())
        else:
            self.variables = []
            logger.warning(
                "No mission data found in %s. Run `glider-dashboard download` first.",
                self.data_dir,
            )

    @classmethod
    def from_config(cls, config: Config) -> DataCatalog:
        """Load the ERDDAP tables and scan the data directory for ``config``."""
        all_metadata = datasets.load_voto_metadata(config)
        metadata = datasets.select_missions(all_metadata, config.selection)
        voto_datasets = datasets.load_voto_datasets(config)
        gdac_datasets = datasets.load_gdac_datasets(config) if config.gdac.enabled else None
        return cls(config.data_dir, all_metadata, metadata, voto_datasets, gdac_datasets)

    @property
    def has_gdac(self) -> bool:
        """Whether GDAC datasets are part of this catalog."""
        return self.gdac_datasets is not None

    def data_id(self, mission_id: str) -> str:
        """Dataset ID whose data is shown for a mission (delayed if available)."""
        return datasets.preferred_dataset_id(mission_id, self.voto_datasets.index)

    def frame(self, mission_id: str, small: bool = False) -> pl.LazyFrame:
        """Lazy frame with the data of a mission.

        Parameters
        ----------
        mission_id : str
            nrt mission ID or GDAC dataset ID.
        small : bool
            Return the depth/time binned version instead of full resolution.
        """
        dsid = self.data_id(mission_id)
        return self._frames[dsid + "_small" if small else dsid]

    def _scan(self, dsid: str, gdac: bool = False) -> bool:
        """Register the full and small parquet files of ``dsid`` if present."""
        files = {
            dsid: self.data_dir / f"{dsid}.parquet",
            dsid + "_small": self.data_dir / f"{dsid}_small.parquet",
        }
        missing = [str(f) for f in files.values() if not f.is_file()]
        if missing:
            logger.warning("Skipping %s, missing %s", dsid, ", ".join(missing))
            return False
        for key, path in files.items():
            frame = pl.scan_parquet(path)
            self._frames[key] = _harmonise_gdac(frame) if gdac else frame
        return True


def _harmonise_gdac(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Make a GDAC frame look like a VOTO one (numeric columns, naive ns time)."""
    if "profile_num" in frame.collect_schema().names():
        # a VOTO dataset on the GDAC has both profile_num and profile_id,
        # which would give duplicate columns after the rename below
        frame = frame.drop("profile_num")
    return (
        frame.drop(cs.string())
        .with_columns(pl.col("time").dt.cast_time_unit("ns").dt.replace_time_zone(None))
        .rename({"profile_id": "profile_num"})
    )


@functools.cache
def get_catalog(config_path: str | None = None) -> DataCatalog:
    """Build the catalog for a config file once per process and reuse it.

    ``panel serve`` executes the app script for every browser session; caching
    here keeps the ERDDAP tables and parquet scans shared between sessions.
    """
    return DataCatalog.from_config(load_config(config_path))
