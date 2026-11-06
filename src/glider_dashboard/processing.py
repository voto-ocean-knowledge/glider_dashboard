"""Pure data-processing helpers operating on polars frames."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def concat_missions(frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Concatenate mission frames, keeping ``profile_num`` unique.

    The profile numbers of every frame are offset by the last profile number
    of the previous frame, so they keep increasing monotonically across
    missions. Frames with differing columns are combined with
    ``how="diagonal_relaxed"`` (missing columns become null).
    """
    frames = list(frames)
    for index in range(1, len(frames)):
        offset = (
            frames[index - 1].select("profile_num").last().collect()["profile_num"].item()
        )
        frames[index] = frames[index].with_columns(pl.col("profile_num") + offset)
    return pl.concat(frames, how="diagonal_relaxed")


def add_dive_column(df: pl.LazyFrame | pl.DataFrame):
    """Add a ``dives`` column: ``profile_num`` for dives, +0.5 for climbs."""
    return df.with_columns(
        dives=pl.when(pl.col("profile_direction") == 1)
        .then(pl.col("profile_num"))
        .otherwise(pl.col("profile_num") + 0.5)
    )


def mld_profile(
    df: pl.DataFrame,
    variable: str,
    thresh: float,
    ref_depth: float,
    verbose: bool = True,
) -> pl.DataFrame:
    """Mixed layer depth of a single profile using a threshold criterion.

    Parameters
    ----------
    df : polars.DataFrame
        One profile with columns ``variable``, ``depth``, ``time`` and
        ``profile_num``. Depth is positive downwards.
    variable : str
        Variable for the threshold criterion (e.g. temperature or density).
    thresh : float
        Difference to the reference value that marks the base of the mixed
        layer.
    ref_depth : float
        Reference depth (positive, in m).
    verbose : bool
        Log why the MLD could not be determined.

    Returns
    -------
    polars.DataFrame
        One row with columns ``mld`` (positive, NaN if undetermined) and
        ``time`` (time of the first observation in the profile).
    """
    divenum = df["profile_num"].first()
    ptime = df["time"].first()
    df = df.with_columns(pl.col("depth").neg()).drop_nulls(subset=[variable, "depth"])

    message = None
    mld = np.nan
    if len(df) == 0:
        message = f"no observations found for specified variable in dive {divenum}"
    elif np.nanmin(np.abs(df["depth"] + ref_depth)) > 5:
        message = f"no observations within 5 m of ref_depth for dive {divenum}"
    else:
        # profile_direction is not available for GDAC data, infer it from depth
        direction = 1 if (df["depth"].first() > df["depth"].last()) else -1
        # arrays in order of increasing depth
        var_arr = df[variable][::direction]
        depth = df["depth"][::direction]
        i = np.nanargmin(np.abs(depth + ref_depth))
        dd = var_arr - var_arr[int(i)]
        # mask out all values that are shallower than ref_depth
        dd[depth > ref_depth] = np.nan
        if len(dd.filter(abs(dd) > thresh)) > 0:
            mld = depth[int(np.argmax(abs(dd) > thresh))]
        else:
            message = (
                "threshold criterion never true (all mixed or shallow profile) "
                f"for profile {divenum}"
            )
    if verbose and message:
        logger.info(message)
    return pl.DataFrame({"mld": [-mld], "time": [ptime]})
