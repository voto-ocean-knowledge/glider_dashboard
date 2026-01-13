import glob
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

nc_datasets = glob.glob("../voto_erddap_data_cache/*.nc")
print("now converting all files to parquet, please wait...")
for file in nc_datasets:
    if Path(file.replace("nc", "parquet")).is_file():
        print(f"{file.replace('nc', 'parquet')} already exists, skip")
        continue
    else:
        # Create large (often full resolution) output
        print(f"now creating {file.replace('nc', 'parquet')}")
        df = xr.open_dataset(file, drop_variables="ad2cp_time").to_pandas().sort_index()
        if df.index.diff().mean() < np.timedelta64(600, "ms"):
            df = df.resample("1s").mean()
        df = pl.from_dataframe(df.astype(np.float32))
        df.write_parquet(file.replace("nc", "parquet"))
        # Create subsampled small output, but only if
        # the current file is a delayed mode file
        # (by checking if file str contains "delayed")
        if "delayed" in file:
            df = df.filter(pl.col("profile_num") % 10 == 0)
        df.write_parquet(file.replace(".nc", "_small.parquet"))

parquet_datasets = glob.glob("../voto_erddap_data_cache/*.parquet")
for file in parquet_datasets:
    if "small" in file:
        continue
    if "SEA" in file:
        continue
    if Path(file.replace(".parquet", "_small.parquet")).is_file():
        print(f"{file} already has a small version, skip")
        continue
    else:
        print(f"creating small version of {file}")
        df = pl.scan_parquet(file)
        if "nrt" not in file:
            df.sink_parquet(file.replace(".parquet", "_small.parquet"))
