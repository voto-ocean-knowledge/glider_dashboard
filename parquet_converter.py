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
        df = xr.open_dataset(file, drop_variables="ad2cp_time").to_pandas().sort_index()
        df = pl.from_dataframe(df.astype(np.float32))
        df.write_parquet(file.replace("nc", "parquet"))
