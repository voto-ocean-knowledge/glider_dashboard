import glob

import numpy as np
import polars as pl
import xarray as xr

nc_datasets = glob.glob("../voto_erddap_data_cache/*.nc")
for file in nc_datasets:
    df = xr.open_dataset(file).to_pandas()
    df = pl.from_dataframe(df.astype(np.float32))
    df.write_parquet(file.replace("nc", "parquet"))
