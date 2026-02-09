import os.path
import pathlib
import urllib.request
from urllib.request import urlretrieve

import numpy as np
import polars as pl
import xarray
from erddapy import ERDDAP

import utils

# all metadata exists for the metadata visualisation
all_metadata = utils.load_metadata_VOTO()

###### filter metadata to prepare download ##############
# ToDO: Rename all_datasets to filtered_datasets
metadata = utils.filter_metadata()
metadata = metadata.drop(
    ["nrt_SEA067_M15", "nrt_SEA079_M14", "nrt_SEA061_M63"], errors="ignore"
)  #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
metadata = metadata.sort_values(by="time_coverage_start (UTC)")

allDatasetsVOTO = utils.load_allDatasets_VOTO()
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, allDatasetsVOTO)  # hacky

###### download actual data ##############################
cache_dir = pathlib.Path("../voto_erddap_data_cache")
# dsids = ['../voto_erddap_data_cache/'+element+'.nc' for element in metadata.index]
# import pdb
# pdb.set_trace()
print(all_dataset_ids)
for dataset_id in all_dataset_ids:
    url = f"https://erddap.observations.voiceoftheocean.org/erddap/files/{dataset_id}/mission_timeseries.nc"
    file_Path = f"../voto_erddap_data_cache/{dataset_id}.nc"
    if os.path.isfile(file_Path):
        print(f"{file_Path} already exists, skip")
    else:
        urllib.request.urlretrieve(url, file_Path)
    if dataset_id[0:7] == "delayed":
        dsid = dataset_id.replace("delayed_", "")
        url = f"https://erddap.observations.voiceoftheocean.org/erddap/files/gliderad2cp_files/{dsid}_adcp_proc.nc"
        file_Path_adcp = f"../voto_erddap_data_cache/{dsid}_adcp_proc.nc"
        if os.path.isfile(file_Path_adcp):
            print(f"{file_Path_adcp} already exists, skip")
        else:
            try:
                urllib.request.urlretrieve(url, file_Path_adcp)
            except:
                print(f"NO ADCP data for {dataset_id} found on voto erddap")

for dataset_id in all_dataset_ids:
    # if not (dataset_id[0:7] == "delayed"):
    #    continue
    if os.path.isfile(f"../voto_erddap_data_cache/{dataset_id}.parquet"):
        print(
            f"combined {dataset_id} data/adcp file already exists, skip"
        )  # not necessarily :/
        continue
    print(f"combining {dataset_id} variables with adcp file")
    file_Path = f"../voto_erddap_data_cache/{dataset_id}.nc"
    dsid = dataset_id.replace("delayed_", "")
    file_Path_adcp = f"../voto_erddap_data_cache/{dsid}_adcp_proc.nc"
    ds = (
        xarray.open_mfdataset(file_Path, drop_variables="ad2cp_time")
        .drop_duplicates(dim="time")
        .load()
    )

    try:
        ds2 = (
            xarray.open_mfdataset(file_Path_adcp)
            .set_index({"profile_index": "time"})
            .drop_duplicates(dim="profile_index")
            .dropna(dim="profile_index", subset=["profile_index"])
        )

        currentdirections = ds2.interp(
            profile_index=ds["time"], depth=ds["depth"], method="linear"
        ).reset_coords("profile_index")  # [
        # [
        #    "velocity_N_DAC_reference_sb_corrected",
        #    "velocity_E_DAC_reference_sb_corrected",
        # ]
        # ]
        ds[["u", "v", "glider_speed_through_water", "shear_E_mean", "shear_N_mean"]] = (
            currentdirections[
                [
                    "velocity_N_DAC_reference_sb_corrected",
                    "velocity_E_DAC_reference_sb_corrected",
                    "speed_through_water",
                    "shear_E_mean",
                    "shear_N_mean",
                ]
            ]
        )
    except:
        print(f"no adcp data for {dsid} found, skip combining")
        # continue
    # ds2.sortby('depth','profile_index').sel(profile_index=np.datetime64('2024-01-10'), method='nearest')#

    df = ds.to_pandas().sort_index()
    if df.index.diff().mean() < np.timedelta64(600, "ms"):
        df = df.resample("1s").mean()
    df = pl.from_dataframe(df.astype(np.float32))
    df.write_parquet(f"../voto_erddap_data_cache/{dataset_id}.parquet")
    df = df.filter(pl.col("profile_num") % 10 == 0)
    df.write_parquet(
        f"../voto_erddap_data_cache/{dataset_id}_small.parquet"
    )  # "file.replace(".nc", "_small.parquet").replace("_combined", ""))

    # file.replace("nc", "parquet").replace("_combined", ""))
    # ds.to_netcdf(f"../voto_erddap_data_cache/{dataset_id}_combined.nc", "w")

# download_glider_dataset(
#    all_dataset_ids,  # all_dataset_ids may not actually be all datasets
#    # variables=variables,
# )

if utils.GDAC_data:
    allDatasetsGDAC = utils.load_allDatasets_GDAC()
    for dsid in allDatasetsGDAC.index:
        print("now downloading", dsid)
        e = ERDDAP(
            server="https://gliders.ioos.us/erddap",
            protocol="tabledap",
            response="nc",
        )
        e.dataset_id = dsid
        url = e.get_download_url()
        filepath = f"../voto_erddap_data_cache/{dsid}.nc"
        if os.path.isfile(filepath):
            print("file already exists, skip and continue")
            continue
        urlretrieve(url, filepath)
