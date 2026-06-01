import os.path
import shutil

# import urllib.request
from urllib.request import urlretrieve

import numpy as np
import polars as pl
import urllib3
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
# allDatasetsGDAC = utils.load_allDatasets_GDAC()
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, allDatasetsVOTO)  # hacky

###### download actual data ##############################
# cache_dir = pathlib.Path("../voto_erddap_data_cache")
# dsids = ['../voto_erddap_data_cache/'+element+'.nc' for element in metadata.index]
# import pdb
# pdb.set_trace()
print(all_dataset_ids)
""""
for dataset_id in all_dataset_ids:
    url = f"https://erddap.observations.voiceoftheocean.org/erddap/files/{dataset_id}/mission_timeseries.nc"
    file_Path = f"../voto_erddap_data_cache/{dataset_id}.nc"
    if os.path.isfile(file_Path):
        print(f"{file_Path} already exists, skip")
        # pass and look for ADCP file below
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
                print(f"no adcp data for {dataset_id}")
"""

for dataset_id in all_dataset_ids:
    # import pdb
    # pdb.set_trace()

    print("now downloading", dataset_id)
    # planned: iterate over 5 day periods to help with annoying low specced ERDDAP servers
    print(
        allDatasetsVOTO.loc[dataset_id]["minTime (UTC)"],
        allDatasetsVOTO.loc[dataset_id]["maxTime (UTC)"],
    )
    e = ERDDAP(
        server="https://erddap.observations.voiceoftheocean.org/erddap/",
        protocol="tabledap",
        response="nc",
    )
    e.dataset_id = dataset_id
    url = e.get_download_url()
    filepath = os.path.join(utils.cache_location, f"{dataset_id}.nc")
    if os.path.isfile(filepath):
        print("file already exists, skip and continue")
        continue
    urlretrieve(url, filepath)
    if dataset_id[0:7] == "delayed":
        dsid = dataset_id.replace("delayed_", "")
        url = f"https://erddap.observations.voiceoftheocean.org/erddap/files/gliderad2cp_files/{dsid}_adcp_proc.nc"
        file_Path_adcp = f"../voto_erddap_data_cache/{dsid}_adcp_proc.nc"
        if os.path.isfile(file_Path_adcp):
            print(f"{file_Path_adcp} already exists, skip")
        else:
            try:
                urllib.request.urlretrieve(url, file_Path_adcp)
            except Exception as e:
                print(f"Error for {dataset_id}: {e}")

for dataset_id in all_dataset_ids:
    # if not (dataset_id[0:7] == "delayed"):
    #    # we do not have/provide VOTO nrt ADCP data
    #    continue
    if os.path.isfile(os.path.join(utils.cache_location, f"{dataset_id}.parquet")):
        print(
            f"combined {dataset_id} data/adcp file already exists, skip"
        )  # not necessarily :/
        continue
    print(f"combining {dataset_id} variables with adcp file")
    file_Path = os.path.join(utils.cache_location, f"{dataset_id}.nc")
    dsid = dataset_id.replace("delayed_", "")
    file_Path_adcp = os.path.join(utils.cache_location, f"{dsid}_adcp_proc.nc")

    ds = xarray.open_mfdataset(file_Path, drop_variables="ad2cp_time")
    # import pdb
    # pdb.set_trace()
    if "time" not in ds["depth"].dims:
        # unfortunately some datasets come with "row" dimension from ERDDAP currently,
        # instead of "time" how it is meant to be (?).
        ds = ds.swap_dims({"row": "time"})

    ds = ds.drop_duplicates(dim="time").load()
    if os.path.isfile(file_Path_adcp):
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
    else:
        print(f"no adcp data for {dsid} found, skip combining")
        # continue
    # ds2.sortby('depth','profile_index').sel(profile_index=np.datetime64('2024-01-10'), method='nearest')#

    df = ds.to_pandas().sort_index()
    if df.index.diff().mean() < np.timedelta64(600, "ms"):
        df = df.resample("1s").mean()
    df = pl.from_dataframe(df.astype(np.float32))
    df.write_parquet(os.path.join(utils.cache_location, f"{dataset_id}.parquet"))
    df = df.filter(pl.col("profile_num") % 10 == 0)
    df.write_parquet(
        os.path.join(utils.cache_location, f"{dataset_id}_small.parquet")
    )  # "file.replace(".nc", "_small.parquet").replace("_combined", ""))

    # file.replace("nc", "parquet").replace("_combined", ""))
    # ds.to_netcdf(f"../voto_erddap_data_cache/{dataset_id}_combined.nc", "w")

# download_glider_dataset(
#    all_dataset_ids,  # all_dataset_ids may not actually be all datasets
#    # variables=variables,
# )

if utils.GDAC_data:
    allDatasetsGDAC = utils.load_allDatasets_GDAC()
    try:
        allDatasetsGDAC.drop(
            index="allDatasets"
        )  # "allDatasets aggregation table on ERDDAP"
    except:
        pass
    print(allDatasetsGDAC)
    for dsid in allDatasetsGDAC.index:
        filepath = os.path.join(utils.cache_location, f"{dsid}.nc")
        if os.path.isfile(filepath):
            print(f"file {filepath} already exists, skip")
            continue
        else:
            print(f"file {filepath} still needs downloading!")
        print("now downloading", dsid)
        e = ERDDAP(
            server="https://gliders.ioos.us/erddap",
            protocol="tabledap",
            response="nc",
        )
        e.dataset_id = dsid

        # import pdb; pdb.set_trace();
        # continue
        info_url = e.get_info_url(dataset_id=dsid, response="csv")
        dsmeta = pl.read_csv(info_url)
        # dsmeta.filter(pl.col('Row Type')=='variable')
        number_of_variables = (
            dsmeta.filter(pl.col("Row Type") == "variable")
            .count()
            .select("Variable Name")
            .item()
        )
        if number_of_variables > 200:
            print(f"unreasonable many variables in {dsid}:{number_of_variables}, skip")
            continue

        tstart = allDatasetsGDAC.loc[dsid]["minTime (UTC)"]
        tend = allDatasetsGDAC.loc[dsid]["maxTime (UTC)"]
        ds_time_slices = []
        counter = 0
        url = e.get_download_url()
        print(url)
        filepath = os.path.join(utils.cache_location, f"{dsid}.nc")
        if os.path.isfile(filepath):
            print(f"file {filepath} already exists, skip and continue")
            continue

        timeout = urllib3.Timeout(connect=120, read=300)
        # http = urllib3.PoolManager(timeout=default_timeout)
        c = urllib3.PoolManager(timeout=timeout)

        with (
            c.request("GET", url, preload_content=False) as resp,
            open(filepath, "wb") as out_file,
        ):
            shutil.copyfileobj(resp, out_file)

        resp.release_conn()  # not 100% sure this is required though

        # reso = urllib2.urlopen(url)
        # with open(filepath, "wb") as f:
        #    f.write(resp.read())
        # urlretrieve(url, filepath)
        print(f"direct download of {filepath} was sucessful")
