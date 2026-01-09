import os.path
import pathlib
from urllib.request import urlretrieve

from erddapy import ERDDAP

import utils
from download_glider_data import utils as dutils

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
dutils.cache_dir = pathlib.Path("../voto_erddap_data_cache")
variables = [
    "temperature",
    "salinity",
    "depth",
    "potential_density",
    "profile_num",
    "profile_direction",
    "chlorophyll",
    "turbidity",
    "oxygen_concentration",
    "phycocyanin",
    "phycocyanin_tridente",
    "cdom",
    "fdom",
    "backscatter",
    "backscatter_scaled",
    "longitude",
    "latitude",
    "downwelling_PAR",
]
# dsids = ['../voto_erddap_data_cache/'+element+'.nc' for element in metadata.index]
dutils.download_glider_dataset(
    all_dataset_ids,  # all_dataset_ids may not actually be all datasets
    # variables=variables,
)

allDatasetsGDAC = utils.load_allDatasets_GDAC()
for dsid in allDatasetsGDAC.index:
    print("now downloading", dsid)
    e = ERDDAP(
        server="https://gliders.ioos.us/erddap",
        protocol="tabledap",
        response="parquet",
    )
    e.dataset_id = dsid
    url = e.get_download_url()
    filepath = f"../voto_erddap_data_cache/{dsid}.parquet"
    if os.path.isfile(filepath):
        print("file already exists, skip and continue")
        continue
    urlretrieve(url, filepath)
