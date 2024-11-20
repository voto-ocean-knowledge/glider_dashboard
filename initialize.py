import utils
import pathlib
import pickle
from download_glider_data import utils as dutils

# all metadata exists for the metadata visualisation
all_metadata, _ = utils.load_metadata()

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
metadata = metadata.sort_values(by='time_coverage_start (UTC)')
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, all_datasets) # hacky

###### download actual data ##############################
dutils.cache_dir = pathlib.Path('../voto_erddap_data_cache')
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration', 'phycocyanin', 'phycocyanin_tridente',
           'cdom', 'backscatter_scaled', 'longitude', 'latitude']
# dsids = ['../voto_erddap_data_cache/'+element+'.nc' for element in metadata.index]
dsdict = dutils.download_glider_dataset(
    all_dataset_ids, # all_dataset_ids are actually the filtered dataserts from utils.filter_metadata...
    metadata,
    variables=variables)


