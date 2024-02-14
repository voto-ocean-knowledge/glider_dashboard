import utils
import pathlib
import pickle
from download_glider_data import utils as dutils

# all metadata exists for the metadata visualisation
all_metadata, _ = utils.load_metadata()

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, all_datasets) # hacky

###### download actual data ##############################
dutils.cache_dir = pathlib.Path('../voto_erddap_data_cache')
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration',
	       #'methane_concentration',
           'cdom', 'backscatter_scaled', 'longitude']
dsdict = dutils.download_glider_dataset(all_dataset_ids, metadata,
                                        variables=variables)
print('all datasets loaded, caching them to pickle file...')
# open a file, where you ant to store the data
#file = open('cached_data_dictionary.pickle', 'wb')
#pickle.dump(dsdict, file)
#file.close()
