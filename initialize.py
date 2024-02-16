import utils
import pathlib
import pickle
import xarray
from download_glider_data import utils as dutils
import time

t1 = time.perf_counter()
# all metadata exists for the metadata visualisation
all_metadata, _ = utils.load_metadata()

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
metadata = metadata.sort_values(by='time_coverage_start (UTC)')
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, all_datasets) # hacky

###### download actual data ##############################
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration',
	       #'methane_concentration',
           'cdom', 'backscatter_scaled', 'longitude']
dsids = ['../voto_erddap_data_cache/'+element.replace('nrt', 'delayed')+'.nc' for element in metadata.index]
#dsids = ['../voto_erddap_data_cache/'+element+'.nc' for element in metadata.index]

def drop_duplicates(obj, keep="first"):
    #if dim not in obj.dims:
    #    raise ValueError(f"'{dim}' not found in dimensions")
    indexes = {'time': ~obj.get_index('time').duplicated(keep=keep)}
    return obj.isel(indexes)

#print(metadata['time_coverage_start (UTC)'])
#print(dsids)
t2 = time.perf_counter()
print('metadata processed in', t2-t1)
# not sure/not tested if all the arguments are necessary...


ds = xarray.open_mfdataset(
    dsids,
    concat_dim='time2',
    combine='nested',
    parallel=True,
    #preprocess=drop_duplicates,
    #coords='minimal',
    #data_vars='minimal',
    coords="minimal",
    compat="override",
    #join="override",
    #data_vars=["temperature", "salinity"],
    decode_cf=False,
    decode_times=False,
    #parallel=False,#True,
    chunks={'time': 1e6},
    engine= "h5netcdf",#, "pynio", "pseudonetcdf", "zarr",
    #method='parallel',
    #chunks=160000,
    #compat='override'
    )


#datasets = [xarray.open_mfdataset(dsid) for dsid in dsids]
t3 = time.perf_counter()
print('read/process data:', t3-t2)
#ds = ds.unify_chunks().to_dask_dataframe().set_index('time')
#ds = print(ds['temperature'].head())#.compute()#.resample('5s').first().compute()
print('accesss data:', t3-t2)
t4 = time.perf_counter()
import pdb; pdb.set_trace();


