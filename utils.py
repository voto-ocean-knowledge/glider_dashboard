from erddapy import ERDDAP
import pprint
from ast import literal_eval
import pandas as pd
import numpy as np


def load_metadata():
    server = "https://erddap.observations.voiceoftheocean.org/erddap"
    e = ERDDAP(
        server=server,
        protocol="tabledap",
        response="csv",
    )
    e.dataset_id = "meta_metadata_table"
    metadata = e.to_pandas(
        index_col="datasetID",
        parse_dates=True,
    )

    e.dataset_id = "allDatasets"
    all_datasets = e.to_pandas(
        index_col="datasetID",
        parse_dates=True,
    )

    def obj_to_string(x):
        return pprint.pformat(x)

    def variable_exists(x, variable):
        return variable in x

    def basin_simplify(basin):
        if basin=='Eastern Gotland Basin, Northern Baltic Proper':
            return 'Eastern Gotland'
        if basin=='Northern Baltic Proper, Eastern Gotland Basin':
            return 'Eastern Gotland'
        elif basin=='Western Gotland Basin':
            return 'Western Gotland'
        elif basin=='Eastern Gotland Basin':
            return 'Eastern Gotland'
        elif basin=='Western Gotland Basin, Eastern Gotland Basin':
            return 'Western Gotland'
        elif basin=='Kattegat':
            return 'Skagerrak, Kattegat'
        elif basin=='Kattegat, Skagerrak':
            return 'Skagerrak, Kattegat'
        elif basin=='Skagerrak':
            return 'Skagerrak, Kattegat'
        elif basin=='Northern Baltic Proper':
            return 'Eastern Gotland'
            return 'Skagerrak, Kattegat'
        elif basin=='\\u00c3\\u0085land Sea':
            return 'Åland Sea'
        else:
            return basin

    metadata['optics_serial'] = metadata.optics_serial.apply(obj_to_string)
    metadata['irradiance_serial'] = metadata.irradiance_serial.apply(obj_to_string)
    metadata['altimeter_serial'] = metadata.altimeter_serial.apply(obj_to_string)
    metadata['glider_serial'] = metadata.glider_serial.apply(obj_to_string)
    metadata['basin'] = metadata.basin.apply(basin_simplify)

    # create list of all variables
    all_variables_set = set()
    menuentries = []
    menuentries_variables = []
    newmetadatacolumns = {}
    for index in range(0,len(metadata.index)):
        all_variables_set.update(literal_eval(metadata.iloc[index].variables))
    all_variables_set

    for variable in list(all_variables_set):
        newmetadatacolumns[variable+'_available'] = metadata.variables.apply(variable_exists, args=(variable,))
        menuentries.append({'label':variable+'_available', 'value':variable+'_available'})
        menuentries_variables.append({'label':variable,variable+'_available' 'value':variable})
    metadata = metadata.join(pd.DataFrame.from_dict(newmetadatacolumns))
    metadata['time_coverage_end (UTC)'] = pd.to_datetime(metadata['time_coverage_end (UTC)'])
    metadata['time_coverage_start (UTC)'] = pd.to_datetime(metadata['time_coverage_start (UTC)'])
    return metadata, all_datasets


def filter_metadata():
    # Better to return filtered DataFrame instead of IDs?
    mode = 'all' # 'nrt', 'delayed'
    metadata, all_datasets = load_metadata()
    metadata = metadata[
        (metadata['project']=='SAMBA') &
        (metadata['basin']=='Bornholm Basin') &
        (metadata['time_coverage_start (UTC)'].dt.year>2022) &
        #(metadata['time_coverage_start (UTC)'].dt.year>2022) &
        (metadata['time_coverage_start (UTC)'].dt.month>9)
        ]
    #for basins
    metadata = drop_overlaps(metadata)
    return metadata, all_datasets

def add_delayed_dataset_ids(metadata, all_datasets):
    nrt_dataset_ids = list(metadata.index)
    delayed_dataset_ids = [
        datasetid.replace('nrt', 'delayed') if datasetid.replace('nrt', 'delayed') in all_datasets.index else datasetid
        for datasetid in metadata.index]

    all_dataset_ids = nrt_dataset_ids+delayed_dataset_ids
    return all_dataset_ids#metadata.loc[all_dataset_ids]


def drop_overlaps(metadata):
    drop_overlap=True
    dropped_datasets = []
    for basin in ['Bornholm Basin', 'Skagerrak, Kattegat',
        'Western Gotland', 'Eastern Gotland', 'Åland Sea']:
        meta = metadata[metadata['basin']==basin]
        for index in range(0, len(meta)):
            glidercounter = 1
            maskedregions = []
            color = 'k'
            for index2 in range(0, index):
                r1 = dict(start=meta.iloc[index]['time_coverage_start (UTC)'],
                        end=meta.iloc[index]['time_coverage_end (UTC)'])
                r2 = dict(start=meta.iloc[index2]['time_coverage_start (UTC)'],
                        end=meta.iloc[index2]['time_coverage_end (UTC)'])
                latest_start = max(r1['start'], r2['start'])
                earliest_end = min(r1['end'], r2['end'])
                delta = (earliest_end - latest_start).days + 1
                overlap = max(0, delta)
                if overlap > 1:
                    glidercounter += 1
                    # if two Glider datasets are overlapping by more than a
                    # day, they are plotted in multiple rows...
                    if drop_overlap:
                        # ...and optionally dropped
                        dropped_datasets.append(meta.index[index])
                        color = 'red'

    print('dropping datasets {}'.format(dropped_datasets))
    metadata = metadata.drop(dropped_datasets)
    return metadata

def voto_seaexplorer_dataset(ds):
    """
    Adapts a VOTO xarray dataset, for example downloaded from the VOTO ERDAP
    server (https://erddap.observations.voiceoftheocean.org/erddap/index.html)
    to be used in GliderTools

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Dataset containing the all columns in the source file and dives column
    """
    ds = add_dive_column(ds)
    return ds


# this is a version were I only change the profile_nums, to try if no-concatenation helps with datashader performance
def voto_concat_datasets(datasets):
    """
    Concatenates multiple datasets along the time dimensions, profile_num
    and dives variable(s) are adapted so that they start counting from one
    for the first dataset and monotonically increase.

    Parameters
    ----------
    datasets : list of xarray.Datasets

    Returns
    -------
    xarray.Dataset
        concatenated Dataset containing all the data from the list of datasets
    """
    # in case the datasets have a different set of variables, emtpy variables are created
    # to allow for concatenation (concat with different set of variables leads to error)
    #mlist = [set(dataset.variables.keys()) for dataset in datasets]
    #allvariables = set.union(*mlist)
    #for dataset in datasets:
    #    missing_vars = allvariables - set(dataset.variables.keys())
    #    for missing_var in missing_vars:
    #        dataset[missing_var] = np.nan

    # renumber profiles, so that profile_num still is unique in concat-dataset
    for index in range(1, len(datasets)):
        datasets[index]["profile_num"] += (
            datasets[index - 1].copy()["profile_num"].max()
        )
    return datasets


def add_dive_column(ds):
    """add dive column to dataset

    Parameters:
    -----------
    ds: xarray.Dataset

    Returns:
    --------
    xarray.Dataset
        Dataset containing a dives column
    """
    ds["dives"] = np.where(ds.profile_direction == 1, ds.profile_num, ds.profile_num + 0.5)
    return ds

