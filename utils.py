import pprint
from ast import literal_eval

import pandas as pd
from erddapy import ERDDAP

project = "SAMBA"
basin = "Bornholm Basin"
year = 2024
month = 13
GDAC_data = False


def load_ERDDAP_Datasets(erddap_url, dataset_id, format):
    server = erddap_url
    e = ERDDAP(
        server=server,
        protocol="tabledap",
        response=format,
    )
    e.dataset_id = dataset_id
    return e.to_pandas(index_col="datasetID", date_format="%f")


def load_allDatasets_VOTO():
    allDatasetsVOTO = load_ERDDAP_Datasets(
        "https://erddap.observations.voiceoftheocean.org/erddap", "allDatasets", "csv"
    )
    allDatasetsVOTO["minTime (UTC)"] = pd.to_datetime(allDatasetsVOTO["minTime (UTC)"])
    allDatasetsVOTO["maxTime (UTC)"] = pd.to_datetime(allDatasetsVOTO["maxTime (UTC)"])
    # allDatasetsVOTO = allDatasetsVOTO[allDatasetsVOTO["minTime (UTC)"].dt.year == year]
    # allDatasetsVOTO = allDatasetsVOTO[allDatasetsVOTO["minTime (UTC)"].dt.month < month]
    return allDatasetsVOTO


def load_allDatasets_GDAC():
    # I believe I want to plot these datasets actually against precise_time and not against time (which really is profile averaged time)
    allDatasetsGDAC = load_ERDDAP_Datasets(
        "https://gliders.ioos.us/erddap", "allDatasets", "csv"
    )
    allDatasetsGDAC[
        allDatasetsGDAC["cdm_data_type"] == "TrajectoryProfile"
    ]  # filter out allDatasets table and other non-glider datasets
    # allDatasetsGDAC = allDatasetsGDAC
    allDatasetsGDAC["minTime (UTC)"] = pd.to_datetime(allDatasetsGDAC["minTime (UTC)"])
    allDatasetsGDAC["maxTime (UTC)"] = pd.to_datetime(allDatasetsGDAC["maxTime (UTC)"])
    allDatasetsGDAC = allDatasetsGDAC[allDatasetsGDAC["minTime (UTC)"].dt.year == year]
    allDatasetsGDAC = allDatasetsGDAC[allDatasetsGDAC["minTime (UTC)"].dt.month < month]
    allDatasetsGDAC.drop("maracoos_05-20240801T1650-delayed")
    allDatasetsGDAC = allDatasetsGDAC.iloc[0:90]
    allDatasetsGDAC = allDatasetsGDAC[
        allDatasetsGDAC["institution"] != "C-PROOF"
    ]  # THIS is just here because C-PROOF files currently don't download from GDAC
    # ]C-PROOF
    return allDatasetsGDAC


def load_metadata_VOTO():
    # e.dataset_id = "meta_metadata_table"
    # metadata = e.to_pandas(index_col="datasetID", date_format="%f")

    metadata = load_ERDDAP_Datasets(
        "https://erddap.observations.voiceoftheocean.org/erddap",
        "meta_metadata_table",
        "csv",
    )

    def obj_to_string(x):
        return pprint.pformat(x)

    def variable_exists(x, variable):
        return variable in x

    def basin_simplify(basin):
        if type(basin) == float:
            return "undefined"  # Could happen that basin is empty, resulting in basin == np.nan here
        if basin.split(",")[0] == "Bornholm Basin":
            return "Bornholm Basin"
        elif basin.split(",")[0] == "Eastern Gotland Basin":
            return "Eastern Gotland"
        elif basin in [
            "Northern Baltic Proper, Eastern Gotland Basin",
            "Northern Baltic Proper",
        ]:
            return "Eastern Gotland"
        elif basin.split(",")[0] in ["Skagerrak", "Kattegat"]:
            return "Skagerrak, Kattegat"
        elif basin.split(",")[0] == "Western Gotland Basin":
            return "Western Gotland"
        elif basin.split(",")[0] in [
            "Åland Sea",
            "\\u00c5land Sea",
            "\\u00c3\\u0085land Sea",
        ]:
            return "Åland Sea"
        else:
            return basin

    metadata["optics_serial"] = metadata.optics_serial.apply(obj_to_string)
    metadata["irradiance_serial"] = metadata.irradiance_serial.apply(obj_to_string)
    metadata["altimeter_serial"] = metadata.altimeter_serial.apply(obj_to_string)
    metadata["glider_serial"] = metadata.glider_serial.apply(obj_to_string)
    metadata["basin"] = metadata.basin.apply(basin_simplify)
    metadata["time_coverage_end (UTC)"] = pd.to_datetime(
        metadata["time_coverage_end (UTC)"]
    )
    metadata["time_coverage_start (UTC)"] = pd.to_datetime(
        metadata["time_coverage_start (UTC)"]
    )

    # allDatasets["minTime (UTC)"] = pd.to_datetime(allDatasets["minTime (UTC)"])
    # allDatasets["maxTime (UTC)"] = pd.to_datetime(allDatasets["maxTime (UTC)"])

    # import pdb

    # pdb.set_trace()
    return metadata  # , allDatasets


def variable_exists(x, variable):
    # import pdb; pdb.set_trace();
    return variable in x


def create_available_variables_columns(metadata):
    # create list of all variables
    all_variables_set = set()
    menuentries = []
    menuentries_variables = []
    newmetadatacolumns = {}
    for index in range(0, len(metadata.index)):
        all_variables_set.update(literal_eval(metadata.iloc[index].variables))
    all_variables_set

    for variable in list(all_variables_set):
        newmetadatacolumns[variable + "_available"] = metadata.variables.apply(
            variable_exists, args=(variable,)
        )
        menuentries.append(
            {"label": variable + "_available", "value": variable + "_available"}
        )
        # menuentries_variables.append({'label':variable,variable+'_available' 'value':variable})
    metadata = metadata.join(pd.DataFrame.from_dict(newmetadatacolumns))
    return metadata


def filter_metadata():
    # Better to return filtered DataFrame instead of IDs?
    mode = "all"  # 'nrt', 'delayed'
    metadata = load_metadata_VOTO()

    """
    metadata = metadata[
        (metadata["project"] == project)
        & (metadata["basin"] == basin)
        & (metadata["time_coverage_start (UTC)"].dt.year == year)
        & (metadata["time_coverage_start (UTC)"].dt.month < month)
    ]
    """
    # Terrible style here.
    # all_datasets = allDatasets[allDatasets["minTime (UTC)"].dt.year == year]
    # all_datasets = all_datasets[
    #     all_datasets["institution"] != "Voice of the Ocean Foundation"
    # ]
    # all_datasets =
    # all_datasets = all_datasets[
    #    all_datasets["institution"] == "Skidaway Institute of Oceanography"
    # ]C-PROOF
    # for basins
    # metadata = drop_overlaps(metadata)
    return metadata  # , all_datasets


def add_delayed_dataset_ids(metadata, all_datasets):
    nrt_dataset_ids = list(metadata.index)
    delayed_dataset_ids = [
        datasetid.replace("nrt", "delayed")
        if datasetid.replace("nrt", "delayed") in all_datasets.index
        else datasetid
        for datasetid in metadata.index
    ]

    all_dataset_ids = nrt_dataset_ids + delayed_dataset_ids
    return all_dataset_ids  # metadata.loc[all_dataset_ids]


def drop_overlaps(metadata):
    drop_overlap = True
    dropped_datasets = []
    for basin in [
        "Bornholm Basin",
        "Skagerrak, Kattegat",
        "Western Gotland",
        "Eastern Gotland",
        "Åland Sea",
    ]:
        meta = metadata[metadata["basin"] == basin]
        for index in range(0, len(meta)):
            glidercounter = 1
            maskedregions = []
            color = "k"
            for index2 in range(0, index):
                r1 = dict(
                    start=meta.iloc[index]["time_coverage_start (UTC)"],
                    end=meta.iloc[index]["time_coverage_end (UTC)"],
                )
                r2 = dict(
                    start=meta.iloc[index2]["time_coverage_start (UTC)"],
                    end=meta.iloc[index2]["time_coverage_end (UTC)"],
                )
                latest_start = max(r1["start"], r2["start"])
                earliest_end = min(r1["end"], r2["end"])
                delta = (earliest_end - latest_start).days + 1
                overlap = max(0, delta)
                if overlap > 1:
                    glidercounter += 1
                    # if two Glider datasets are overlapping by more than a
                    # day, they are plotted in multiple rows...
                    if drop_overlap:
                        # ...and optionally dropped
                        dropped_datasets.append(meta.index[index])
                        color = "red"

    # print('dropping datasets {}'.format(dropped_datasets))
    metadata = metadata.drop(dropped_datasets)
    return metadata


def drop_overlaps_fast(metadata):
    with pd.option_context("mode.chained_assignment", None):
        metadata["duration"] = (
            metadata["time_coverage_end (UTC)"] - metadata["time_coverage_start (UTC)"]
        )
        metadata["startdate"] = metadata["time_coverage_start (UTC)"].dt.date
    remaining = (
        metadata.sort_values(["startdate", "duration"], ascending=[True, False])[
            ["startdate"]
        ]
        .drop_duplicates()
        .index
    )
    # import pdb; pdb.set_trace();
    return metadata.loc[remaining]


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
    # mlist = [set(dataset.variables.keys()) for dataset in datasets]
    # allvariables = set.union(*mlist)
    # for dataset in datasets:
    #    missing_vars = allvariables - set(dataset.variables.keys())
    #    for missing_var in missing_vars:
    #        dataset[missing_var] = np.nan

    # renumber profiles, so that profile_num still is unique in concat-dataset
    for index in range(1, len(datasets)):
        datasets[index]["profile_num"] += (
            datasets[index - 1].copy()["profile_num"].max()
        )
    return datasets


def voto_concat_datasets2(datasets):
    import polars as pl

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
    for index in range(1, len(datasets)):
        datasets[index] = datasets[index].with_columns(
            pl.col("profile_num") + (index * 10000)
        )  # datasets[index - 1].select(pl.col("profile_num")).max()  # .collect())
    ds = pl.concat([data for data in datasets], how="diagonal_relaxed")
    # ds = add_dive_column(ds)

    return ds


# def dask_add_dives(profile_nu):


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
    # ds["dives"] = np.where(ds.profile_direction == 1, ds.profile_num, ds.profile_num + 0.5)
    ds["dives"] = ds.profile_num.where(ds.profile_direction == 1, ds.profile_num + 0.5)
    return ds
