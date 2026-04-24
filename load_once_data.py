import numpy as np
import pandas as pd
import param
import polars as pl
import polars.selectors as cs

import utils

all_metadata = utils.load_metadata_VOTO()

###### filter metadata to prepare download ##############
metadata = utils.filter_metadata()

metadata = metadata.drop(
    ["nrt_SEA067_M15", "nrt_SEA079_M14", "nrt_SEA061_M63"], errors="ignore"
)  # temporary data inconsistency
metadata["time_coverage_start (UTC)"] = metadata[
    "time_coverage_start (UTC)"
].dt.tz_convert(None)
metadata["time_coverage_end (UTC)"] = metadata["time_coverage_end (UTC)"].dt.tz_convert(
    None
)
allDatasetsVOTO = utils.load_allDatasets_VOTO()
if utils.GDAC_data:
    allDatasetsGDAC = utils.load_allDatasets_GDAC()
    allDatasets = pd.concat([allDatasetsVOTO, allDatasetsGDAC])
else:
    allDatasets = allDatasetsVOTO

dsdict = {}

# HERE I MUST DIFFERENTIATE INTO TWO DATASETS: ONE COMPLETE METADATA FOR THE METADASHBOARD,
# AND ONE/TWO DATASETS THAT ARE FILTERED AND SHWON IN THE DASHBOARD
# all_dataset_names = list(all_datasets.index) + list(metadata.index)

all_dataset_names = set(allDatasetsVOTO.index).intersection(
    [element.replace("nrt", "delayed") for element in metadata.index]
)
all_dataset_names = list(all_dataset_names)
all_dataset_names += list(
    metadata.index
)  # Add nrt data because I currently use it for statistics
if utils.GDAC_data:
    all_dataset_names += list(allDatasetsGDAC.index)

all_dataset_names = list(all_dataset_names) + [
    dataset_name + "_small" for dataset_name in all_dataset_names
]

fDs = allDatasets.loc[[name for name in all_dataset_names if "_small" not in name]]
fDs["minTime (UTC)"] = fDs["minTime (UTC)"].dt.tz_localize(None)
fDs["maxTime (UTC)"] = fDs["maxTime (UTC)"].dt.tz_localize(None)
allDatasets["minTime (UTC)"] = allDatasets["minTime (UTC)"].dt.tz_localize(None)
allDatasets["maxTime (UTC)"] = allDatasets["maxTime (UTC)"].dt.tz_localize(None)

for dsid in list(allDatasetsVOTO.index) + [
    id + "_small" for id in allDatasetsVOTO.index
]:
    if dsid not in all_dataset_names:
        continue
    dsdict[dsid] = pl.scan_parquet(f"../voto_erddap_data_cache/{dsid}.parquet")

if utils.GDAC_data:
    for dsid in list(allDatasetsGDAC.index) + [
        id + "_small" for id in allDatasetsGDAC.index
    ]:
        dsdict[dsid] = pl.scan_parquet(f"../voto_erddap_data_cache/{dsid}.parquet")
        dsdict[dsid] = (
            dsdict[dsid]
            .drop(cs.string())
            .with_columns(
                pl.col("time").dt.cast_time_unit("ns").dt.replace_time_zone(None)
                # .cast(pl.Float32, strict=False) # if this is activated, time is cast into float32, which leads to bugs in keeping x-range across parameter changes
            )
            .rename({"profile_id": "profile_num"})
        )

# variables_selectable = ["time", "depth", "temperature", "pressure", "salinity"]
variables_selectable = (
    pl.concat(dsdict.values(), how="diagonal_relaxed").collect_schema().names()
)
variables_selectable.sort()  # inplace function

####### specify global plot variables ####################
# df.index = cudf.to_datetime(df.index)


def plot_limits(plot, element):
    # function to limit user interaction. Can prevent crashes
    # caused by data before 0AD, data out of range...
    plot.handles["x_range"].min_interval = np.timedelta64(2, "h")
    plot.handles["x_range"].max_interval = np.timedelta64(
        int(5 * 3.15e7), "s"
    )  # 5 years
    plot.handles["y_range"].min_interval = 10
    plot.handles["y_range"].max_interval = 500


def mld_profile(df, variable, thresh, ref_depth, verbose=True):
    exception = False
    divenum = df["profile_num"].first()  # df.index[0]
    ptime = df["time"].first()
    # df["depth"] = df["depth"].neg()
    df = df.with_columns(pl.col("depth").neg())
    df = df.drop_nulls(subset=[variable, "depth"])

    if len(df) == 0:
        mld = np.nan
        exception = True
        message = """no observations found for specified variable in dive {}
                """.format(divenum)
    elif np.nanmin(np.abs(df["depth"] + ref_depth)) > 5:
        exception = True
        message = """no observations within 5 m of ref_depth for dive {}
                """.format(divenum)
        mld = np.nan
    else:
        # not using direction because it is not present at GDAC
        # direction = df["profile_direction"].first()
        direction = 1 if (df["depth"].first() > df["depth"].last()) else -1
        # create arrays in order of increasing depth
        var_arr = df[variable][:: int(direction)]
        depth = df["depth"][:: int(direction)]
        # get index closest to ref_depth
        i = np.nanargmin(np.abs(depth + ref_depth))
        # create difference array for threshold variable
        dd = var_arr - var_arr[int(i)]
        # mask out all values that are shallower then ref_depth
        dd[depth > ref_depth] = np.nan
        # get all values in difference array within treshold range
        mixed = dd.filter(abs(dd) > thresh)
        if len(mixed) > 0:
            idx_mld = np.argmax(abs(dd) > thresh)
            mld = depth[int(idx_mld)]
        else:
            exception = True
            mld = np.nan
            message = """threshold criterion never true (all mixed or \
                shallow profile) for profile {}""".format(divenum)
    if verbose and exception:
        print(message)
    return pl.DataFrame({"mld": [-mld], "time": [ptime]})


def create_cbar_range(variable):
    return param.Range(
        default=(
            0,
            1,
            # dictionaries.ranges_dict[variable][0],
            # dictionaries.ranges_dict[variable][1],
        ),  # this is not respected anyway, but below in redefinition
        doc=f"Cbar limits for {variable}",
        label=variable,
        precedence=-10,
    )


cbar_range_sliders = {
    f"pick_cbar_range_{variable}": create_cbar_range(variable)
    for variable in variables_selectable
}
